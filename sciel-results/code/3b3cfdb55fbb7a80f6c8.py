import numpy as np
from scipy.fft import dct, idct
import time

N = 50

def _evaluate(h):
    """Compute the objective value for given heights h."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # l2 = sum (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2) for i in 0..m
    y2 = y[1:] ** 2  # y[i+1]^2 for i in 0..m
    y1 = y[:-1] ** 2  # y[i]^2 for i in 0..m
    ycross = y[:-1] * y[1:]  # y[i]*y[i+1] for i in 0..m
    l2 = (w / 3.0) * np.sum(y1 + ycross + y2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _dct_to_heights(coeffs):
    """Convert DCT coefficients to nonnegative heights."""
    h = idct(coeffs, type=2, norm='ortho')
    return np.abs(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with the seed
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # DCT parametrization: use low-dimensional coefficient vector
    # h = |idct(coeffs)|, where coeffs has dimension K
    # We'll optimize over K coefficients
    
    def _objective_from_coeffs(coeffs):
        h = _dct_to_heights(coeffs)
        if np.any(~np.isfinite(h)):
            return -1.0
        return -_evaluate(h)  # negate for minimization
    
    # Try different dimensionality K
    for K in [5, 8, 10, 12, 15]:
        if time.time() > deadline - 30:
            break
        
        # Initialize coefficients: DCT of the current best
        init_coeffs = dct(best_h, type=2, norm='ortho')
        # Zero out high-frequency components
        init_coeffs[K:] = 0.0
        
        # Use scipy minimize
        from scipy.optimize import minimize, differential_evolution
        
        # First try L-BFGS-B
        try:
            res = minimize(_objective_from_coeffs, init_coeffs, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10})
            if time.time() > deadline - 20:
                break
            if res.fun < -best_val:
                h_new = _dct_to_heights(res.x)
                val_new = _evaluate(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
        except Exception:
            pass
        
        # Try Nelder-Mead
        try:
            res = minimize(_objective_from_coeffs, init_coeffs, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() > deadline - 10:
                break
            if res.fun < -best_val:
                h_new = _dct_to_heights(res.x)
                val_new = _evaluate(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
        except Exception:
            pass
    
    # Local refinement: direct optimization on h with non-negativity
    if time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            
            def _obj_direct(h):
                h = np.maximum(h, 0)
                return -_evaluate(h)
            
            res = minimize(_obj_direct, best_h, method='L-BFGS-B',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < -best_val:
                h_new = np.maximum(res.x, 0)
                val_new = _evaluate(h_new)
                if val_new > best_val:
                    best_val = val_new
                    best_h = h_new
        except Exception:
            pass
    
    # Ensure nonnegativity and finiteness
    best_h = np.maximum(best_h, 0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
