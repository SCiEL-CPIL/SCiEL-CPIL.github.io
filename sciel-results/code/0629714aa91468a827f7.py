import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_START = time.time()
_DEADLINE = 600.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

# Precompute grid and projection indices
_x = np.arange(_N)
_y = np.arange(_N)
_X, _Y = np.meshgrid(_x, _y, indexing='xy')

# Projections
_val_d = _X - _Y
_val_m0 = _X
_val_m1 = _X + _Y
_val_m2 = _X + 2 * _Y

# Map to non-negative integers for bincount
# X-Y: -9..9 -> 0..18
_min_d = _val_d.min()
_inv_d = (_val_d - _min_d).astype(int)
_size_d = int(_val_d.max() - _min_d) + 1

# X: 0..9
_inv_m0 = _val_m0.astype(int)
_size_m0 = int(_val_m0.max()) + 1

# X+Y: 0..18
_inv_m1 = _val_m1.astype(int)
_size_m1 = int(_val_m1.max()) + 1

# X+2Y: 0..27
_inv_m2 = _val_m2.astype(int)
_size_m2 = int(_val_m2.max()) + 1

def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _objective(params):
    A, B, C = params
    log_w = A * _X + B * _Y + C * _X * _Y
    # Shift for numerical stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        return 0.0
    P = w / total
    
    # Project and compute entropy
    # D: X - Y
    p_d = np.bincount(_inv_d.ravel(), weights=P.ravel(), minlength=_size_d)
    D = _shannon_bits(p_d)
    
    # M components
    p_m0 = np.bincount(_inv_m0.ravel(), weights=P.ravel(), minlength=_size_m0)
    H0 = _shannon_bits(p_m0)
    
    p_m1 = np.bincount(_inv_m1.ravel(), weights=P.ravel(), minlength=_size_m1)
    H1 = _shannon_bits(p_m1)
    
    p_m2 = np.bincount(_inv_m2.ravel(), weights=P.ravel(), minlength=_size_m2)
    H2 = _shannon_bits(p_m2)
    
    M = max(H0, H1, H2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def _objective_grad(params):
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0, np.zeros(3)
    
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        val_minus = _objective(params_minus)
        
        if not np.isfinite(val_plus) or not np.isfinite(val_minus):
            grad[i] = 0.0
        else:
            grad[i] = (val_plus - val_minus) / (2 * eps)
            
    return val, grad

def arithmetic_kakeya_distribution() -> np.ndarray:
    # Initial guess: uniform
    x0 = [0.0, 0.0, 0.0]
    
    def _neg_obj(params):
        val = _objective(params)
        if not np.isfinite(val):
            return 0.0
        return -val
        
    def _neg_obj_grad(params):
        val, grad = _objective_grad(params)
        if not np.isfinite(val):
            return 0.0, np.zeros(3)
        return -val, -grad
        
    best_params = x0
    best_val = _objective(best_params)
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    starts = [x0]
    # Structured starts based on manual exploration of the landscape
    # We want to boost corners to increase D (spread of X-Y)
    # and potentially concentrate mass to reduce M (especially H2)
    
    # Try to make X-Y uniform: boost (0,9) and (9,0) corners?
    # X-Y at (0,9) is -9. At (9,0) is 9.
    # To boost these, we need A*X + B*Y + C*XY to be high at (0,9) and (9,0).
    # At (0,9): 9B + 0.
    # At (9,0): 9A + 0.
    # So A > 0, B > 0 might help?
    # But this also affects X+2Y.
    # Let's just use a grid of starts.
    
    for A in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        for B in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            for C in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
                starts.append([A, B, C])
                
    # Random starts
    for _ in range(50):
        A = rng.uniform(-3, 3)
        B = rng.uniform(-3, 3)
        C = rng.uniform(-3, 3)
        starts.append([A, B, C])
        
    # Optimize from each start
    for s in starts:
        if _remaining() < 30:
            break
        try:
            res = minimize(_neg_obj_grad, s, method='L-BFGS-B', jac=True,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerances
    if _remaining() > 20:
        try:
            res = minimize(_neg_obj_grad, best_params, method='L-BFGS-B', jac=True,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < -best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Generate final P
    A, B, C = best_params
    log_w = A * _X + B * _Y + C * _X * _Y
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = w / total
        
    # Final safety checks
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s < 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P
