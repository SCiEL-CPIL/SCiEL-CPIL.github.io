"""Second autocorrelation inequality - optimized with refined soft-max and robust local polish."""
import numpy as np
import time
from scipy.optimize import minimize, check_grad

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    # y_sq[i] = y[i]^2 + y[i]y[i+1] + y[i+1]^2
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    """Softmax parameterization for non-negative, normalized h."""
    # Shift for numerical stability
    p = params - np.max(params)
    e = np.exp(p)
    h = e / e.sum()
    return h

def _logsumexp(x, beta):
    """Compute log-sum-exp of x with temperature 1/beta."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0: return -np.inf
    xm = np.max(x)
    if xm < -1e300:
        return -np.inf
    # log(sum(exp(-beta * (x - xm)))) / beta + xm
    # = log(sum(exp(-beta * x + beta * xm))) / beta + xm
    # = log(sum(exp(-beta * (x - xm)))) / beta + xm
    # Let z = x - xm. log(sum(exp(-beta * z))) = log(exp(0) + ...)
    # To avoid underflow/overflow issues with large beta:
    # sum(exp(-beta * z)) where z <= 0.
    # exp(-beta * z) = exp(beta * (xm - x))
    # Wait, standard logsumexp is log(sum(exp(x))).
    # Here we want log(sum(exp(beta * x))) / beta.
    # This is the "soft max" approximation.
    # log(sum(exp(beta * x))) = beta * xm + log(sum(exp(beta * (x - xm))))
    # So logsumexp_scaled = xm + (1/beta) * log(sum(exp(beta * (x - xm))))
    
    z = beta * (x - xm)
    # max z is 0
    s = np.sum(np.exp(z))
    if s == 0: return -np.inf
    return xm + np.log(s) / beta

def _soft_value(params, beta):
    """
    Smooth surrogate for the objective.
    Replaces max|g| with soft-max(|g|).
    """
    h = _h_from_params(params)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g)
    
    # Soft max of |g|
    soft_inf = _logsumexp(abs_g, beta)
    # exp(soft_inf) is the soft max value
    if norm_1 < 1e-300:
        return 0.0
    
    val = l2 / (norm_1 * np.exp(soft_inf))
    return val

def _neg_soft_value(params, beta):
    return -_soft_value(params, beta)

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

def _neg_value_exact(h):
    """Objective function taking h directly for coordinate descent."""
    return -_compute_value(h)

def construct():
    t0 = time.time()
    deadline = t0 + 148.0  # 2s safety margin

    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    def update_best(h_new, v_new):
        nonlocal best_h, best_val
        if v_new > best_val:
            best_val = v_new
            best_h = h_new.copy()

    def try_optimize(params0, method='L-BFGS-B', maxiter=1000, use_soft=True, beta=50.0, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        
        if use_soft:
            objective = lambda p: _neg_soft_value(p, beta)
        else:
            objective = _neg_value
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(objective, params0, method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            elif method == 'Nelder-Mead':
                res = minimize(objective, params0, method='Nelder-Mead',
                               options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            elif method == 'Powell':
                res = minimize(objective, params0, method='Powell',
                               options={'maxiter': maxiter, 'xtol': 1e-8, 'ftol': 1e-10})
            else:
                return
                
            h_new = _h_from_params(res.x)
            v_new = _compute_value(h_new)
            update_best(h_new, v_new)
            
            # Also try the exact function result if soft was used
            if use_soft:
                # Sometimes the soft max leads to a point that is slightly better in exact terms
                # or we can polish from here
                pass
                
        except Exception as e:
            pass

    def coordinate_refine(h, n_iter=20):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float).copy()
        h = np.maximum(h, 1e-15)
        h = h / h.sum()
        v = _compute_value(h)
        update_best(h, v)
        
        # Adaptive step sizes
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for _ in range(n_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for delta_frac in step_sizes:
                    delta = delta_frac * h[i]
                    if delta < 1e-15: continue
                    
                    # Try increasing
                    h_new = h.copy()
                    h_new[i] += delta
                    h_new = np.maximum(h_new, 1e-15)
                    h_new = h_new / h_new.sum()
                    v_new = _compute_value(h_new)
                    if v_new > best_val + 1e-12:
                        h = h_new
                        best_val = v_new
                        improved = True
                        break
                        
                    # Try decreasing
                    h_new = h.copy()
                    h_new[i] -= delta
                    if h_new[i] < 1e-15:
                        h_new[i] = 1e-15
                    h_new = h_new / h_new.sum()
                    v_new = _compute_value(h_new)
                    if v_new > best_val + 1e-12:
                        h = h_new
                        best_val = v_new
                        improved = True
                        break
            if not improved:
                break
        return best_h, best_val

    # Generate diverse initial candidates
    candidates = []
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)

    # Uniform
    candidates.append(np.ones(N))
    
    # Gaussians with varying widths
    for a in [0.5, 1, 2, 4, 8, 16, 32]:
        candidates.append(np.exp(-a * x**2))
        
    # Bimodal
    for c in [0.2, 0.3, 0.4, 0.5]:
        for a in [4, 8, 16, 32]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
            
    # Polynomials (tent-like)
    for k in [1.5, 2, 3, 4, 6]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
        
    # Cosine bumps
    candidates.append(np.maximum(0, np.cos(np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(2 * np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(1.5 * np.pi * xh)))
    
    # Plateaus
    for start, end in [(5, 45), (10, 40), (15, 35), (0, 50), (20, 30)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        # Smooth edges slightly
        candidates.append(plat)
        
    # Random smooth (Fourier)
    rng = np.random.default_rng(42)
    for _ in range(15):
        coeffs = rng.standard_normal(5)
        h_f = np.ones(N)
        for k in range(1, 6):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh) * (1/k)
        h_f = np.abs(h_f) + 0.1
        candidates.append(h_f)
        
    # Random sparse
    for _ in range(10):
        h_r = np.zeros(N)
        idx = rng.choice(N, size=5, replace=False)
        h_r[idx] = 1.0
        # Smooth it a bit
        from scipy.ndimage import gaussian_filter1d
        h_r = gaussian_filter1d(h_r, 2.0)
        h_r = np.maximum(h_r, 1e-5)
        candidates.append(h_r)

    # Phase 1: Soft-max optimization from diverse candidates
    # Coarse to fine beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for h0 in candidates:
        if time.time() > deadline - 40:
            break
            
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        
        # Coarse to fine soft-max optimization
        for beta in betas:
            if time.time() > deadline - 15:
                break
            try_optimize(params0, 'L-BFGS-B', maxiter=1000, use_soft=True, beta=beta)
            # Update params0 to the best found so far for the next beta stage
            params0 = np.log(best_h)
            
        # Refine with Nelder-Mead on the exact function
        if time.time() < deadline - 15:
            try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=2000, use_soft=False)
            
        # If time permits, try Powell
        if time.time() < deadline - 10:
            try_optimize(np.log(best_h), 'Powell', maxiter=1000, use_soft=False)

    # Phase 2: Coordinate descent polish on the global best
    if time.time() < deadline - 10:
        coordinate_refine(best_h, n_iter=50)
        
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=5000, use_soft=False)
        
    if time.time() < deadline - 2:
        coordinate_refine(best_h, n_iter=20)

    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
        
    return list(best_h)

# EVOLVE-BLOCK-END
