"""Second autocorrelation inequality - combined soft-max surrogate + NM polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_params(params):
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _logsumexp(x, beta):
    """Compute log-sum-exp of x with temperature 1/beta (soft-max)."""
    x = np.asarray(x, dtype=float)
    xm = np.max(x)
    if xm < -1e300:
        return -np.inf
    return xm + np.log(np.sum(np.exp(-beta * (x - xm)))) / beta

def _soft_value(params, beta):
    """Smooth surrogate: replace max|g| with soft-max of |g|."""
    h = _h_from_params(params)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g)
    soft_max = _logsumexp(abs_g, beta)
    if norm_1 < 1e-300:
        return 0.0
    return l2 / (norm_1 * np.exp(soft_max))

def _neg_soft_value(params, beta):
    return -_soft_value(params, beta)

def _neg_value(params):
    h = _h_from_params(params)
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    def try_optimize(params0, method='Nelder-Mead', maxiter=3000, use_soft=False, beta=10.0, **kwargs):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            if use_soft:
                objective = lambda p: _neg_soft_value(p, beta)
            else:
                objective = _neg_value
                
            if method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            elif method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-7}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-9, 'ftol': 1e-12}
            else:
                opts = {}
            res = minimize(objective, params0, method=method, options=opts)
            
            # Always evaluate the true objective to update best
            h_new = _h_from_params(res.x)
            v_new = _compute_value(h_new)
            
            if v_new > best_val:
                best_val = v_new
                best_h = h_new
        except Exception:
            pass

    def coordinate_refine(h, n_iter=50):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=float).copy()
        h = np.maximum(h, 1e-15)
        h = h / h.sum()
        v = _compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h
            
        for _ in range(n_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    h_new = h.copy()
                    h_new[i] = max(h_new[i] + delta, 1e-15)
                    h_new = h_new / h_new.sum()
                    v_new = _compute_value(h_new)
                    if v_new > best_val + 1e-10:
                        h = h_new
                        best_val = v_new
                        improved = True
                        break
            if not improved:
                break
        return best_h, best_val

    rng = np.random.default_rng(50)
    x = np.linspace(-1, 1, N)
    xh = np.linspace(-0.5, 0.5, N)

    # Generate diverse initial candidates
    candidates = []
    # Uniform
    candidates.append(np.ones(N))
    # Various Gaussians
    for a in [0.3, 0.5, 1, 2, 4, 8, 16, 32]:
        candidates.append(np.exp(-a * x**2))
    # Polynomials
    for k in [2, 3, 4, 5, 6, 8]:
        candidates.append(np.maximum(0, 1 - np.abs(x)**k))
    # Cosine
    candidates.append(np.maximum(0, np.cos(np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(2 * np.pi * xh)))
    candidates.append(np.maximum(0, np.cos(np.pi * xh)**2))
    # Sine
    candidates.append(np.maximum(0, np.sin(np.pi * xh)))
    candidates.append(np.maximum(0, np.sin(np.pi * xh)**2))
    # Bimodal
    for c in [0.2, 0.3, 0.4, 0.5]:
        for a in [4, 8, 12, 20]:
            candidates.append(np.exp(-a*(x-c)**2) + np.exp(-a*(x+c)**2))
    # Plateau-like
    for start, end in [(5,45),(10,40),(15,35),(0,50)]:
        plat = np.zeros(N)
        plat[start:end] = 1.0
        candidates.append(plat)
    # Triangle
    candidates.append(np.maximum(0, 1 - 2*np.abs(xh)))
    # Random
    for _ in range(20):
        h_r = np.abs(rng.standard_normal(N)) + 0.01
        candidates.append(h_r)
    # Smooth random via low-frequency Fourier
    for _ in range(10):
        coeffs = rng.standard_normal(6)
        h_f = np.ones(N)
        for k in range(1, 6):
            h_f += coeffs[k-1] * np.cos(2*np.pi*k*xh)
        h_f = np.abs(h_f) + 0.01
        candidates.append(h_f)

    # Phase 1: Soft-max optimization from diverse candidates
    # Use decreasing beta (coarse to fine) to navigate the landscape
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    for h0 in candidates:
        if time.time() > deadline - 25:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        
        # Coarse to fine soft-max optimization
        for beta in betas:
            if time.time() > deadline - 10:
                break
            try_optimize(params0, 'L-BFGS-B', maxiter=500, use_soft=True, beta=beta)
            # Update params0 to the best found so far for the next beta stage
            params0 = np.log(best_h)
            
        # Refine with Nelder-Mead on the exact function
        if time.time() < deadline - 10:
            try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=1000)

    # Phase 2: Coordinate descent polish on the global best
    if time.time() < deadline - 10:
        coordinate_refine(best_h, n_iter=100)

    # Phase 3: Final high-precision polish with NM and Powell
    if time.time() < deadline - 5:
        try_optimize(np.log(best_h), 'Powell', maxiter=500)
    if time.time() < deadline - 3:
        try_optimize(np.log(best_h), 'Nelder-Mead', maxiter=2000)

    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
# EVOLVE-BLOCK-END
