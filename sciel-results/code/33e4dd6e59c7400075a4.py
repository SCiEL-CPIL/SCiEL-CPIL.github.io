import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0
M = 2 * N - 1  # length of g

def _value(h):
    """
    Computes the exact objective value for a given height array h.
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the objective using log-sum-exp for norm_inf.
    beta controls smoothness: larger beta -> closer to exact max.
    """
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    abs_g = np.abs(g)
    norm_1 = np.sum(abs_g) / (m + 1)
    
    # Smooth max via log-sum-exp
    max_g = np.max(abs_g)
    if max_g < 1e-12:
        return 0.0
    # log-sum-exp trick for numerical stability
    shifted = abs_g / max_g
    log_sum_exp = np.log(np.sum(np.exp(beta * (shifted - 1.0)))) / beta
    norm_inf_smooth = max_g * np.exp(log_sum_exp)
    
    if norm_1 < 1e-12 or norm_inf_smooth < 1e-12:
        return 0.0
        
    return l2 / (norm_1 * norm_inf_smooth)

def _neg_smooth_value(h, beta=50.0):
    return -_smooth_value(h, beta)

def _neg_value(h):
    return -_value(h)

def _lbfgs_smooth(h, beta=50.0, maxiter=3000, deadline=None):
    """
    L-BFGS-B optimization using the smooth surrogate.
    """
    bounds_full = [(0, None)] * N
    try:
        res = minimize(
            lambda x: _neg_smooth_value(x, beta),
            h, 
            method='L-BFGS-B',
            bounds=bounds_full,
            options={
                'maxiter': maxiter,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0, None)
            return h_polished, _value(h_polished)
    except Exception:
        pass
    return h, _value(h)

def _lbfgs_exact(h, maxiter=1000, deadline=None):
    """
    L-BFGS-B polish using exact objective (may struggle with kinks).
    """
    bounds_full = [(0, None)] * N
    try:
        res = minimize(
            _neg_value,
            h, 
            method='L-BFGS-B',
            bounds=bounds_full,
            options={
                'maxiter': maxiter,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0, None)
            return h_polished, _value(h_polished)
    except Exception:
        pass
    return h, _value(h)

def _coordinate_descent(h, max_iter=200, tol=1e-12, deadline=None):
    """
    Coordinate descent with local line search.
    """
    h = np.array(h, dtype=float)
    h = np.maximum(h, 0)
    best_v = _value(h)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            best_local = old_val
            best_local_v = best_v
            
            # Try multiplicative and additive perturbations
            candidates = []
            if old_val > 1e-6:
                for scale in [0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
                    candidates.append(old_val * scale)
            else:
                for trial in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
                    candidates.append(trial)
            
            for trial in candidates:
                h[i] = trial
                v = _value(h)
                if v > best_local_v:
                    best_local_v = v
                    best_local = trial
            
            h[i] = best_local
            if best_local_v > best_v + tol:
                best_v = best_local_v
                improved = True
        if not improved:
            break
    return h, best_v

def _nelder_mead_polish(h, deadline=None):
    """
    Nelder-Mead for robustness near non-smooth points.
    """
    try:
        res = minimize(
            _neg_value,
            h,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-14,
                'fatol': 1e-14,
                'adaptive': True
            }
        )
        if res.x is not None:
            h_polished = np.clip(res.x, 0, None)
            return h_polished, _value(h_polished)
    except Exception:
        pass
    return h, _value(h)

def _make_initial_guesses(rng):
    """Generate diverse initial guesses."""
    x = np.arange(N) / N
    guesses = []
    
    # Uniform
    guesses.append(np.ones(N))
    
    # Single Gaussians
    for center in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for sigma in [0.05, 0.1, 0.15, 0.2, 0.3]:
            guesses.append(np.exp(-((x - center)**2) / (2 * sigma**2)))
    
    # Two Gaussians
    for c1 in [0.2, 0.3, 0.4]:
        for c2 in [0.6, 0.7, 0.8]:
            for s in [0.1, 0.15, 0.2]:
                guesses.append(np.exp(-((x - c1)**2) / (2*s**2)) + np.exp(-((x - c2)**2) / (2*s**2)))
    
    # Power-law ramps
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
        h = (x ** beta)
        guesses.append(h)
        guesses.append(((1-x) ** beta))
    
    # Cosine bumps
    for c in [0.3, 0.5, 0.7]:
        for width in [0.15, 0.25, 0.4]:
            d = np.minimum(np.abs(x - c), 1.0 - np.abs(x - c))
            d = np.clip(d / width, 0, 1)
            guesses.append(np.cos(np.pi * d / 2)**4)
    
    # Random smooth functions
    for _ in range(20):
        k = rng.integers(2, 6)
        centers = rng.uniform(0, 1, k)
        sigmas = rng.uniform(0.05, 0.3, k)
        amps = rng.uniform(0.5, 2.0, k)
        h = np.zeros(N)
        for c, s, a in zip(centers, sigmas, amps):
            h += a * np.exp(-((x - c)**2) / (2 * s**2))
        guesses.append(h)
    
    return guesses

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(42)

    # Generate and evaluate initial guesses
    guesses = _make_initial_guesses(rng)
    best_h = None
    best_v = 0.0
    for h in guesses:
        h = np.maximum(h, 0)
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_v = _value(best_h)
    
    # Phase 1: Smooth L-BFGS-B from multiple starting points
    # Use decreasing beta for annealing effect
    betas = [20.0, 50.0, 100.0]
    for beta in betas:
        if time.time() > deadline - 30:
            break
        n_starts = 15
        for i in range(n_starts):
            if time.time() > deadline - 25:
                break
            # Pick a random initial guess or perturb best
            if i == 0:
                h0 = best_h.copy()
            else:
                h0 = guesses[i % len(guesses)].copy()
                # Add small noise
                h0 += rng.normal(0, 0.1, N)
                h0 = np.maximum(h0, 0)
            
            h_new, v_new = _lbfgs_smooth(h0, beta=beta, maxiter=2000, deadline=deadline - 20)
            if v_new > best_v:
                best_v = v_new
                best_h = h_new
    
    # Phase 2: More smooth optimization from best with tighter beta
    if time.time() < deadline - 20:
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 15:
                break
            h_new, v_new = _lbfgs_smooth(best_h, beta=beta, maxiter=3000, deadline=deadline - 12)
            if v_new > best_v:
                best_v = v_new
                best_h = h_new
    
    # Phase 3: Coordinate descent
    if time.time() < deadline - 10:
        best_h, best_v = _coordinate_descent(best_h, max_iter=300, deadline=deadline - 8)
    
    # Phase 4: Exact L-BFGS-B polish
    if time.time() < deadline - 5:
        best_h, best_v = _lbfgs_exact(best_h, maxiter=1500, deadline=deadline - 3)
    
    # Phase 5: Nelder-Mead for robustness
    if time.time() < deadline - 3:
        h_nm, v_nm = _nelder_mead_polish(best_h, deadline=deadline - 1)
        if v_nm > best_v:
            best_v = v_nm
            best_h = h_nm
    
    # Phase 6: Final coordinate descent
    if time.time() < deadline - 1:
        best_h, best_v = _coordinate_descent(best_h, max_iter=200, deadline=deadline - 0.5)
    
    # Final safety
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
