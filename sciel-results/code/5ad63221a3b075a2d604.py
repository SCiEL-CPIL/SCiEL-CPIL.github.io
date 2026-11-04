"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

_start_time = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _score_h(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Uses FFT-based cross-correlation for efficiency.
    """
    sw = 2.0 / N
    autocorr = np.correlate(h, h, mode='full')
    R = sw * autocorr[N-1 : 2*N-1]
    C = 1.0 - R
    return np.max(C)

def _project(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    Uses bisection on a Lagrange multiplier for exact projection.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    lo = -2.0
    hi = 2.0
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0.0, 1.0))
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s_mid = sum_at_lambda(mid)
        if abs(s_mid - target) < 1e-12:
            break
        if s_mid > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    h = np.clip(h - lam, 0.0, 1.0)
    
    # Final adjustment for any residual error due to floating point
    residual = target - np.sum(h)
    if abs(residual) > 1e-14:
        interior = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.sum(interior) > 0:
            h[interior] += residual / np.sum(interior)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _lse_objective(h, beta):
    """
    LogSumExp of correlations for smooth approximation of max.
    """
    sw = 2.0 / N
    autocorr = np.correlate(h, h, mode='full')
    R = sw * autocorr[N-1 : 2*N-1]
    C = 1.0 - R
    max_c = np.max(C)
    if max_c > 500:
        max_c = 500.0
    lse = max_c + np.log(np.sum(np.exp(beta * (C - max_c)))) / beta
    return lse

def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to c_i <= t for all i.
    Uses smooth penalty for constraint violations.
    """
    h = x[:N]
    t = x[N]
    sw = 2.0 / N
    autocorr = np.correlate(h, h, mode='full')
    R = sw * autocorr[N-1 : 2*N-1]
    C = 1.0 - R
    violations = np.maximum(0, C - t)
    penalty = np.sum(violations ** 2) * beta
    return t + penalty

def _optimize_lse(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
    """
    Optimize using L-BFGS-B with LSE objective.
    """
    bounds = [(0, 1) for _ in range(N)]
    
    def obj(h):
        h_proj = _project(h)
        return _lse_objective(h_proj, beta)

    try:
        res = minimize(
            obj, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        h_opt = res.x
    except Exception:
        h_opt = h0
    
    h_opt = _project(h_opt)
    return h_opt

def _optimize_epigraph(h0, t0, beta, max_iter, ftol=1e-10, gtol=1e-10):
    """
    Optimize using epigraph formulation with L-BFGS-B.
    """
    bounds = [(0, 1) for _ in range(N)] + [(0, N)]
    
    def obj(x):
        h = x[:N]
        t = x[N]
        h_proj = _project(h)
        return _epigraph_objective(np.concatenate([h_proj, [t]]), beta)

    try:
        res = minimize(
            obj, 
            np.concatenate([h0, [t0]]), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        h_opt = res.x[:N]
    except Exception:
        h_opt = h0
    
    h_opt = _project(h_opt)
    return h_opt

def construct():
    global _start_time
    deadline = _start_time + _TIME_LIMIT

    best_h = None
    best_val = np.inf

    # 1. Uniform
    h0 = np.full(N, 0.5)
    val0 = _score_h(h0)
    best_h = h0.copy()
    best_val = val0

    # 2. Alternating binary
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt = _project(h_alt)
    val_alt = _score_h(h_alt)
    if val_alt < best_val:
        best_h = h_alt.copy()
        best_val = val_alt

    # 3. Low-frequency cosine modulation
    n_vals = np.arange(N)
    h_cos = 0.5 + 0.2 * np.cos(2 * np.pi * n_vals / N)
    h_cos = _project(h_cos)
    val_cos = _score_h(h_cos)
    if val_cos < best_val:
        best_h = h_cos.copy()
        best_val = val_cos

    # 4. Sturmian word candidates (quasi-periodic sequences)
    for alpha in [0.6180339887, 0.7071067812, 0.75]:
        h_sturm = np.floor((n_vals + 1) * alpha) - np.floor(n_vals * alpha)
        h_sturm = h_sturm.astype(float)
        h_sturm = _project(h_sturm)
        val_sturm = _score_h(h_sturm)
        if val_sturm < best_val:
            best_h = h_sturm.copy()
            best_val = val_sturm

    # 5. Multi-restart LSE optimization
    rng = np.random.default_rng(42)
    restarts = []
    
    # Random perturbations
    for i in range(3):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = _project(h_rand)
        beta = 5.0 + i * 5.0
        restarts.append((h_rand, beta, 800))
        
    # Bimodal initializations
    for i in range(3):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = _project(h_rand)
        beta = 20.0 + i * 5.0
        restarts.append((h_rand, beta, 600))

    for idx, (h0_r, beta, max_iter) in enumerate(restarts):
        if time.time() > deadline - 40:
            break
        try:
            h_opt = _optimize_lse(h0_r, beta, max_iter)
            current_score = _score_h(h_opt)
            if current_score < best_val:
                best_val = current_score
                best_h = h_opt.copy()
        except Exception:
            pass

    # 6. Epigraph optimization on best solution
    if time.time() < deadline - 30:
        try:
            t0 = best_val
            h_epigraph = _optimize_epigraph(best_h, t0, 50.0, 1500, ftol=1e-12, gtol=1e-12)
            epigraph_score = _score_h(h_epigraph)
            if epigraph_score < best_val:
                best_val = epigraph_score
                best_h = h_epigraph.copy()
        except Exception:
            pass

    # 7. High-precision polish pass
    if time.time() < deadline - 10:
        try:
            polish_beta = 100.0
            h_polished = _optimize_lse(best_h, polish_beta, 2000, ftol=1e-12, gtol=1e-12)
            polish_score = _score_h(h_polished)
            if polish_score < best_val:
                best_val = polish_score
                best_h = h_polished.copy()
        except Exception:
            pass

    best_h = _project(best_h)
    return best_h
# EVOLVE-BLOCK-END
