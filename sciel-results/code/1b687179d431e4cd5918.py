"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses L-BFGS-B with penalty for sum constraint and log-sum-exp smoothing.
Includes random restarts and coordinate descent polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Exact Euclidean projection onto {h in [0,1]^N : sum(h) = N/2}."""
    h = np.array(h, dtype=np.float64, copy=True)
    target = _TARGET

    # Quick check if already feasible
    if abs(np.sum(h) - target) < 1e-13 and np.min(h) >= -1e-13 and np.max(h) <= 1 + 1e-13:
        return np.clip(h, 0.0, 1.0)

    lo = np.min(h) - 2.0
    hi = np.max(h) + 2.0

    def calc_sum(tau):
        return np.sum(np.clip(h - tau, 0.0, 1.0))

    # Ensure bracket
    if calc_sum(lo) < target:
        lo -= 10.0
    if calc_sum(hi) > target:
        hi += 10.0

    # Bisection for tau
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = calc_sum(mid)
        if abs(s - target) < 1e-14:
            lo = hi = mid
            break
        if s > target:
            lo = mid
        else:
            hi = mid

    tau = (lo + hi) / 2.0
    h_proj = np.clip(h - tau, 0.0, 1.0)

    # Final exact correction
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-15:
        mask = (h_proj > 1e-10) & (h_proj < 1.0 - 1e-10)
        if np.sum(mask) > 0:
            h_proj[mask] += diff / np.sum(mask)
            h_proj = np.clip(h_proj, 0.0, 1.0)
        else:
            if diff > 0:
                idx = np.argmin(h_proj)
            else:
                idx = np.argmax(h_proj)
            h_proj[idx] += diff
            h_proj = np.clip(h_proj, 0.0, 1.0)

    return h_proj


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s
    return val, grad


def _objective_and_grad(h, beta, lambda_pen):
    """
    Smooth objective + penalty for sum constraint.
    f(h) = S_beta(C(h)) + lambda * (sum(h) - N/2)^2
    """
    cs = _overlap_values(h)
    val_smooth, g_c = _smooth_max_and_grad(cs, beta)
    
    # Gradient of smooth max w.r.t h
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc < 1e-15:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
            
    # Penalty term
    sum_h = np.sum(h)
    pen_val = lambda_pen * (sum_h - _TARGET) ** 2
    pen_grad = 2.0 * lambda_pen * (sum_h - _TARGET) * np.ones(n)
    
    total_val = val_smooth + pen_val
    total_grad = g_h + pen_grad
    
    return total_val, total_grad


def _coord_descent_polish(h, n_rounds=50, time_budget=8.0):
    """Coordinate descent targeting the maximum C_k constraints."""
    t_start = time.time()
    h = _project(h.copy())
    best_score = _score(h)
    best_h = h.copy()
    
    # Identify active set
    cs = _overlap_values(h)
    max_c = np.max(cs)
    
    for round_idx in range(n_rounds):
        if time.time() - t_start > time_budget:
            break
        
        for i in range(N):
            if time.time() - t_start > time_budget:
                break
            
            # Try small adjustments
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project(h_trial)
                
                s_trial = _score(h_trial)
                if s_trial < best_score - 1e-12:
                    best_score = s_trial
                    best_h = h_trial.copy()
                    h = h_trial.copy()
                    
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Initial Candidates
    i = np.arange(N)
    candidates = []
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    candidates.append(h_cos)
    
    # Constant
    h_const = np.full(N, 0.5)
    candidates.append(h_const)
    
    # Sine
    h_sin = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N)
    candidates.append(h_sin)
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.8, 0.2)
    candidates.append(h_alt)
    
    # Random
    h_rand = np.random.uniform(0.1, 0.9, N)
    candidates.append(h_rand)
    
    # More diverse
    h_cos2 = 0.5 + 0.3 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h_cos2)
    
    h_saw = 0.5 + 0.4 * (2.0 * (i / N - 0.5))
    candidates.append(h_saw)
    
    h_tri = 0.5 + 0.3 * np.abs(2.0 * (i / N % 1.0) - 1.0) - 0.3
    candidates.append(h_tri)
    
    h_mix = 0.5 + 0.2 * np.cos(2.0 * np.pi * i / N) + 0.15 * np.cos(4.0 * np.pi * i / N)
    candidates.append(h_mix)
    
    h_sin_phase = 0.5 + 0.4 * np.sin(2.0 * np.pi * i / N + np.pi / 4)
    candidates.append(h_sin_phase)
    
    # Update best with initial candidates
    for h0 in candidates:
        _update_best(h0)
        
    # Optimization Phases
    # We use L-BFGS-B with penalty.
    # Beta schedule: start low, increase.
    betas = [10.0, 50.0, 100.0, 200.0]
    
    # Penalty coefficient: large enough to enforce sum constraint
    lambda_pen = 100.0
    
    for beta in betas:
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        # For each beta, do a few restarts
        # Use the current best as one start point, and some random ones
        start_points = [best_h.copy()] if best_h is not None else [np.full(N, 0.5)]
        
        # Add random restarts
        for _ in range(3):
            h_rand = np.random.uniform(0.1, 0.9, N)
            start_points.append(h_rand)
            
        for h0 in start_points:
            if time.time() - t0 > _DEADLINE - 5:
                break
                
            h0_proj = _project(h0)
            
            # Define objective and gradient for scipy
            def obj_grad(x):
                val, grad = _objective_and_grad(x, beta, lambda_pen)
                return val, grad
                
            try:
                res = minimize(
                    obj_grad, 
                    h0_proj, 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=[(0, 1)] * N, 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = res.x
                _update_best(h_opt)
            except Exception:
                pass
                
        # After L-BFGS-B, do a quick coordinate polish if time permits
        if best_h is not None and time.time() - t0 < _DEADLINE - 5:
            h_cd, s_cd = _coord_descent_polish(best_h, n_rounds=30, time_budget=5.0)
            _update_best(h_cd)
            
    # Final high-beta refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        beta_final = 500.0
        h0 = best_h.copy()
        
        def obj_grad_final(x):
            val, grad = _objective_and_grad(x, beta_final, lambda_pen)
            return val, grad
            
        try:
            res = minimize(
                obj_grad_final, 
                h0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(0, 1)] * N, 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = res.x
            _update_best(h_opt)
        except Exception:
            pass
            
    # Final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_cd, s_cd = _coord_descent_polish(best_h, n_rounds=50, time_budget=5.0)
        _update_best(h_cd)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h)
    return best_h
