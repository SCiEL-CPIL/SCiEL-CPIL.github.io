"""Rewritten program for Erdos minimum overlap task."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, brentq

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all linear cross-correlations R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1,...,n-1
    using FFT for O(n log n) speed."""
    n = len(h)
    h_padded = np.zeros(2 * n)
    h_padded[:n] = h
    H = np.fft.rfft(h_padded, n=2*n)
    acf = np.fft.irfft(H * np.conj(H), n=2*n)
    # acf[k] = sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=0,...,n-1
    corrs = acf[1:n] * (2.0 / n)
    return corrs

def _compute_min_corr(h):
    """Compute min over shifts of the cross-correlation."""
    corrs = _compute_all_corrs_fft(h)
    return np.min(corrs)

def _project(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    def f(theta):
        w = np.clip(v - theta, 0, 1)
        return np.sum(w) - target_sum
    
    lo = np.min(v) - 1.0
    hi = np.max(v) + 1.0
    
    if f(lo) < 0:
        lo -= 10.0
    if f(hi) > 0:
        hi += 10.0
        
    try:
        theta = brentq(f, lo, hi, xtol=1e-12)
    except ValueError:
        w = np.clip(v, 0, 1)
        for _ in range(50):
            diff = target_sum - np.sum(w)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = w < 1.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            else:
                mask = w > 0.0
                cnt = np.sum(mask)
                if cnt == 0: break
                w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
        return w

    w = np.clip(v - theta, 0, 1)
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-12:
        mask = (w > 1e-9) & (w < 1.0 - 1e-9)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0, 1)
    return w

def _soft_min_obj_grad(x, beta=100.0):
    """Objective and gradient using soft-min approximation with FFT-based correlations."""
    h = x
    n = len(h)
    
    corrs = _compute_all_corrs_fft(h)
    
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = -np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    weights = exp_neg_beta / sum_exp_neg
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w != 0:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    grad_obj = -grad
    
    return obj, grad_obj

def _subgradient_step(h, target_sum, step_size=0.01):
    """Take a subgradient step on the min-correlation constraint."""
    n = len(h)
    corrs = _compute_all_corrs_fft(h)
    min_idx = np.argmin(corrs)
    k_star = min_idx + 1
    
    grad = np.zeros(n)
    for j in range(n - k_star):
        grad[j] += (2.0 / n) * h[j + k_star]
        grad[j + k_star] += (2.0 / n) * h[j]
    
    # We want to increase min corr, so move in direction of grad
    h_trial = h + step_size * grad
    h_trial = _project(h_trial, target_sum)
    return h_trial

def _coordinate_polish(h, target_sum, max_iters=100):
    """Polish using coordinate descent on the min-correlation."""
    n = len(h)
    for _ in range(max_iters):
        if _time_remaining() < 5.0:
            break
        improved = False
        for i in range(n):
            if _time_remaining() < 5.0:
                break
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005]:
                h_trial = h.copy()
                h_trial[i] += delta
                h_trial = _project(h_trial, target_sum)
                trial_corr = _compute_min_corr(h_trial)
                current_corr = _compute_min_corr(h)
                if trial_corr > current_corr + 1e-10:
                    h = h_trial
                    improved = True
                    break
        if not improved:
            break
    return h

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_corr = _compute_min_corr(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5, 7]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            restarts.append(0.5 + 0.3*np.sin(freq*x + phase))
            restarts.append(0.5 + 0.3*np.cos(freq*x + phase))
    # Mixed sinusoidal
    np.random.seed(123)
    for _ in range(10):
        f1 = np.random.randint(1, 8)
        f2 = np.random.randint(1, 8)
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        a1 = np.random.uniform(0.2, 0.4)
        a2 = np.random.uniform(0.2, 0.4)
        restarts.append(0.5 + a1*np.sin(f1*x + p1) + a2*np.sin(f2*x + p2))
    
    # Phase 1: Soft-min optimization with increasing beta per restart
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    for x0 in restarts:
        if _time_remaining() < 10.0:
            break
        
        x = _project(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project(res.x, target_sum)
                res_corr = _compute_min_corr(res_x)
                if res_corr > best_corr:
                    best_corr = res_corr
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: High-beta polish on best solution
    if _time_remaining() > 15.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_grad,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project(res.x, target_sum)
                res_corr = _compute_min_corr(res_x)
                if res_corr > best_corr:
                    best_corr = res_corr
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 3: Subgradient and coordinate polish
    if _time_remaining() > 10.0:
        h = best_x.copy()
        for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if _time_remaining() < 5.0:
                break
            for _ in range(50):
                if _time_remaining() < 5.0:
                    break
                h_trial = _subgradient_step(h, target_sum, step_size)
                trial_corr = _compute_min_corr(h_trial)
                if trial_corr > _compute_min_corr(h) + 1e-10:
                    h = h_trial
                else:
                    break
        best_x = h
        best_corr = _compute_min_corr(best_x)
    
    # Phase 4: Coordinate polish
    if _time_remaining() > 5.0:
        best_x = _coordinate_polish(best_x, target_sum, max_iters=50)
        best_corr = _compute_min_corr(best_x)
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
