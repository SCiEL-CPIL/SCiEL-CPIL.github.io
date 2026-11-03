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

def _fixed_point_iteration(h, target_sum, beta=50.0, max_iters=500, tol=1e-10):
    """Fixed-point iteration derived from optimality conditions.
    
    The idea: at optimum, the gradient of the max-overlap objective is orthogonal
    to the feasible set. We iterate:
      1. Compute overlap vector R_k for current h
      2. Compute soft-min weights w_k = softmax(-beta * R_k)
      3. Compute gradient direction g = -sum_k w_k * dR_k/dh
      4. Update h <- h + alpha * g, then project to feasible set
      5. Repeat until convergence
    
    This is structurally different from L-BFGS because it uses a fixed step
    size schedule and the natural gradient structure of the problem.
    """
    n = len(h)
    h = _project(h.copy(), target_sum)
    current_corr = _compute_min_corr(h)
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        corrs = _compute_all_corrs_fft(h)
        
        # Soft-min weights
        max_corr = np.max(corrs)
        exp_vals = np.exp(-beta * (corrs - max_corr))
        weights = exp_vals / np.sum(exp_vals)
        
        # Compute gradient: g[i] = -sum_k w_k * (2/n) * (h[i+k] + h[i-k])
        g = np.zeros(n)
        for k in range(1, n):
            wk = weights[k-1] * (2.0 / n)
            if wk > 1e-15:
                g[:n-k] -= wk * h[k:]
                g[k:] -= wk * h[:n-k]
        
        # Adaptive step size
        alpha = 0.1 / (1.0 + it * 0.05)
        
        # Update and project
        h_trial = h + alpha * g
        h_trial = _project(h_trial, target_sum)
        
        trial_corr = _compute_min_corr(h_trial)
        
        if trial_corr > current_corr:
            h = h_trial
            current_corr = trial_corr
        else:
            # Try smaller step
            alpha *= 0.5
            h_trial = h + alpha * g
            h_trial = _project(h_trial, target_sum)
            trial_corr = _compute_min_corr(h_trial)
            if trial_corr > current_corr:
                h = h_trial
                current_corr = trial_corr
            else:
                # Try opposite direction
                h_trial = h - alpha * g
                h_trial = _project(h_trial, target_sum)
                trial_corr = _compute_min_corr(h_trial)
                if trial_corr > current_corr:
                    h = h_trial
                    current_corr = trial_corr
                else:
                    # No improvement, break
                    break
    
    return h, current_corr

def _spectral_fixed_point(h, target_sum, max_iters=300):
    """Spectral fixed-point method using eigenvalue structure.
    
    Represents the overlap problem in Fourier space where the correlation
    operator becomes diagonal. Iterate in spectral domain.
    """
    n = len(h)
    h = _project(h.copy(), target_sum)
    current_corr = _compute_min_corr(h)
    
    # Compute FFT of h
    H = np.fft.rfft(h)
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Compute correlations in spatial domain
        corrs = _compute_all_corrs_fft(h)
        
        # Find the worst shift
        min_idx = np.argmin(corrs)
        k_star = min_idx + 1
        
        # Compute the "pressure" at each position from the worst shift
        # At position i, the contribution to R_{k_star} comes from pairs (i, i+k_star)
        pressure = np.zeros(n)
        for j in range(n - k_star):
            pressure[j] += h[j + k_star]
            pressure[j + k_star] += h[j]
        
        # Update: move h away from high-pressure regions
        # This is like a diffusion step
        alpha = 0.05 / (1.0 + it * 0.02)
        h_trial = h - alpha * (pressure - np.mean(pressure))
        h_trial = _project(h_trial, target_sum)
        
        trial_corr = _compute_min_corr(h_trial)
        if trial_corr > current_corr:
            h = h_trial
            current_corr = trial_corr
        else:
            # Try smaller step
            alpha *= 0.5
            h_trial = h - alpha * (pressure - np.mean(pressure))
            h_trial = _project(h_trial, target_sum)
            trial_corr = _compute_min_corr(h_trial)
            if trial_corr > current_corr:
                h = h_trial
                current_corr = trial_corr
            else:
                break
    
    return h, current_corr

def _simulated_annealing(h, target_sum, max_iters=2000, T0=0.1, T_end=0.001):
    """Simulated annealing in the space of step heights.
    
    A different search paradigm: accept worse solutions with probability
    exp(-delta/T) to escape local minima.
    """
    n = len(h)
    h = _project(h.copy(), target_sum)
    current_corr = _compute_min_corr(h)
    best_h = h.copy()
    best_corr = current_corr
    
    for it in range(max_iters):
        if _time_remaining() < 5.0:
            break
        
        # Temperature schedule
        T = T0 * (T_end / T0) ** (it / max_iters)
        
        # Random perturbation
        step = 0.05 * T / T0
        perturbation = np.random.normal(0, step, n)
        h_trial = h + perturbation
        h_trial = _project(h_trial, target_sum)
        
        trial_corr = _compute_min_corr(h_trial)
        delta = trial_corr - current_corr
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            h = h_trial
            current_corr = trial_corr
            if current_corr > best_corr:
                best_corr = current_corr
                best_h = h.copy()
    
    return best_h, best_corr

def _construct_diverse_initializations(n):
    """Generate diverse initializations for the fixed-point methods."""
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = int(n * frac)
        if 0 < k < n:
            restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*(n/2.0)/(n-k))]))
            restarts.append(np.concatenate([np.full(k, 2.0*(n/2.0)/k), np.full(n-k, 0.0)]))
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
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            restarts.append(0.5 + 0.35*np.sin(freq*x + phase))
    # Mixed sinusoidal
    np.random.seed(123)
    for _ in range(10):
        f1 = np.random.randint(1, 9)
        f2 = np.random.randint(1, 9)
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        a1 = np.random.uniform(0.2, 0.45)
        a2 = np.random.uniform(0.2, 0.45)
        restarts.append(0.5 + a1*np.sin(f1*x + p1) + a2*np.sin(f2*x + p2))
    
    return restarts

def construct():
    """Construct step heights minimizing maximum overlap using fixed-point iteration."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_corr = _compute_min_corr(best_x)
    
    restarts = _construct_diverse_initializations(n)
    
    # Phase 1: Fixed-point iteration with various betas
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project(x0, target_sum)
        
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if _time_remaining() < 4.0:
                break
            try:
                h_trial, corr_trial = _fixed_point_iteration(x, target_sum, beta=beta, max_iters=300)
                if corr_trial > best_corr:
                    best_corr = corr_trial
                    best_x = h_trial
                x = h_trial
            except Exception:
                pass
    
    # Phase 2: Spectral fixed-point on best solution
    if _time_remaining() > 10.0:
        for _ in range(3):
            if _time_remaining() < 4.0:
                break
            try:
                h_trial, corr_trial = _spectral_fixed_point(best_x, target_sum, max_iters=200)
                if corr_trial > best_corr:
                    best_corr = corr_trial
                    best_x = h_trial
            except Exception:
                pass
    
    # Phase 3: Simulated annealing to escape local minima
    if _time_remaining() > 15.0:
        try:
            h_trial, corr_trial = _simulated_annealing(best_x, target_sum, max_iters=1500, T0=0.08, T_end=0.0005)
            if corr_trial > best_corr:
                best_corr = corr_trial
                best_x = h_trial
        except Exception:
            pass
    
    # Phase 4: Final fixed-point polish with high beta
    if _time_remaining() > 8.0:
        for beta in [300.0, 500.0]:
            if _time_remaining() < 4.0:
                break
            try:
                h_trial, corr_trial = _fixed_point_iteration(best_x, target_sum, beta=beta, max_iters=400)
                if corr_trial > best_corr:
                    best_corr = corr_trial
                    best_x = h_trial
            except Exception:
                pass
    
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
