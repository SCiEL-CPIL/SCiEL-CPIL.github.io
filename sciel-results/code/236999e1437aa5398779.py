"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0

def _smooth_max_corr(h, beta=80.0):
    """Smooth approximation of max correlation using log-sum-exp."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    return m + np.log(np.sum(np.exp((corr - m) * beta)) / len(corr)) / beta

def _compute_grad_smooth_max(h, beta=80.0):
    """Compute gradient of smooth max correlation w.r.t. h analytically."""
    corr = np.correlate(h, 1 - h, mode="full")
    m = np.max(corr)
    weights = np.exp((corr - m) * beta)
    weights /= np.sum(weights)
    
    # corr[k] = sum_j h[j] * (1 - h[j + k - (N-1)]) for k in [0, 2N-2]
    # d corr[k] / d h[i] = (1 - h[i + k - (N-1)]) + h[i] * (-1) if indices valid
    # More precisely: corr[k] = sum_{j} h[j] * (1-h[j+k-(N-1)]) where 0<=j<N and 0<=j+k-(N-1)<N
    # Let shift = k - (N-1). Then corr[k] = sum_j h[j]*(1-h[j+shift]) where j in [max(0,-shift), min(N, N-shift)]
    # d corr[k]/d h[i] = (1-h[i+shift]) if i+shift in [0,N) plus -h[i-shift] if i-shift in [0,N)
    # Actually: d/dh[i] [h[j]*(1-h[j+shift])] = (1-h[i+shift]) when j=i, and -h[i] when j+shift=i i.e. j=i-shift
    # So d corr[k]/d h[i] = (1-h[i+shift]) * [0<=i+shift<N] - h[i-shift] * [0<=i-shift<N]
    
    grad = np.zeros(N)
    for k in range(len(corr)):
        if weights[k] < 1e-15:
            continue
        shift = k - (N - 1)
        # Term 1: (1 - h[i+shift]) for i where 0 <= i+shift < N
        i_min1 = max(0, -shift)
        i_max1 = min(N, N - shift)
        if i_min1 < i_max1:
            idx1 = np.arange(i_min1, i_max1)
            grad[idx1] += weights[k] * (1 - h[idx1 + shift])
        # Term 2: -h[i-shift] for i where 0 <= i-shift < N
        i_min2 = max(0, shift)
        i_max2 = min(N, N + shift)
        if i_min2 < i_max2:
            idx2 = np.arange(i_min2, i_max2)
            grad[idx2] -= weights[k] * h[idx2 - shift]
    
    return grad

def _project_to_feasible(h):
    """Project h onto feasible set: sum=N/2, 0<=h<=1."""
    h = np.array(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = N / 2.0 - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _fast_local_search(h, max_iters=500, tol=1e-13, beta=80.0):
    """Fast local search using analytical gradient and adaptive step size."""
    h = np.array(h, dtype=float).copy()
    h = _project_to_feasible(h)
    best_val = _correlation_value(h)
    
    # Adaptive learning rate
    lr = 0.05
    stall_count = 0
    
    for it in range(max_iters):
        grad = _compute_grad_smooth_max(h, beta)
        
        # Subtract mean to respect sum constraint approximately
        grad_centered = grad - np.mean(grad)
        
        h_new = h - lr * grad_centered
        h_new = _project_to_feasible(h_new)
        
        val = _correlation_value(h_new)
        if val < best_val - tol:
            h = h_new.copy()
            best_val = val
            stall_count = 0
            # Increase lr slightly if improving
            lr = min(lr * 1.05, 0.2)
        else:
            stall_count += 1
            lr *= 0.8
            if stall_count > 30:
                break
    
    return h, best_val

def _sinusoidal_start(frequencies, phases, amplitudes, offset=0.5):
    """Generate a starting point from sinusoidal components."""
    t = np.arange(N)
    h = np.zeros(N)
    for f, p, a in zip(frequencies, phases, amplitudes):
        h += a * np.sin(2 * np.pi * f * t / N + p)
    h = offset + h
    h = _project_to_feasible(h)
    return h

def _generate_starts(n_starts=10, seed=42):
    """Generate diverse starting points including structured sinusoidal patterns."""
    rng = np.random.RandomState(seed)
    starts = []
    
    # Start 1: uniform 0.5
    starts.append(np.full(N, 0.5))
    
    # Start 2: alternating
    alt = np.ones(N)
    alt[::2] = 0.0
    alt[1::2] = 1.0
    alt = _project_to_feasible(alt)
    starts.append(alt)
    
    # Structured sinusoidal starts
    # Use incommensurate frequencies to avoid periodicity
    freq_sets = [
        [1, 7, 13],
        [2, 9, 17],
        [3, 11, 19],
        [5, 8, 14],
    ]
    
    for freqs in freq_sets:
        phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
        amplitudes = rng.uniform(0.1, 0.3, size=len(freqs))
        h = _sinusoidal_start(freqs, phases, amplitudes)
        starts.append(h)
    
    # Random starts
    while len(starts) < n_starts:
        h = rng.uniform(0.1, 0.9, size=N)
        h = _project_to_feasible(h)
        starts.append(h)
    
    return starts

def _slsqp_optimize(h_init, deadline):
    """Optimize using SLSQP with smooth objective and analytical gradient."""
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    
    def obj(x):
        return _smooth_max_corr(x, beta=80.0)
    
    def obj_grad(x):
        return _compute_grad_smooth_max(x, beta=80.0)
    
    try:
        result = minimize(
            obj, h_init, method='SLSQP',
            jac=obj_grad,
            bounds=bounds, constraints=constraints,
            options={'maxiter': 150, 'ftol': 1e-12}
        )
        if time.time() > deadline:
            return h_init, _correlation_value(h_init)
        
        h_cand = _project_to_feasible(result.x)
        val = _correlation_value(h_cand)
        return h_cand, val
    except Exception:
        return h_init, _correlation_value(h_init)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    starts = _generate_starts(n_starts=10, seed=42)
    
    best_h = starts[0].copy()
    best_val = _correlation_value(best_h)
    
    # Phase 1: SLSQP optimization from multiple starts
    for i, h_init in enumerate(starts):
        if time.time() > deadline - 40.0:
            break
        
        h_opt, val_opt = _slsqp_optimize(h_init, deadline)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 2: Fast local search from best and a few others
    candidates = [best_h]
    for h_init in starts[:3]:
        if time.time() > deadline - 20.0:
            break
        h_opt, val_opt = _fast_local_search(h_init, max_iters=300)
        candidates.append(h_opt)
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: High-precision polish on the best solution
    if time.time() < deadline - 5.0:
        h_final, val_final = _fast_local_search(best_h, max_iters=1000, tol=1e-14)
        if val_final < best_val:
            best_h = h_final
            best_val = val_final
    
    # Final validation
    best_h = _project_to_feasible(best_h)
    return best_h
# EVOLVE-BLOCK-END
