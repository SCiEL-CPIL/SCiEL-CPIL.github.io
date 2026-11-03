import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    # Binary search for tau such that sum(clamp(y + tau, 0, 1)) = target_sum
    # g(tau) is monotonic increasing.
    # Range of tau: [min(y) - 1, max(y)]
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
    
    # Check if solution is within bounds without clipping (all active)
    if g(lo) > 0:
        # Sum is too high even at lowest tau? Should not happen if target is in [0, N]
        # But let's just clip
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(50): # 50 iterations of binary search is plenty for float64 precision
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _pgd_optimize(h0, beta, steps=500, lr=0.01):
    """
    Projected Gradient Descent optimization.
    """
    h = h0.copy()
    target_sum = N / 2.0
    
    # Adaptive learning rate
    current_lr = lr
    prev_val = _smooth_max_grad(h, beta)[0]
    
    for i in range(steps):
        if time.time() > _DEADLINE - 5:
            break
            
        val, grad = _smooth_max_grad(h, beta)
        
        # Simple backtracking line search
        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        # If function value increases, reduce step size
        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            # Increase step size slightly if progress was good
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            # No progress, reduce learning rate
            current_lr = max(current_lr * 0.5, 1e-6)
            
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        # Ensure strict feasibility
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Random starts
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)
        
    # 2. Optimization Phases
    # Phase 1: Coarse optimization with lower beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=300, lr=0.05)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=500, lr=0.01)
        _update_best(h_opt)
        
    # Phase 3: High beta (sharper max approximation)
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _pgd_optimize(best_h, beta=100.0, steps=500, lr=0.005)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt = _pgd_optimize(best_h, beta=200.0, steps=200, lr=0.001)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
