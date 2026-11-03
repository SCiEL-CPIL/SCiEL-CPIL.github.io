"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses vectorized FFT-based gradients with multi-phase PGD and exact polish."""
import time
import numpy as np

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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _weighted_grad_cs(h, w):
    """
    Compute sum_k w_k * dC_k/dh.
    dC_k/dh_j = -(2/n) * (h[j-k] if j>=k else 0 + h[j+k] if j+k<n else 0)
    sum_k w_k * dC_k/dh_j = -(2/n) * (sum_{k=0}^{j} w[k]*h[j-k] + sum_{k=0}^{n-1-j} w[k]*h[j+k])
    """
    n = len(h)
    
    # First sum: for each j, sum_{k=0}^{j} w[k]*h[j-k]
    # This is the linear convolution of w and h, taking the first n elements
    first_sum = np.convolve(w, h, 'full')[:n]
    
    # Second sum: for each j, sum_{k=0}^{n-1-j} w[k]*h[j+k]
    # For n=100, this is fast enough with a loop
    second_sum = np.zeros(n)
    for j in range(n):
        second_sum[j] = np.dot(w[:n-j], h[j:n])
    
    grad = -(2.0 / n) * (first_sum + second_sum)
    return grad


def _smooth_max_value_grad(h, beta):
    """
    Compute the log-sum-exp smooth max value and its gradient.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    
    # Numerical stability
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = m + np.log(sum_exp) / beta
    
    grad = _weighted_grad_cs(h, w)
    
    return val, grad


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
        
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _pgd_optimize(h0, beta, steps, lr, deadline):
    """
    Projected Gradient Descent with backtracking line search.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_lr = lr
    prev_val, _ = _smooth_max_value_grad(h, beta)
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        val, grad = _smooth_max_value_grad(h, beta)
        
        # Backtracking line search
        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_value_grad(new_h_proj, beta)
        
        while new_val > prev_val and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_value_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            if i % 20 == 0:
                current_lr = min(current_lr * 1.03, 0.05)
        else:
            current_lr = max(current_lr * 0.5, 1e-7)
            
    return h


def _exact_polish(h, steps=200, deadline=None):
    """
    Final polish using exact max with subgradient method.
    """
    h = h.copy()
    target_sum = N / 2.0
    lr = 0.001
    
    for i in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        cs = _linear_overlap_values(h)
        max_val = np.max(cs)
        # Subgradient: average over all maximizers
        max_indices = np.where(cs >= max_val - 1e-12)[0]
        w = np.zeros(N)
        w[max_indices] = 1.0 / len(max_indices)
        
        grad = _weighted_grad_cs(h, w)
        
        new_h = h - lr * grad
        new_h = _project_sum_box(new_h, target_sum)
        
        # Check if it improved or is neutral
        new_cs = _linear_overlap_values(new_h)
        if np.max(new_cs) <= max_val + 1e-14:
            h = new_h
        else:
            lr /= 2.0
            if lr < 1e-10:
                break
                
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # Initial Candidates
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)
    
    # Multiple frequency candidates
    for freq in [2, 3, 4, 5]:
        h_f = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_f = _project_sum_box(h_f, N / 2.0)
        _update_best(h_f)
        
    # Random starts
    for _ in range(8):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)
        
    # Optimization Phases
    # Phase 1: Coarse optimization
    if best_h is not None and time.time() - t0 < _DEADLINE - 80:
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=500, lr=0.03, deadline=_DEADLINE - 80)
        _update_best(h_opt)
        
    # Phase 2: Medium beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _pgd_optimize(best_h, beta=50.0, steps=800, lr=0.01, deadline=_DEADLINE - 40)
        _update_best(h_opt)
        
    # Phase 3: High beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _pgd_optimize(best_h, beta=150.0, steps=500, lr=0.003, deadline=_DEADLINE - 15)
        _update_best(h_opt)
        
    # Phase 4: Ultra-high beta polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _pgd_optimize(best_h, beta=400.0, steps=300, lr=0.001, deadline=_DEADLINE - 5)
        _update_best(h_opt)
        
    # Final exact polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _exact_polish(best_h, steps=100, deadline=_DEADLINE - 1)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    
    return best_h
