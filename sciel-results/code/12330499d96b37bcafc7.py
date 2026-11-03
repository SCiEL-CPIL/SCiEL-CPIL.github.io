import time
import numpy as np

N = 100
_DEADLINE = time.time() + 135

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

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    y = np.asarray(y, dtype=np.float64)
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

def _compute_r(h):
    """Compute linear autocorrelation r_k"""
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]

def _grad_min_r(h, beta):
    """
    Compute gradient of -log-sum-exp(-beta * r_k)
    This is equivalent to a weighted sum of gradients of r_k,
    where weights are proportional to exp(-beta * r_k).
    """
    n = len(h)
    r = _compute_r(h)
    
    # Weights for log-sum-exp of -r
    # We want to maximize min(r_k), so we minimize max(-r_k).
    # Let f(h) = -1/beta * log(sum(exp(-beta * r_k)))
    # grad f = sum w_k * grad r_k
    # where w_k = exp(-beta * r_k) / sum(exp(-beta * r_j))
    
    # Shift to prevent underflow
    r_min = np.min(r)
    exp_terms = np.exp(-beta * (r - r_min))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    grad = np.zeros(n)
    for k in range(n):
        if w[k] < 1e-12:
            continue
        # grad r_k
        # r_k = sum_{i=k}^{n-1} h_i h_{i-k}
        # d(r_k)/d(h_j) = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n)
        
        # Vectorized computation
        # h_{j-k} for j>=k is h[0:n-k] shifted?
        # Let's just loop k, it's N=100, so 100 iterations.
        if k > 0:
            grad[k:] += w[k] * h[:n-k]
        if k < n:
            grad[:n-k] += w[k] * h[k:]
            
    return grad

def _pgd_optimize(h0, beta, steps, lr):
    h = np.asarray(h0, dtype=np.float64).copy()
    n = len(h)
    target_sum = n / 2.0
    
    current_lr = lr
    prev_val = _score(h)
    
    for i in range(steps):
        if time.time() > _DEADLINE - 5:
            break
            
        grad = _grad_min_r(h, beta)
        
        new_h = h + current_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        
        new_val = _score(new_h_proj)
        
        # Line search
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            # Increase learning rate slightly if progress
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.5)
        else:
            # Halve learning rate
            current_lr *= 0.5
            # If learning rate is too small, break or restart?
            if current_lr < 1e-6:
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
    # 1. Constant
    h_const = np.full(N, 0.5)
    _update_best(h_const)
    
    # 2. Random
    for _ in range(10):
        h_rand = np.random.uniform(0, 1, N)
        _update_best(h_rand)
        
    # 3. Smooth waves
    i = np.arange(N)
    h_cos = 0.5 + 0.4 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Optimize from best candidate
    if best_h is not None:
        # Use a moderate beta to start
        h_opt = _pgd_optimize(best_h, beta=10.0, steps=500, lr=0.1)
        _update_best(h_opt)
        
        # Increase beta for sharper focus on the minimum r_k
        if time.time() - t0 < _DEADLINE - 30:
            h_opt = _pgd_optimize(best_h, beta=50.0, steps=500, lr=0.05)
            _update_best(h_opt)
            
        # Very sharp beta
        if time.time() - t0 < _DEADLINE - 10:
            h_opt = _pgd_optimize(best_h, beta=200.0, steps=500, lr=0.01)
            _update_best(h_opt)
            
        # Final polish
        if time.time() - t0 < _DEADLINE - 2:
            h_opt = _pgd_optimize(best_h, beta=1000.0, steps=200, lr=0.001)
            _update_best(h_opt)
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h
