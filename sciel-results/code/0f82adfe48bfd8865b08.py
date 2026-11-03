"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Nesterov accelerated gradient descent on a smooth-max surrogate 
with adaptive temperature and projection onto the constraint set.
"""
import time
import numpy as np

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145


def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _project(h, target_sum):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on the Lagrange multiplier for the sum constraint.
    """
    h = np.asarray(h, dtype=np.float64)
    low = -np.max(h) - 1.0
    high = 1.0 - np.min(h) + 1.0
    
    # If sum is already correct and within bounds, return h
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12 and np.all(h >= -1e-12) and np.all(h <= 1 + 1e-12):
        return np.clip(h, 0.0, 1.0)

    def sum_of_clipped(offset):
        return np.sum(np.clip(h + offset, 0.0, 1.0))

    # Bisection
    for _ in range(60):
        mid = (low + high) / 2.0
        s_mid = sum_of_clipped(mid)
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
    
    offset = (low + high) / 2.0
    h_proj = np.clip(h + offset, 0.0, 1.0)
    
    # Final correction for numerical precision
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        # Adjust the element with the most "room"
        if diff > 0:
            # Add to smallest
            idx = np.argmin(h_proj)
            h_proj[idx] += diff
        else:
            # Subtract from largest
            idx = np.argmax(h_proj)
            h_proj[idx] += diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj


def _smooth_max_obj_grad(h, beta):
    """
    Compute the value and gradient of the smooth max of C_k.
    F(h) = (1/beta) * log(sum(exp(beta * C_k)))
    grad F = sum(w_k * grad C_k)
    """
    cs = _linear_overlap_values(h)
    
    # Compute weights w_k
    # Shift to avoid overflow
    cs_max = np.max(cs)
    exp_cs = np.exp(beta * (cs - cs_max))
    sum_exp = np.sum(exp_cs)
    w = exp_cs / sum_exp
    
    val = (1.0 / beta) * (np.log(sum_exp) + beta * cs_max)
    
    # Compute gradient
    # grad C_k_j = -(2/N) * (h_{j-k} [j>=k] + h_{j+k} [j+k<N])
    # We need grad_F_j = sum_k w_k * grad C_k_j
    # grad_F_j = -(2/N) * ( sum_{k=0}^j w_k h_{j-k} + sum_{k=0}^{N-1-j} w_k h_{j+k} )
    
    # Let A_j = sum_{k=0}^j w_k h_{j-k}  (Convolution of w and h, truncated)
    # Let B_j = sum_{k=0}^{N-1-j} w_k h_{j+k} (Correlation-like)
    
    # Efficient calculation using FFT
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
        
    w_pad = np.zeros(n_fft)
    w_pad[:N] = w
    h_pad = np.zeros(n_fft)
    h_pad[:N] = h
    
    W = np.fft.rfft(w_pad)
    H = np.fft.rfft(h_pad)
    
    # Convolution w * h
    conv = np.fft.irfft(W * H, n=n_fft)
    # A_j is conv[j] for j=0..N-1? 
    # conv[j] = sum_{k=0}^j w[k] h[j-k] if we treat them as sequences starting at 0.
    # Yes, linear convolution.
    A = conv[:N]
    
    # For B_j = sum_{k=0}^{N-1-j} w_k h_{j+k}
    # Let k' = N-1-j-k. Then k = N-1-j-k'.
    # This is effectively a correlation.
    # Alternatively, note that B_j is the j-th element of the correlation of w and h?
    # Correlation c[j] = sum_k w[k] h[k+j].
    # We want sum_{k=0}^{N-1-j} w[k] h[k+j].
    # This is exactly the correlation at lag j.
    
    # Compute correlation of w and h
    # corr[j] = sum_k w[k] h[k+j]
    # Using FFT: corr = ifft(conj(fft(w)) * fft(h))
    # But we need to be careful with padding.
    
    # Let's use the same convolution trick.
    # Reverse w?
    # corr[j] = sum_{k} w[k] h[k+j]
    # Let w_rev[k] = w[N-1-k].
    # Then conv(w_rev, h)[N-1+j] = sum_{m} w_rev[m] h[N-1+j-m]
    # = sum_{m} w[N-1-m] h[N-1+j-m]
    # Let k = N-1-m => m = N-1-k
    # = sum_{k} w[k] h[j+k]
    # So corr[j] = conv(w_rev, h)[N-1+j]
    
    w_rev = w[::-1]
    w_rev_pad = np.zeros(n_fft)
    w_rev_pad[:N] = w_rev
    W_rev = np.fft.rfft(w_rev_pad)
    
    conv_rev = np.fft.irfft(W_rev * H, n=n_fft)
    B = conv_rev[N-1 : N-1+N]
    
    grad = -(2.0 / N) * (A + B)
    
    return val, grad


def _nesterov_accelerated_gradient(h0, beta, n_iter=500, lr=0.01):
    """
    Nesterov accelerated gradient descent on the smooth max objective
    with backtracking line search.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    
    # Initialize momentum
    h_prev = h.copy()
    t = 1.0
    
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        # Nesterov step: evaluate gradient at y = h + (t-1)/(t+1) * (h - h_prev)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        # Backtracking line search
        step_lr = lr
        h_new = None
        for bt in range(20):
            h_candidate = y - step_lr * grad
            h_candidate = _project(h_candidate, N / 2.0)
            val_candidate = _score(h_candidate)
            # Simple Armijo-like condition: accept if score doesn't increase too much
            # Since we're minimizing, accept if val_candidate < val_y + 0.5 * step_lr * np.sum(grad * (h_candidate - y))
            # But this is expensive; use a simpler heuristic
            if val_candidate <= val_y + 1e-6:
                h_new = h_candidate
                break
            step_lr *= 0.5
        
        if h_new is None:
            # Fallback: use the original step
            h_new = y - lr * grad
            h_new = _project(h_new, N / 2.0)
        
        # Update momentum
        h_prev = h
        h = h_new
        t = t_new
        
        # Check improvement on actual objective
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h, best_score


def _gradient_polish(h0, beta=500.0, max_time=15.0):
    """
    Refine h using Nesterov accelerated gradient descent with a high beta
    to approximate the true max, effectively polishing the solution.
    """
    h = np.array(h0, dtype=np.float64)
    h = _project(h, N / 2.0)
    best_h = h.copy()
    best_score = _score(h)
    
    start_time = time.time()
    lr = 0.02
    t = 1.0
    h_prev = h.copy()
    
    while time.time() < start_time + max_time:
        if time.time() > _DEADLINE - 2:
            break
            
        # Nesterov step
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        momentum = (t - 1) / t_new
        y = h + momentum * (h - h_prev)
        
        val_y, grad = _smooth_max_obj_grad(y, beta)
        
        # Step
        h_new = y - lr * grad
        h_new = _project(h_new, N / 2.0)
        
        s_new = _score(h_new)
        if s_new < best_score - 1e-12:
            best_score = s_new
            best_h = h_new.copy()
            # Adaptive learning rate: increase if improving
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease if not improving
            lr *= 0.95
            if lr < 1e-5:
                break
                
        h_prev = h
        h = h_new
        t = t_new
        
    return best_h, best_score


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # Phase 1: Nesterov Accelerated Gradient Descent with adaptive beta
    # We will run several iterations with increasing beta
    
    # Initial guess: Random or structured
    # Start with a few different initializations
    initializations = [
        np.full(N, 0.5),
        0.5 + 0.3 * np.sin(2 * np.pi * np.arange(N) / N),
        0.5 + 0.2 * np.random.randn(N)
    ]
    
    # Clip and project initializations
    inits = []
    for init in initializations:
        h_init = np.clip(init, 0.0, 1.0)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    # Add some random inits
    for _ in range(5):
        h_init = np.random.rand(N)
        h_init = _project(h_init, N / 2.0)
        inits.append(h_init)
        
    beta = 10.0
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 20:
            break
            
        for h_init in inits:
            if time.time() > _DEADLINE - 10:
                break
                
            # Run Nesterov with current beta
            h_opt, s_opt = _nesterov_accelerated_gradient(h_init, beta, n_iter=300, lr=0.05)
            
            if s_opt < best_score:
                best_score = s_opt
                best_h = h_opt.copy()
                
        # Increase beta for next restart
        beta *= 1.3
        
    # Phase 2: High-Precision Polish with Gradient Descent
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_ref = best_h.copy()
        h_ref, best_score = _gradient_polish(h_ref, beta=500.0, max_time=_DEADLINE - time.time() - 5)
        best_h = h_ref
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project(best_h, N / 2.0)
    return best_h
