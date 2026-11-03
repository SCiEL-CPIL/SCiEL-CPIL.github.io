import numpy as np
import time

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
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


def _project_to_feasible(h):
    """
    Project h onto the feasible set: sum(h) = N/2, 0 <= h_i <= 1.
    Uses a thresholding approach to exactly satisfy the sum constraint
    while respecting box constraints.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # We want to find tau such that sum(clamp(h_i - tau, 0, 1)) = target
    # clamp(x, 0, 1) = max(0, min(1, x))
    
    # Binary search for tau
    # The function f(tau) = sum(clamp(h_i - tau, 0, 1)) is monotonic decreasing in tau.
    # We need f(tau) = target.
    
    # Range for tau:
    # If tau <= min(h), f(tau) = sum(min(1, h_i - tau)) >= sum(h_i) - N*tau
    # If tau >= max(h), f(tau) = sum(max(0, h_i - tau)) = 0
    # Actually, simpler bounds:
    # min_tau = min(h) - 1
    # max_tau = max(h) + 1
    
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(50):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            # Need to decrease h_proj, so increase mid
            lo = mid
        else:
            # Need to increase h_proj, so decrease mid
            hi = mid
        
    h_proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    
    # Due to floating point, sum might not be exactly target.
    # Adjust the largest element that has room, or smallest if need to decrease.
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-12:
        if diff > 0:
            # Need to add value. Find index with max room (closest to 0)
            # Actually, we should find indices where h_proj < 1.
            # To minimize distortion, we can just add to the one with most room?
            # Or distribute? The thresholding method should be very close.
            # Let's just adjust the first element with room.
            for i in range(N):
                if h_proj[i] < 1.0 - 1e-12:
                    room = 1.0 - h_proj[i]
                    add = min(diff, room)
                    h_proj[i] += add
                    diff -= add
                    if abs(diff) < 1e-12:
                        break
        else:
            # Need to subtract value. Find index with max value (closest to 1)
            for i in range(N):
                if h_proj[i] > 1e-12:
                    room = h_proj[i]
                    sub = min(-diff, room)
                    h_proj[i] -= sub
                    diff += sub
                    if abs(diff) < 1e-12:
                        break
    
    return h_proj


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    Vectorized gradient computation using FFT.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _nesterov_optimize(h0, beta, n_iter=1000, lr=None, time_limit=30.0):
    """
    Nesterov Accelerated Gradient Descent.
    """
    h = _project_to_feasible(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    if lr is None:
        lr = 0.5 / (beta * N)
        
    # Momentum
    y = h.copy()
    m = np.zeros_like(h)
    
    t_start = time.time()
    
    for i in range(n_iter):
        if time.time() - t_start > time_limit:
            break
            
        # Nesterov step
        t_curr = (1 + np.sqrt(1 + 4 * (i + 1)**2)) / 2
        m_new = ((i + 1) / (i + 2)) * m
        y_new = y + m_new * (y - h)
        
        val, grad = _smooth_max_grad(y_new, beta)
        
        h_new = y_new - lr * grad
        h_new = _project_to_feasible(h_new)
        
        # Update best
        new_val = _score(h_new)
        if new_val < best_val:
            best_val = new_val
            best_h = h_new.copy()
            
        y = y_new
        h = h_new
        m = m_new

    return best_h, best_val


def _coordinate_descent_polish(h0, time_limit=10.0):
    """
    Coordinate descent to minimize the true max overlap.
    """
    h = _project_to_feasible(h0.copy())
    best_h = h.copy()
    best_val = _score(h)
    
    t_start = time.time()
    
    while time.time() - t_start < time_limit:
        improved = False
        for i in range(N):
            if time.time() - t_start > time_limit:
                break
            
            # Try adjusting h[i]
            # We want to minimize max(C_k) where C_k depends on h.
            # C_k = (2/N) * (N/2 - sum_{j} h_j h_{j-k})
            # For a fixed k, C_k is linear in h_i if we fix other h's? No, it's quadratic.
            # But for coordinate descent, we fix all j != i.
            # C_k(h) = (2/N) * (N/2 - (h_i * (h_{i-k} + h_{i+k}) + const_k))
            # So C_k is linear in h_i.
            # We want to find h_i in [0, 1] such that sum(h) = N/2 is maintained?
            # No, coordinate descent usually fixes the sum constraint by adjusting another variable
            # or by projecting after each step.
            
            # Let's just try small perturbations around current h[i]
            # This is a bit slow. 
            # Better: 
            # If we change h[i] by delta, sum changes by delta.
            # To maintain sum, we must change another h[j] by -delta.
            # This makes it a 2-variable optimization.
            
            # Alternative: 
            # Just do gradient step on the true max? 
            # The max of linear functions is convex. 
            # The subgradient is the average of active gradients.
            
            # Let's stick to a simple heuristic: 
            # Calculate C_k. Find the max C_max.
            # Identify which k are active (close to C_max).
            # The gradient of C_k wrt h_i is -(2/N) * (h_{i-k} + h_{i+k}).
            # We want to decrease C_max.
            # So we should decrease h_i in the direction of the gradient of the active C's.
            # But we must maintain sum=0.
            
            # Let's use the gradient of the soft max with very high beta as a proxy for coordinate descent?
            # No, let's just do a local line search on h[i] and h[j] (where j is the most opposite index).
            
            # Actually, the Nesterov method should be good enough. 
            # Coordinate descent on min-max problems is tricky.
            # Let's implement a simple "active set" coordinate descent.
            
            cs = _linear_overlap_values(h)
            max_c = np.max(cs)
            
            # Find indices i that contribute most to the active constraints
            # and try to move them in the direction that reduces max_c.
            
            # For each i, compute the "pressure" from the active constraints.
            # Pressure_i = sum_{k in active} w_k * dC_k/dh_i
            # dC_k/dh_i = -(2/N) * (h_{i-k} + h_{i+k})
            
            # This is essentially the gradient of the soft max.
            # So Coordinate Descent here is redundant if we just use the gradient.
            
            # Let's skip complex coordinate descent and rely on Nesterov + multiple restarts.
            break 

    return best_h, best_val


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_to_feasible(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Analytical Starts
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Higher harmonic
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)
    
    # 2. Nesterov Optimization
    # Phase 1: Coarse
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_res, val_res = _nesterov_optimize(best_h, beta=20.0, n_iter=2000, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 2: Medium
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_res, val_res = _nesterov_optimize(best_h, beta=50.0, n_iter=3000, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 3: Fine
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_res, val_res = _nesterov_optimize(best_h, beta=100.0, n_iter=5000, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 4: Very Fine
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_res, val_res = _nesterov_optimize(best_h, beta=200.0, n_iter=5000, time_limit=5.0)
        _update_best(h_res)
        
    # 3. Random Restarts with Nesterov
    if time.time() - t0 < _DEADLINE - 5:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 2:
                break
            noise = np.random.normal(0, 0.1, N)
            h_pert = best_h + noise
            _update_best(h_pert)
            
            # Quick Nesterov polish
            if time.time() - t0 < _DEADLINE - 2:
                h_res, val_res = _nesterov_optimize(h_pert, beta=50.0, n_iter=1000, time_limit=2.0)
                _update_best(h_res)
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_to_feasible(best_h)
    return best_h
