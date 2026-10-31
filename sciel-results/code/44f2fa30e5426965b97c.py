import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave margin before 150s kill

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for step function on [-1/4, 1/4]."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    dx = 1.0 / N  # interval width
    l1 = dx * np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = dx * np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_scaled = conv * dx
    conv_inf = np.max(conv_scaled)
    if conv_inf <= 0:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def compute_ratio_normalized(h):
    """Compute ratio with l1 normalized to 1, for easier optimization."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf = np.max(conv)
    if conv_inf <= 0:
        return 0.0
    dx = 1.0 / N
    return l2_sq / (dx * l1 * conv_inf)

def soft_max(x, beta=50.0):
    """Differentiable approximation of max(x) using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if m < -1e10:
        return m
    # Stable log-sum-exp: m + log(sum(exp(x - m)))
    shifted = x - m
    exp_sum = np.sum(np.exp(beta * shifted))
    return m + np.log(exp_sum) / beta

def soft_max_grad(x, beta=50.0):
    """Gradient of soft_max w.r.t. x."""
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    shifted = x - m
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    return weights

def smooth_negative_ratio(h, beta=50.0):
    """
    Smooth surrogate: minimize -log(ratio) or equivalently maximize log(ratio).
    ratio = l2_sq / (dx * l1 * conv_inf)
    log(ratio) = log(l2_sq) - log(dx) - log(l1) - log(conv_inf)
    
    We want to maximize log(ratio), so minimize -log(ratio).
    Use soft_max for conv_inf to make it differentiable.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-10)  # Ensure positivity for log
    
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf_smooth = soft_max(conv, beta=beta)
    
    dx = 1.0 / N
    
    if l1 <= 1e-10 or l2_sq <= 1e-10 or conv_inf_smooth <= 1e-10:
        return 1e10
    
    log_ratio = np.log(l2_sq) - np.log(dx) - np.log(l1) - np.log(conv_inf_smooth)
    return -log_ratio

def smooth_negative_ratio_grad(h, beta=50.0):
    """
    Gradient of smooth_negative_ratio w.r.t. h.
    -log(ratio) = -log(l2_sq) + log(dx) + log(l1) + log(conv_inf)
    
    d/dh_i [log(l2_sq)] = 2*h_i / l2_sq
    d/dh_i [log(l1)] = 1/l1
    d/dh_i [log(conv_inf)] = d/dh_i [log(soft_max(conv))]
    
    soft_max(conv) = m + (1/beta)*log(sum(exp(beta*(conv_j - m))))
    d/dconv_k [soft_max] = exp(beta*(conv_k - m)) / sum(exp(beta*(conv_j - m))) = weights[k]
    d/dh_i [conv_k] = sum_j (h_{k-i+j} if k-i+j in range) ... 
    Actually conv_k = sum_j h_j * h_{k-j} (for full convolution)
    d/dh_i [conv_k] = h_{k-i} + h_{k-N+1-i} if valid, but let's use the standard formula.
    
    For full convolution of length N with itself, result has length 2N-1.
    conv_k = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h_j * h_{k-j}
    
    d/dh_i conv_k = h_{k-i} (if 0<=k-i<N) + h_{k-i} (if 0<=k-i<N) ... wait.
    conv_k = sum_j h_j h_{k-j}
    d/dh_i conv_k = sum_j (delta_{j,i} h_{k-j} + h_j delta_{k-j,i})
                  = h_{k-i} + h_{k-i} if k-i is in [0,N-1]
                  = 2*h_{k-i} if k-i in [0,N-1]
    
    So d/dh_i [log(conv_inf)] = sum_k weights[k] * 2*h_{k-i} / conv_inf
    where weights[k] = exp(beta*(conv_k - m)) / sum(exp(beta*(conv_j - m)))
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 1e-10)
    
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    conv_inf_smooth = soft_max(conv, beta=beta)
    
    if l1 <= 1e-10 or l2_sq <= 1e-10 or conv_inf_smooth <= 1e-10:
        return np.zeros_like(h)
    
    # Gradient of -log(ratio) = -log(l2_sq) + log(dx) + log(l1) + log(conv_inf)
    # d/dh_i [-log(l2_sq)] = -2*h_i / l2_sq
    # d/dh_i [log(l1)] = 1/l1
    # d/dh_i [log(conv_inf)] = sum_k weights[k] * (d/dh_i conv_k) / conv_inf_smooth
    #                        = sum_k weights[k] * 2*h_{k-i} / conv_inf_smooth (if k-i in range)
    
    weights = soft_max_grad(conv, beta=beta)
    
    # Compute sum_k weights[k] * h_{k-i} for each i
    # conv has length 2N-1, index k from 0 to 2N-2
    # For fixed i, k-i must be in [0, N-1], so k in [i, i+N-1]
    # h_{k-i} where k-i ranges from 0 to N-1
    
    # Let's compute this efficiently
    # term_i = sum_{k=i}^{i+N-1} weights[k] * h[k-i]
    # This is like a cross-correlation
    
    # Create a matrix or use sliding window
    # For each i, we need sum of weights[i:i+N] * h[0:N]
    
    # Use np.correlate or manual
    # weights has length 2N-1
    # For i=0: sum_{k=0}^{N-1} weights[k] * h[k]
    # For i=1: sum_{k=1}^{N} weights[k] * h[k-1]
    # ...
    # For i=N-1: sum_{k=N-1}^{2N-2} weights[k] * h[k-(N-1)]
    
    # This is the convolution of weights (reversed?) with h, but let's just do it directly
    # Actually, it's the cross-correlation of weights and h
    
    # Let's use a simple loop since N=50 is small
    conv_term = np.zeros(N)
    for i in range(N):
        k_start = i
        k_end = i + N
        if k_end > len(weights):
            k_end = len(weights)
        conv_term[i] = np.dot(weights[k_start:k_end], h[0:k_end-k_start])
    
    # d/dh_i [log(conv_inf)] = 2 * conv_term[i] / conv_inf_smooth
    grad_conv = 2.0 * conv_term / conv_inf_smooth
    
    # Total gradient
    grad = -2.0 * h / l2_sq + 1.0 / l1 + grad_conv
    
    return grad

def gradient_descent_smooth(h0, max_iter=5000, lr=0.1, beta=50.0, restarts=5):
    """
    Gradient descent on the smooth surrogate.
    """
    h = h0.copy().astype(float)
    best_h = h.copy()
    best_ratio = compute_ratio_normalized(h)
    
    for restart in range(restarts):
        # Add small noise for restart
        if restart > 0:
            h = h0.copy() + np.random.normal(0, 0.1, size=N)
            h = np.maximum(h, 0.0)
        
        current_loss = smooth_negative_ratio(h, beta=beta)
        
        for i in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return best_h, best_ratio
            
            grad = smooth_negative_ratio_grad(h, beta=beta)
            
            # Gradient step
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0.0)
            
            new_loss = smooth_negative_ratio(h_new, beta=beta)
            
            if new_loss < current_loss:
                h = h_new
                current_loss = new_loss
                # Check actual ratio
                actual_ratio = compute_ratio_normalized(h)
                if actual_ratio > best_ratio:
                    best_ratio = actual_ratio
                    best_h = h.copy()
            else:
                # Reduce learning rate
                lr *= 0.99
            
            # Adaptive beta: increase as we get closer
            beta = min(100.0, 50.0 + i * 0.01)
        
        # After each restart, try to polish
        if time.time() - START_TIME < TIME_LIMIT - 10:
            h_polish, ratio_polish = polish_local(h, max_iter=2000)
            if ratio_polish > best_ratio:
                best_ratio = ratio_polish
                best_h = h_polish.copy()
    
    return best_h, best_ratio

def polish_local(h0, max_iter=2000, lr=0.01):
    """
    Local polish using simulated annealing on the exact ratio.
    """
    h = h0.copy().astype(float)
    best_h = h.copy()
    best_ratio = compute_ratio_normalized(h)
    current_ratio = best_ratio
    temp = 0.1
    
    for i in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Random perturbation
        h_new = h.copy()
        idx = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        h_new[idx] = max(0, h_new[idx] + delta)
        
        new_ratio = compute_ratio_normalized(h_new)
        
        if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temp):
            h = h_new
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = h.copy()
        
        temp *= 0.999
    
    return best_h, best_ratio

def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) using smooth spectral design
    with gradient-based optimization on a differentiable surrogate.
    """
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Initialize with several random starting points
    rng = np.random.RandomState(42)
    
    # Start with sparse binary-like patterns (Sidon-inspired)
    for n_ones in [5, 7, 10, 13]:
        for seed in range(5):
            h_cand = np.zeros(N)
            positions = rng.choice(N, size=n_ones, replace=False)
            h_cand[positions] = 1.0
            ratio = compute_ratio_normalized(h_cand)
            if ratio > best_ratio:
                best_ratio = ratio
                best_h = h_cand.copy()
    
    # Try uniform initialization
    h_uniform = np.ones(N)
    ratio = compute_ratio_normalized(h_uniform)
    if ratio > best_ratio:
        best_ratio = ratio
        best_h = h_uniform.copy()
    
    # Gradient-based optimization on smooth surrogate
    if time.time() - START_TIME < TIME_LIMIT - 30:
        h_opt, ratio_opt = gradient_descent_smooth(best_h, max_iter=3000, lr=0.1, beta=50.0, restarts=3)
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_h = h_opt
    
    # Polish with local search
    if time.time() - START_TIME < TIME_LIMIT - 10:
        h_polish, ratio_polish = polish_local(best_h, max_iter=2000)
        if ratio_polish > best_ratio:
            best_ratio = ratio_polish
            best_h = h_polish
    
    # Try a few more random starts with full optimization
    for _ in range(3):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        n_ones = rng.randint(5, 15)
        h_cand = np.zeros(N)
        positions = rng.choice(N, size=n_ones, replace=False)
        h_cand[positions] = 1.0
        h_opt, ratio_opt = gradient_descent_smooth(h_cand, max_iter=2000, lr=0.1, beta=50.0, restarts=2)
        if ratio_opt > best_ratio:
            best_ratio = ratio_opt
            best_h = h_opt
    
    return [float(x) for x in best_h]
