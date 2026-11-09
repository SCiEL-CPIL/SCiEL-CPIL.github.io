import numpy as np
import time

N = 50

def compute_ratio(h):
    """
    Compute the ratio (L2^2) / (L1 * SupConv) for normalized h (sum=1).
    If h is not normalized, we normalize it first.
    """
    h = np.asarray(h)
    s = np.sum(h)
    if s == 0:
        return 0.0
    h_norm = h / s
    l2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    sup_conv = np.max(conv)
    if sup_conv == 0:
        return 0.0
    return l2_sq / sup_conv

def optimize(h0, steps=2000, lr=0.01):
    """
    Gradient ascent on the simplex.
    """
    h = h0.copy()
    s = np.sum(h)
    if s == 0:
        h = np.ones(N) / N
    else:
        h = h / s
    
    best_h = h.copy()
    best_ratio = compute_ratio(h)
    
    for _ in range(steps):
        # Compute gradient of log(ratio) or ratio?
        # Ratio R = A / B, where A = sum h^2, B = max conv.
        # Let's use finite differences or analytic gradient.
        # Analytic gradient is tricky due to max.
        # Let's use a simple numerical gradient or random perturbations.
        
        # Random perturbation + local search is often more robust for this small N.
        # But let's try a few steps of gradient.
        
        # Gradient of A = 2h.
        # Gradient of B: B = max_k (h * h)_k.
        # Let k* be the index of max conv.
        # (h*h)_k = sum_i h_i h_{k-i}.
        # d/dh_j (h*h)_k = h_{k-j} + h_{k-j} (if k-j != j) or 2 h_j (if k-j=j).
        # Basically, the gradient of the conv at the peak index.
        
        conv = np.convolve(h, h, mode='full')
        k_star = np.argmax(conv)
        
        # Gradient of conv at k_star
        grad_conv = np.zeros(N)
        for j in range(N):
            # h_j appears in h_j * h_{k_star - j} and h_{k_star - j} * h_j
            # If k_star - j is in [0, N-1], it contributes.
            idx = k_star - j
            if 0 <= idx < N:
                grad_conv[j] += h[idx]
                if idx != j:
                    grad_conv[j] += h[idx]
            # Wait, convolution is sum_i h_i h_{k-i}.
            # Term for j is h_j h_{k-j} + h_{k-j} h_j if k-j != j?
            # No, the sum is over i.
            # If i=j, term is h_j h_{k-j}.
            # If i=k-j, term is h_{k-j} h_j.
            # If j = k-j, i.e., 2j=k, then it's the same term h_j^2, derivative is 2h_j.
            # If j != k-j, there are two terms in the sum: one where i=j, one where i=k-j.
            # So derivative is h_{k-j} + h_{k-j} = 2 h_{k-j}.
            
        # Let's re-calculate carefully.
        # C_k = sum_{i=0}^{N-1} h_i h_{k-i} (with h_x=0 if x out of bounds).
        # d C_k / d h_j = h_{k-j} + h_{k-j} (if k-j != j) 
        # Actually, d/dh_j (h_i h_{k-i}) is non-zero only if i=j or k-i=j.
        # If i=j, term is h_j h_{k-j}. Deriv is h_{k-j}.
        # If k-i=j => i=k-j. Term is h_{k-j} h_j. Deriv is h_j? No, deriv wrt h_j of h_{k-j} h_j is h_{k-j}.
        # So if j != k-j, we have two contributions: h_{k-j} and h_{k-j}. Total 2 h_{k-j}.
        # If j == k-j, we have one contribution: h_j. Deriv of h_j^2 is 2h_j.
        
        grad_conv = np.zeros(N)
        for j in range(N):
            idx = k_star - j
            if 0 <= idx < N:
                if idx == j:
                    grad_conv[j] = 2 * h[j]
                else:
                    grad_conv[j] = 2 * h[idx]
        
        # R = A / B
        # dR = (B dA - A dB) / B^2
        # dA = 2h . dh
        # dB = grad_conv . dh
        # We want to maximize R, so we move in direction of gradient.
        # grad_R = (B * 2h - A * grad_conv) / B^2
        # Since we only care about direction, we can use:
        # direction = B * 2h - A * grad_conv
        
        A = np.sum(h**2)
        B = conv[k_star]
        
        if B == 0:
            break
            
        direction = B * 2 * h - A * grad_conv
        
        # Project direction onto simplex?
        # Simple step: h_new = h + lr * direction
        # Then normalize and clip.
        
        h_new = h + lr * direction
        h_new = np.maximum(h_new, 0)
        s_new = np.sum(h_new)
        if s_new == 0:
            h_new = np.ones(N) / N
        else:
            h_new = h_new / s_new
            
        r_new = compute_ratio(h_new)
        
        if r_new > best_ratio:
            best_ratio = r_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.9
            if lr < 1e-6:
                break
                
    return best_ratio, best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_ratio = 0.0
    best_h = np.ones(N) / N
    
    # Initial best
    r_seed, _ = compute_ratio(best_h)
    best_ratio = r_seed
    
    # Strategies
    # 1. Random starts
    # 2. Structured starts (bumps, spikes)
    
    num_starts = 200
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        if i % 4 == 0:
            h0 = np.random.rand(N)
        elif i % 4 == 1:
            # Gaussian
            x = np.linspace(-1, 1, N)
            sigma = 0.1 + 0.4 * np.random.rand()
            h0 = np.exp(-x**2 / (2*sigma**2))
        elif i % 4 == 2:
            # Sparse spike
            h0 = np.zeros(N)
            idx = np.random.randint(N)
            h0[idx] = 1.0
            # Add some noise
            h0 += np.random.rand(N) * 0.1
        else:
            # Block
            h0 = np.zeros(N)
            start = np.random.randint(N-5)
            width = np.random.randint(2, 10)
            h0[start:start+width] = 1.0
            h0 += np.random.rand(N) * 0.1

        r, h_opt = optimize(h0, steps=500, lr=0.05)
        
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt

    return list(best_h)
