"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Start with ones (score 1.0)
    h = np.ones(N)
    best_h = h.copy()
    best_value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    
    # Strategy: use ADMM-like approach to flatten autoconvolution
    # Minimize max |conv(h,h) - c| subject to sum(h) = 1
    # We'll use a smooth approximation and gradient descent with projection
    
    # First, try a simple approach: make h have alternating signs to reduce peak
    # conv(h,h) for alternating signs will have cancellation
    
    # Try: h[i] = 1 for even i, -1 for odd i
    # This gives sum = 0, which is invalid. Need sum != 0.
    
    # Let's try h[i] = 1 + eps * (-1)^i for small eps
    # Or better: use a structured pattern
    
    # Approach: minimize the max of conv(h,h) using subgradient method
    # with constraint sum(h) = 1
    
    h = np.ones(N) / N  # normalize so sum = 1
    best_value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    best_h = h.copy()
    
    # Use iterative refinement: try to reduce the peak of autoconvolution
    # by adjusting h based on where the peak occurs
    
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        conv = np.convolve(h, h)
        S = np.sum(h)
        if abs(S) < 1e-10:
            h = best_h.copy()
            continue
        
        # The objective is max(conv), we want to minimize this
        # Subgradient: the gradient of max(conv) w.r.t. h is the conv of the indicator
        # of where max occurs with h
        
        max_val = np.max(conv)
        # Find indices where conv is close to max
        mask = (conv >= max_val - 1e-6)
        
        # Gradient of conv[j] w.r.t. h[i] is h[j-i] if j-i is in [0, N-1], 
        # and h[i-j] if i-j is in [0, N-1] (for the other part of full conv)
        # Actually, conv_full[k] = sum_i h[i] * h[k-i] for valid indices
        # d(conv_full[k])/d(h[i]) = h[k-i] + h[i-k] (with appropriate bounds)
        
        # For subgradient descent on max, we need gradient of the max component
        # Let's compute gradient numerically for the max components
        
        grad = np.zeros(N)
        for k in np.where(mask)[0]:
            for i in range(N):
                # d(conv[k])/d(h[i])
                # conv[k] = sum_j h[j] * h[k-j] where 0<=j<N and 0<=k-j<N
                # d/dh[i] = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
                # But wait, when we differentiate sum_j h[j]*h[k-j] w.r.t. h[i]:
                # term j=i: h[k-i]
                # term k-j=i => j=k-i: h[i] * 1, but that's h[k-i]... 
                # Actually it's: if i is in range, h[k-i] + (if k-i != i) h[k-i]
                # No: d/dh[i] [sum_j h[j] h[k-j]] = h[k-i] + h[i-(k-i)]... 
                # Let me be careful: conv[k] = sum_{j=max(0,k-N+1)}^{min(k,N-1)} h[j] h[k-j]
                # d(conv[k])/d(h[i]) = h[k-i] (from j=i term, if valid) + h[i-(k-i)]... 
                # No. d/dh[i] of h[j]*h[k-j]:
                # if j = i: contributes h[k-i]
                # if k-j = i, i.e., j = k-i: contributes h[i]
                # So total: h[k-i] (if 0<=i<N and 0<=k-i<N) + h[i] (if 0<=k-i<N and 0<=i<N)
                # Wait, that's not right either. Let me think again.
                # conv[k] = sum_j h[j] * h[k-j]
                # d/dh[i] conv[k] = sum_j [delta_{j,i} h[k-j] + h[j] delta_{k-j,i}]
                # = h[k-i] (if 0<=k-i<N) + sum_j h[j] delta_{j,k-i}
                # = h[k-i] (if 0<=k-i<N) + h[k-i] (if 0<=k-i<N)
                # = 2 * h[k-i] (if 0<=k-i<N)
                # Hmm, that seems off. Let me verify with small example.
                # h = [a, b], conv = [a^2, ab, b^2]
                # d(conv[1])/da = b, d(conv[1])/db = a
                # Using formula: for k=1, i=0: 2*h[1-0] = 2*b. But actual is b.
                # So the formula gives 2*h[k-i] but actual is h[k-i].
                # The issue: when j=i and k-j=i, they're the same term if i=k/2.
                # More carefully: d/dh[i] sum_j h[j]h[k-j]
                # = sum_j (delta_{ji} h[k-j] + h[j] delta_{k-j,i})
                # = h[k-i] + h[k-i]  [if both indices valid and distinct]
                # But if k-i = i, i.e., k=2i, then it's the same term counted once.
                # So: if k=2i: h[k-i] = h[i], counted once
                #     else: 2*h[k-i]
                
                ki = k - i
                if 0 <= ki < N:
                    if k == 2 * i:
                        grad[i] += h[ki]
                    else:
                        grad[i] += 2 * h[ki]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            h = h - lr * grad / grad_norm
        
        # Project to sum = 1
        S = np.sum(h)
        if abs(S) > 1e-10:
            h = h * (1.0 / S)
        
        # Evaluate
        conv = np.convolve(h, h)
        S = np.sum(h)
        if abs(S) < 1e-10:
            h = best_h.copy()
            continue
        value = abs(2 * N * np.max(conv) / S ** 2)
        if value < best_value:
            best_value = value
            best_h = h.copy()
    
    # Final check
    S = np.sum(best_h)
    if abs(S) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
