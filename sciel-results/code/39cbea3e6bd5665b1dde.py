import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin
    
    # Initial guess: symmetric ones
    h = np.ones(N)
    h /= np.sum(h)
    
    best_h = h.copy()
    best_val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    alpha = 0.1
    step = 0
    
    while time.time() - start_time < timeout:
        # Compute autoconvolution
        c = np.convolve(h, h)
        current_val = abs(2 * N * np.max(c) / np.sum(h)**2)
        
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        # Find index of max
        k_max = np.argmax(c)
        
        # Compute gradient of c[k_max] wrt h
        # c[k] = sum_i h[i] h[k-i]
        # d c[k] / d h[j] = h[k-j] + h[j-k] (with bounds)
        # Efficient way:
        # grad[j] = h[k_max - j] + h[j - k_max]
        # Note: h is indexed 0..N-1.
        # If index out of bounds, term is 0.
        
        grad = np.zeros(N)
        # h[k_max - j] is non-zero if 0 <= k_max - j < N
        # j - k_max is non-zero if 0 <= j - k_max < N
        
        # Vectorized computation
        # Range of j for first term: k_max - (N-1) <= j <= k_max
        # Range of j for second term: k_max <= j <= k_max + N - 1
        
        # Term 1: h[k_max - j]
        idx1 = k_max - np.arange(N)
        mask1 = (idx1 >= 0) & (idx1 < N)
        grad[mask1] += h[idx1[mask1]]
        
        # Term 2: h[j - k_max]
        idx2 = np.arange(N) - k_max
        mask2 = (idx2 >= 0) & (idx2 < N)
        grad[mask2] += h[idx2[mask2]]
        
        # Update h
        h = h - alpha * grad
        
        # Project to sum=1
        h = h / np.sum(h)
        
        # Adaptive step size?
        # If value increases, reduce alpha
        # Simple restart if stuck
        step += 1
        if step % 100 == 0:
            alpha *= 0.9
            if alpha < 1e-6:
                break
                
    return best_h
