"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _overlap_at_shifts(h, shifts=None):
    """Compute overlap sum_i h[i]*(1-h[(i+s)%N]) for each shift s."""
    N = len(h)
    if shifts is None:
        shifts = np.arange(N)
    # For each shift s, overlap = sum_i h[i] * (1 - h[(i+s) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i+s)%N]
    # = N/2 - acf(s)
    # We compute acf(s) = sum_i h[i]*h[(i+s)%N]
    # Use FFT for speed
    h_ext = np.concatenate([h, h])
    # acf(s) = sum_{i=0}^{N-1} h[i]*h[(i+s)%N]
    # This is circular cross-correlation
    acf = np.zeros(N)
    for s in range(N):
        acf[s] = np.dot(h, np.roll(h, -s))
    overlaps = N / 2.0 - acf
    return overlaps

def _max_overlap_smooth(h, beta=100.0):
    """Log-sum-exp smoothing of max overlap."""
    overlaps = _overlap_at_shifts(h)
    # Clip to avoid numerical issues
    overlaps = np.clip(overlaps, -1e-10, None)
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    log_sum = np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    return m + log_sum

def _grad_max_overlap_smooth(h, beta=100.0):
    """Gradient of log-sum-exp smoothed max overlap."""
    N = len(h)
    overlaps = _overlap_at_shifts(h)
    overlaps = np.clip(overlaps, -1e-10, None)
    m = np.max(overlaps)
    if m < 1e-10:
        return np.zeros(N)
    weights = np.exp(beta * (overlaps - m))
    weights /= np.sum(weights)
    # d(overlap_s)/d(h[i]) = (1 - h[(i+s)%N]) - h[i] * (1-h[(i-s)%N]) ... 
    # overlap_s = sum_i h[i]*(1-h[(i+s)%N])
    # d/dh[i] overlap_s = (1-h[(i+s)%N]) + h[i]*d/dh[i](-h[(i+s)%N])
    # If i != (j-s)%N for the term, it's just (1-h[(i+s)%N])
    # But h[(i+s)%N] depends on h[i] only when (i+s)%N == i, i.e., s=0
    # More carefully: overlap_s = sum_j h[j]*(1-h[(j+s)%N])
    # d/dh[i] = (1-h[(i+s)%N]) + sum_j h[j]*d/dh[i](-h[(j+s)%N])
    # d/dh[i] h[(j+s)%N] = 1 if (j+s)%N == i, i.e., j = (i-s)%N
    # So = (1-h[(i+s)%N]) - h[(i-s)%N]
    grad = np.zeros(N)
    for s in range(N):
        # d overlap_s / d h[i] = (1 - h[(i+s)%N]) - h[(i-s)%N]
        d_h = (1.0 - np.roll(h, -s)) - np.roll(h, s)
        grad += weights[s] * d_h
    return grad

def construct():
    deadline = time.time() + 120.0
    best_h = np.full(N, 0.5)
    best_score = _max_overlap_smooth(best_h, beta=50.0)
    
    np.random.seed(42)
    
    # Multiple restarts with diverse initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Diverse initialization
        if restart == 0:
            h = np.full(N, 0.5)
        elif restart == 1:
            # Cosine pattern
            k = np.random.randint(1, N//2)
            h = 0.5 + 0.3 * np.cos(2 * np.pi * k * np.arange(N) / N)
            h = np.clip(h, 0.01, 0.99)
            # Normalize to sum = N/2
            h = h / np.sum(h) * (N / 2.0)
            h = np.clip(h, 0.0, 1.0)
        elif restart == 2:
            # Two-frequency cosine
            k1, k2 = np.random.randint(1, N//2, size=2)
            h = 0.5 + 0.2 * np.cos(2 * np.pi * k1 * np.arange(N) / N) + \
                    0.2 * np.cos(2 * np.pi * k2 * np.arange(N) / N)
            h = np.clip(h, 0.01, 0.99)
            h = h / np.sum(h) * (N / 2.0)
            h = np.clip(h, 0.0, 1.0)
        elif restart == 3:
            # Random with structure
            h = 0.5 + 0.3 * np.random.randn(N)
            h = np.clip(h, 0.01, 0.99)
            h = h / np.sum(h) * (N / 2.0)
            h = np.clip(h, 0.0, 1.0)
        else:
            # Perturb best
            h = best_h + 0.1 * np.random.randn(N)
            h = np.clip(h, 0.0, 1.0)
            h = h / np.sum(h) * (N / 2.0)
            h = np.clip(h, 0.0, 1.0)
        
        # Gradient descent with adaptive beta
        lr = 0.01
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Projected gradient descent
            grad = _grad_max_overlap_smooth(h, beta=50.0)
            h_new = h - lr * grad
            # Project to [0,1] and normalize
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new = h_new / np.sum(h_new) * (N / 2.0)
            h_new = np.clip(h_new, 0.0, 1.0)
            
            score_new = _max_overlap_smooth(h_new, beta=50.0)
            if score_new < best_score:
                best_h = h_new.copy()
                best_score = score_new
            
            h = h_new
            lr *= 0.999
        
        # Final polish with higher beta
        if time.time() > deadline - 10.0:
            break
        h = best_h.copy()
        for it in range(200):
            if time.time() > deadline - 5.0:
                break
            grad = _grad_max_overlap_smooth(h, beta=200.0)
            h_new = h - 0.005 * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new = h_new / np.sum(h_new) * (N / 2.0)
            h_new = np.clip(h_new, 0.0, 1.0)
            score_new = _max_overlap_smooth(h_new, beta=200.0)
            if score_new < best_score:
                best_h = h_new.copy()
                best_score = score_new
            h = h_new
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = best_h / np.sum(best_h) * (N / 2.0)
    best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
