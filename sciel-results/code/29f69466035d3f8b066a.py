import numpy as np
import time

N = 100

def _compute_overlaps(h):
    # h is 1D array of length N
    # Overlaps for circular shifts
    # f_k = sum h_i (1 - h_{i-k})
    # Using FFT for speed? N=100 is small, direct is fine.
    overlaps = np.zeros(N)
    for k in range(N):
        # h_shifted is h shifted by k (circular)
        # h_{i-k} corresponds to np.roll(h, k)
        h_shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1 - h_shifted))
    return overlaps

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guess: Ramp
    h = np.linspace(0, 1, N)
    # Adjust sum to N/2
    h = (h - np.mean(h)) * 0 + 0.5 # Reset to 0.5 for safety, or keep ramp?
    # Let's try 0.5 first, then perturb.
    h = np.full(N, 0.5)
    
    best_h = h.copy()
    best_score = np.max(_compute_overlaps(h))
    
    # Gradient Descent on Soft-Max
    # Objective: L = log(sum(exp(beta * f_k)))
    beta = 50.0
    
    for step in range(1000):
        if time.time() > deadline:
            break
            
        f = _compute_overlaps(h)
        # Soft max gradient
        # dL/dh_i = sum_k (beta * exp(beta f_k) / sum exp(beta f)) * df_k/dh_i
        # df_k/dh_i = (1 - h_{i-k}) - h_i * d(1-h_{i-k})/dh_i
        # d(1-h_{i-k})/dh_i = - delta_{i, i-k} (circular)
        # So df_k/dh_i = (1 - h_{i-k}) + h_i * delta_{i, i-k}
        
        weights = np.exp(beta * f) / np.sum(np.exp(beta * f))
        grad = np.zeros(N)
        
        for k in range(N):
            h_shifted = np.roll(h, k)
            # Term 1: (1 - h_shifted)
            grad += weights[k] * (1 - h_shifted)
            # Term 2: h_i * delta
            # delta is 1 at i = i-k => k=0? No.
            # h_{i-k} depends on h_i only if i-k == i => k=0.
            # Wait, circular. h_{i-k} is the element at index (i-k) % N.
            # If we vary h_j, it affects h_{i-k} if j == (i-k) % N.
            # So for a fixed k, h_j affects h_{i-k} if j = i-k.
            # So d h_{i-k} / d h_j = 1 if j = i-k, else 0.
            # Sum over i of h_i * delta_{j, i-k} = h_{j+k}.
            # So grad += weights[k] * h_{j+k} ?
            # Let's re-derive.
            # f_k = sum_i h_i (1 - h_{i-k})
            # df_k/dh_j = sum_i ( delta_{i,j} (1 - h_{i-k}) - h_i delta_{i-k, j} )
            # First term: i=j => 1 - h_{j-k}
            # Second term: i-k=j => i=j+k => - h_{j+k}
            # So df_k/dh_j = (1 - h_{j-k}) - h_{j+k}
            
            # My previous derivation was slightly off.
            # Let's use this.
            pass # Doing inside loop is slow, but N=100 is small.

        # Recompute grad correctly
        grad = np.zeros(N)
        for k in range(N):
            h_j_minus_k = np.roll(h, -k) # h_{j-k}
            h_j_plus_k = np.roll(h, k)   # h_{j+k}
            df_dh = (1 - h_j_minus_k) - h_j_plus_k
            grad += weights[k] * df_dh
        
        # Projected Gradient Step
        # h_new = h - eta * grad
        # Then project to [0,1] and sum N/2
        
        eta = 0.05
        h_new = h - eta * grad
        
        # Projection
        # 1. Clip
        h_new = np.clip(h_new, 0, 1)
        # 2. Adjust sum
        # Binary search for lambda
        # sum(clip(h_new + lambda, 0, 1)) = N/2
        
        target_sum = N / 2.0
        current_sum = np.sum(h_new)
        
        # Simple iterative adjustment
        for _ in range(10):
            diff = target_sum - current_sum
            if abs(diff) < 1e-6:
                break
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            current_sum = np.sum(h_new)
            
        # If sum is still off, force it (might violate bounds slightly, but let's check)
        if abs(np.sum(h_new) - target_sum) > 1e-4:
             # Fallback: distribute remaining
             rem = target_sum - np.sum(h_new)
             h_new += rem / N
             h_new = np.clip(h_new, 0, 1)
             
        h = h_new
        
        current_score = np.max(_compute_overlaps(h))
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h

# EVOLVE-BLOCK-END
