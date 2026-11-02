import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initialize with 0.5
    h = np.full(N, 0.5)
    
    # We want to minimize max_k O_k
    # O_k = sum_{i=0}^{N-1-k} h_i * (1 - h_{i+k})
    
    # Helper to compute all overlaps
    def compute_overlaps(h):
        overlaps = np.zeros(N)
        for k in range(N):
            # h[0:N-k] and h[k:N]
            # Product h[i] * (1 - h[i+k])
            # Sum
            overlaps[k] = np.sum(h[:N-k] * (1 - h[k:]))
        return overlaps

    # Helper to compute gradient of overlap at shift k
    def grad_overlap_k(h, k):
        grad = np.zeros(N)
        if k == 0:
            # O_0 = sum h_i (1-h_i)
            # d/dh_j = 1 - 2 h_j
            grad = 1 - 2 * h
        else:
            # O_k = sum_{i=0}^{N-1-k} h_i - h_i h_{i+k}
            # Term i contributes to h_i and h_{i+k}
            # d/dh_i = 1 - h_{i+k} for i in 0..N-1-k
            # d/dh_{i+k} = -h_i for i in 0..N-1-k
            # So for j in 0..N-1-k:
            # grad[j] += 1 - h[j+k]
            # grad[j+k] += -h[j]
            
            # Vectorized:
            # Indices 0 to N-1-k
            idx = np.arange(N-k)
            grad[idx] += 1 - h[idx + k]
            grad[idx + k] += -h[idx]
            
        return grad

    # Projection onto simplex-like constraint: sum=50, 0<=h<=1
    def project(h):
        # Simple projection: clip and adjust mean
        h = np.clip(h, 0, 1)
        # Adjust to sum 50
        diff = 50 - np.sum(h)
        if diff > 0:
            # Add to smallest elements? Or distribute?
            # Distribute evenly to those < 1
            mask = h < 1
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        else:
            mask = h > 0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        
        # If still not sum 50 due to clipping, do a more careful adjustment
        # This simple method might fail if diff is large.
        # Better: sort and find threshold.
        # But for small steps, simple is ok.
        return h

    # Optimization loop
    # Use a soft-max objective to avoid getting stuck in local minima of single k
    # Objective: sum exp(beta * O_k)
    # Or just minimize the max by focusing on the current max.
    
    best_h = h.copy()
    best_score = np.inf
    
    # Initial score
    overlaps = compute_overlaps(h)
    current_max = np.max(overlaps)
    if current_max < best_score:
        best_score = current_max
        best_h = h.copy()
    
    eta = 0.05 # Step size
    
    for step in range(2000):
        if time.time() > deadline:
            break
            
        overlaps = compute_overlaps(h)
        current_max = np.max(overlaps)
        
        if current_max < best_score:
            best_score = current_max
            best_h = h.copy()
            
        # Find worst k
        k_max = np.argmax(overlaps)
        
        # Compute gradient of O_kmax
        grad = grad_overlap_k(h, k_max)
        
        # Update
        h_new = h - eta * grad
        h_new = project(h_new)
        
        # Check if improved
        overlaps_new = compute_overlaps(h_new)
        max_new = np.max(overlaps_new)
        
        if max_new < current_max:
            h = h_new
        else:
            # Reduce step size or try another k?
            # For now, just keep h
            eta = max(eta * 0.9, 0.001)
            
    return best_h
