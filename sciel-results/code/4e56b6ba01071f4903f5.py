import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Initial guess: uniform
    h0 = np.full(N, 0.5)
    
    def overlap_max(h):
        # h is a vector of N values, each in [0,1], sum = N/2
        # Step width = 2/N
        # Compute overlap for all shifts
        # Shift s corresponds to shifting h by k steps, where s = k * 2/N
        # For shift k, the overlap region is from max(0, -k) to min(N, N-k) in index space
        
        max_ov = 0
        for k in range(-N, N+1):
            # Shift h by k positions
            # h_shifted[i] = h[i-k] if 0 <= i-k < N, else 0
            # Overlap region: i such that 0 <= i < N and 0 <= i-k < N
            # i.e., max(0, k) <= i < min(N, N+k)
            
            start = max(0, k)
            end = min(N, N+k)
            if start >= end:
                continue
            
            # h in overlap region: h[start:end]
            # h_shifted in overlap region: h[start-k:end-k]
            h1 = h[start:end]
            h2 = h[start-k:end-k]
            
            # Overlap integral = sum of h1 * (1-h2) * step_width
            ov = np.sum(h1 * (1 - h2)) * (2.0 / N)
            max_ov = max(max_ov, ov)
        
        return max_ov
    
    # Constraint: sum(h) = N/2, 0 <= h <= 1
    # Use penalty method or constrained optimization
    
    # Let's use a simple iterative approach
    # Start with uniform, then try to reduce the max overlap
    
    h = h0.copy()
    
    for iteration in range(1000):
        # Compute gradient of max overlap with respect to h
        # This is complex, so let's use a simple hill-climbing approach
        
        # Try perturbing a random element
        idx = np.random.randint(N)
        delta = np.random.uniform(-0.1, 0.1)
        h_new = h.copy()
        h_new[idx] = np.clip(h_new[idx] + delta, 0, 1)
        
        # Adjust to maintain sum = N/2
        diff = N/2 - np.sum(h_new)
        if abs(diff) < 1e-10:
            pass
        else:
            # Distribute the difference among other elements
            mask = np.ones(N, dtype=bool)
            mask[idx] = False
            # Only adjust elements that can absorb the change
            can_increase = h_new < 1
            can_decrease = h_new > 0
            can_increase[idx] = False
            can_decrease[idx] = False
            
            if diff > 0:
                # Need to increase sum
                candidates = np.where(can_increase & can_decrease)[0]
                if len(candidates) == 0:
                    candidates = np.where(can_increase)[0]
                if len(candidates) > 0:
                    per = diff / len(candidates)
                    h_new[candidates] = np.clip(h_new[candidates] + per, 0, 1)
            else:
                # Need to decrease sum
                candidates = np.where(can_increase & can_decrease)[0]
                if len(candidates) == 0:
                    candidates = np.where(can_decrease)[0]
                if len(candidates) > 0:
                    per = diff / len(candidates)
                    h_new[candidates] = np.clip(h_new[candidates] + per, 0, 1)
        
        # Check if sum is still N/2
        if abs(np.sum(h_new) - N/2) > 1e-8:
            continue
        
        ov_new = overlap_max(h_new)
        ov_old = overlap_max(h)
        
        if ov_new < ov_old:
            h = h_new
    
    return h
