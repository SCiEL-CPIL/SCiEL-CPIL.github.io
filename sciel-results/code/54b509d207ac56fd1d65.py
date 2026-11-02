"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(h):
    """
    Compute the maximum over shifts s in [0, 2] of the overlap integral.
    For a step function with N steps on [0, 2], the overlap for shift k*dw is:
    sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    We take the maximum over all k = 0, 1, ..., N-1.
    """
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # The optimal solution is h[i] = 0.5 for all i.
    # This gives overlap = sum_{i=0}^{N-k-1} 0.5 * 0.5 * dw = (N-k)*0.25*dw for all k.
    # The maximum is at k=0: N*0.25*dw = N*0.25*(2/N) = 0.5.
    # Any deviation from 0.5 will increase the maximum overlap.
    
    # Let's verify: if h = [0.5]*N, sum = N*0.5 = N/2. Correct.
    # Overlap for shift k: sum_{i=0}^{N-k-1} h[i]*(1-h[i+k])*dw = (N-k)*0.5*0.5*dw = (N-k)*0.25*dw.
    # Max is at k=0: N*0.25*dw = N*0.25*(2/N) = 0.5.
    
    # Can we do better than 0.5?
    # The overlap is sum h[i]*(1-h[i+k])*dw.
    # If we set h to be binary (0 or 1), the overlap is the number of positions where h[i]=1 and h[i+k]=0, times dw.
    # For a binary sequence with N/2 ones, the minimum maximum overlap is achieved by a balanced sequence.
    # But continuous values allow us to reach 0.5 uniformly.
    
    # Actually, let's think about this more carefully.
    # We want to minimize max_k sum_{i=0}^{N-k-1} h[i]*(1-h[i+k])*dw.
    # 
    # If h[i] = 0.5 for all i, then for each k, the overlap is (N-k)*0.25*dw.
    # The maximum is at k=0: N*0.25*dw = 0.5.
    #
    # Can we get below 0.5? 
    # Consider the sum over all k of the overlaps:
    # sum_{k=0}^{N-1} sum_{i=0}^{N-k-1} h[i]*(1-h[i+k])*dw
    # = dw * sum_{i,j} h[i]*(1-h[j]) where i < j (since k = j-i, and i goes from 0 to N-k-1 means j = i+k < N)
    # Wait, that's not quite right. Let me re-index.
    # For a fixed pair (i, j) with i < j, let k = j - i. Then this pair contributes h[i]*(1-h[j])*dw to the overlap for shift k.
    # So the sum over all k of overlap_k = dw * sum_{0<=i<j<N} h[i]*(1-h[j])
    # = dw * sum_{i<j} h[i] - dw * sum_{i<j} h[i]*h[j]
    # 
    # This is getting complex. Let's just try the constant 0.5 solution and see if we can improve with a local search.
    
    # Start with the known optimal candidate: h = 0.5 everywhere
    h = np.full(N, 0.5)
    current_score = _max_overlap(h)
    
    # Try a few different approaches to see if we can beat 0.5
    best_h = h.copy()
    best_score = current_score
    
    # Approach 1: Try to use a gradient-free local search with small perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            h_curr = np.full(N, 0.5)
        else:
            # Random initialization with sum = N/2
            h_curr = rng.random(N)
            h_curr = h_curr / h_curr.sum() * (N / 2.0)
        
        current_score = _max_overlap(h_curr)
        if current_score < best_score:
            best_score = current_score
            best_h = h_curr.copy()
        
        # Local search: try small random perturbations that preserve the sum
        for it in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            # Pick two random indices and transfer a small amount
            i, j = rng.choice(N, size=2, replace=False)
            delta = rng.uniform(-0.1, 0.1)
            
            # Ensure bounds are respected
            new_hi = h_curr[i] + delta
            new_hj = h_curr[j] - delta
            if new_hi < 0 or new_hi > 1 or new_hj < 0 or new_hj > 1:
                continue
            
            h_curr[i] = new_hi
            h_curr[j] = new_hj
            
            new_score = _max_overlap(h_curr)
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_curr.copy()
            else:
                # Revert
                h_curr[i] -= delta
                h_curr[j] += delta
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(best_h)
    if abs(current_sum - N / 2.0) > 1e-12:
        diff = N / 2.0 - current_sum
        # Distribute the difference evenly
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # Re-adjust if clipping changed the sum
        current_sum = np.sum(best_h)
        if abs(current_sum - N / 2.0) > 1e-12:
            diff = N / 2.0 - current_sum
            idx = np.argmin(np.abs(best_h - 0.5))
            best_h[idx] += diff
            best_h[idx] = np.clip(best_h[idx], 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
