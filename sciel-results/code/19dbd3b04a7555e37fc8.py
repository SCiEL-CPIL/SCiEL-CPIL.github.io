# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_max_overlap(h):
    """
    Compute max_k sum_{j=0}^{N-k-1} h[j] * (1 - h[j+k]) efficiently.
    """
    n = len(h)
    # O_k = sum h[j](1-h[j+k]) = sum h[j] - sum h[j]h[j+k]
    # sum h[j] is constant if we maintain the sum constraint strictly, but let's compute raw overlap.
    # Cross-correlation of h and (1-h)
    # np.correlate(h, 1-h, 'full')
    # The lag k part is at index N-1+k
    corr = np.correlate(h, 1.0 - h, mode='full')
    # We want lags 1 to N-1 (k=1..N-1)
    # Indices: N-1 + 1 to N-1 + N-1
    overlaps = corr[N : 2*N-1]
    return np.max(overlaps)

def _local_search(h, max_steps=50000):
    """
    Local search maintaining sum(h) = N/2.
    Moves: Swap a 0 and a 1 (or adjust values if non-binary).
    """
    n = len(h)
    target_sum = n / 2.0
    current_h = h.copy()
    
    # Ensure binary or near-binary for better search? 
    # The problem allows [0,1]. But binary often gives good baselines.
    # Let's allow small perturbations but primarily flip.
    
    current_score = _compute_max_overlap(current_h)
    
    np.random.seed(42)
    
    # Precompute indices of 0s and 1s for fast swapping?
    # Or just random pick.
    
    steps_without_improvement = 0
    max_no_improve = 2000
    
    # Get initial indices
    idx_zeros = np.where(current_h < 0.5)[0]
    idx_ones = np.where(current_h >= 0.5)[0]
    
    for step in range(max_steps):
        if _time_remaining() < 5.0:
            break
        if steps_without_improvement > max_no_improve:
            break
        
        # Pick a random zero and a random one to swap
        if len(idx_zeros) == 0 or len(idx_ones) == 0:
            break
            
        i = idx_zeros[np.random.randint(len(idx_zeros))]
        j = idx_ones[np.random.randint(len(idx_ones))]
        
        # Swap: h[i] becomes 1, h[j] becomes 0
        # This preserves the sum.
        old_i = current_h[i]
        old_j = current_h[j]
        
        current_h[i] = 1.0
        current_h[j] = 0.0
        
        new_score = _compute_max_overlap(current_h)
        
        if new_score < current_score - 1e-10:
            current_score = new_score
            steps_without_improvement = 0
            # Update indices
            # i was zero, now one. j was one, now zero.
            # This is O(N) to update arrays. For N=100, it's fine.
            idx_zeros = np.where(current_h < 0.5)[0]
            idx_ones = np.where(current_h >= 0.5)[0]
        else:
            # Revert
            current_h[i] = old_i
            current_h[j] = old_j
            steps_without_improvement += 1
            
    return current_h, current_score

def construct():
    n = N
    target_sum = n / 2.0
    
    # Generate a few diverse starting points
    best_h = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    # 1. Random binary
    h = np.random.choice([0.0, 1.0], size=n)
    # Adjust to sum n/2
    ones = np.sum(h == 1.0)
    target_ones = n // 2
    if ones > target_ones:
        idx = np.where(h == 1.0)[0]
        flip = np.random.choice(idx, size=ones - target_ones, replace=False)
        h[flip] = 0.0
    elif ones < target_ones:
        idx = np.where(h == 0.0)[0]
        flip = np.random.choice(idx, size=target_ones - ones, replace=False)
        h[flip] = 1.0
        
    res_h, res_score = _local_search(h)
    if res_score < best_score:
        best_score = res_score
        best_h = res_h.copy()
        
    # 2. Alternating pattern
    h_alt = np.tile([0.0, 1.0], n // 2 + 1)[:n]
    # Check sum
    if np.sum(h_alt) != target_sum:
        # Adjust
        diff = target_sum - np.sum(h_alt)
        # Find an element to adjust
        # If diff > 0, we need more mass.
        # For N=100, alternating sum is 50.
        pass
    res_h, res_score = _local_search(h_alt)
    if res_score < best_score:
        best_score = res_score
        best_h = res_h.copy()
        
    # 3. Random smooth (Gaussian noise)
    h_smooth = np.random.rand(n)
    # Project to sum
    h_smooth = (h_smooth - np.mean(h_smooth)) * 2 + 0.5 # Scale to [0,1] roughly
    h_smooth = np.clip(h_smooth, 0, 1)
    # Adjust sum exactly
    diff = target_sum - np.sum(h_smooth)
    h_smooth += diff / n
    h_smooth = np.clip(h_smooth, 0, 1)
    # Binary-ize for search? Or keep continuous?
    # Local search above assumes binary swaps. 
    # Let's just binarize it to use the fast binary search.
    h_bin = (h_smooth >= 0.5).astype(float)
    ones = np.sum(h_bin == 1.0)
    target_ones = n // 2
    if ones > target_ones:
        idx = np.where(h_bin == 1.0)[0]
        flip = np.random.choice(idx, size=ones - target_ones, replace=False)
        h_bin[flip] = 0.0
    elif ones < target_ones:
        idx = np.where(h_bin == 0.0)[0]
        flip = np.random.choice(idx, size=target_ones - ones, replace=False)
        h_bin[flip] = 1.0
        
    res_h, res_score = _local_search(h_bin)
    if res_score < best_score:
        best_score = res_score
        best_h = res_h.copy()

    # Run more iterations if time permits
    iterations = 0
    while _time_remaining() > 10.0:
        # Random restart with perturbation of best
        h_pert = best_h.copy()
        # Flip 2-5 random bits
        num_flips = np.random.randint(2, 6)
        if num_flips % 2 != 0:
            num_flips += 1 # Keep sum same? No, flipping pairs keeps sum same.
        
        # Pick pairs to flip
        # Get current zeros and ones
        idx_zeros = np.where(h_pert < 0.5)[0]
        idx_ones = np.where(h_pert >= 0.5)[0]
        
        for _ in range(num_flips // 2):
            if len(idx_zeros) == 0 or len(idx_ones) == 0: break
            i = idx_zeros[np.random.randint(len(idx_zeros))]
            j = idx_ones[np.random.randint(len(idx_ones))]
            h_pert[i] = 1.0
            h_pert[j] = 0.0
            # Update indices for next flip? 
            # For simplicity, just recompute or assume small changes.
            # Recomputing is O(N).
            idx_zeros = np.where(h_pert < 0.5)[0]
            idx_ones = np.where(h_pert >= 0.5)[0]
            
        res_h, res_score = _local_search(h_pert, max_steps=10000)
        if res_score < best_score:
            best_score = res_score
            best_h = res_h.copy()
            
        iterations += 1
        
    return best_h

# EVOLVE-BLOCK-END
