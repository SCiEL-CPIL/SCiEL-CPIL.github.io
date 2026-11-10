"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_peak(A_set):
    """Compute the peak representation multiplicity for sums in [128, 256].
    Returns (peak, coverage_ok) where coverage_ok is True if all targets are covered."""
    # Fast vectorized computation
    mask = np.zeros(129, dtype=np.int32)
    for x in A_set:
        mask[x] = 1
        
    # Convolution for ordered pairs
    conv = np.convolve(mask, mask, mode='full')
    
    # Sums 128 to 256
    sums = np.arange(128, 257)
    ordered_counts = conv[128:257]
    
    # Diagonal counts D[S] = 1 if S is even and S/2 in A
    is_even = (sums % 2 == 0)
    half_sums = sums // 2
    D = np.zeros(129, dtype=np.int32)
    D[is_even] = mask[half_sums[is_even]]
    
    # Unordered counts
    unordered_counts = (ordered_counts + D) // 2
    
    peak = int(unordered_counts.max())
    coverage_ok = bool((unordered_counts > 0).all())
    
    return peak, coverage_ok

def _local_search(A_set, max_iters=50000, deadline=None):
    """Simulated Annealing local search."""
    A_set = set(A_set)
    peak, cov = _compute_peak(A_set)
    
    # If not covered, we still run SA with penalty
    best_set = set(A_set)
    best_peak = peak
    best_cov = cov
    
    rng = random.Random(42)
    
    # Convert to mask for faster updates in SA? 
    # We will use _compute_peak which takes a set. 
    # To make it faster, we can keep a mask and update it, 
    # but _compute_peak takes a set. 
    # Let's modify _compute_peak to accept a mask? 
    # No, signature must be kept. 
    # So we pass sets. 
    # To speed up, we can create a helper that takes a mask.
    
    def _peak_from_mask(mask):
        conv = np.convolve(mask, mask, mode='full')
        sums = np.arange(128, 257)
        ordered_counts = conv[128:257]
        is_even = (sums % 2 == 0)
        half_sums = sums // 2
        D = np.zeros(129, dtype=np.int32)
        D[is_even] = mask[half_sums[is_even]]
        unordered_counts = (ordered_counts + D) // 2
        peak = int(unordered_counts.max())
        cov = bool((unordered_counts > 0).all())
        return peak, cov

    mask = np.zeros(129, dtype=np.int32)
    for x in A_set:
        mask[x] = 1
        
    current_peak, current_cov = _peak_from_mask(mask)
    current_cost = current_peak + (0 if current_cov else 100)
    
    best_cost = current_cost
    best_mask = mask.copy()
    
    T = 5.0
    cooling = 0.999
    
    while (deadline is None or time.time() < deadline):
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove
            if mask.sum() == 0: continue
            idx = rng.randint(0, 128)
            if mask[idx] == 0:
                true_indices = np.where(mask > 0)[0]
                if len(true_indices) == 0: continue
                idx = rng.choice(true_indices)
            new_mask = mask.copy()
            new_mask[idx] = 0
        elif move_type < 0.8:
            # Add
            idx = rng.randint(0, 128)
            if mask[idx] == 1:
                false_indices = np.where(mask == 0)[0]
                if len(false_indices) == 0: continue
                idx = rng.choice(false_indices)
            new_mask = mask.copy()
            new_mask[idx] = 1
        else:
            # Swap
            true_indices = np.where(mask > 0)[0]
            false_indices = np.where(mask == 0)[0]
            if len(true_indices) == 0 or len(false_indices) == 0:
                continue
            idx_out = rng.choice(true_indices)
            idx_in = rng.choice(false_indices)
            new_mask = mask.copy()
            new_mask[idx_out] = 0
            new_mask[idx_in] = 1
            
        new_peak, new_cov = _peak_from_mask(new_mask)
        new_cost = new_peak + (0 if new_cov else 100)
        
        delta = new_cost - current_cost
        
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            mask = new_mask
            current_peak = new_peak
            current_cov = new_cov
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_mask = mask.copy()
        
        T *= cooling

    # Reconstruct set from best_mask
    best_set = set(np.where(best_mask > 0)[0])
    best_peak, best_cov = _compute_peak(best_set)
    
    return best_set, best_peak, best_cov

def _greedy_construction():
    """Greedy construction: start with a base set and add elements to cover gaps."""
    A_set = set()
    for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
              64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
              118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]:
        A_set.add(x)
    return A_set

def _ensure_coverage(A_set, deadline=None):
    """Ensure all targets in [128, 256] are covered by adding elements if needed."""
    A_set = set(A_set)
    _, cov = _compute_peak(A_set)
    if cov:
        return A_set, True
    return A_set, False

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline with 5s safety margin
    
    # Phase 1: Initial construction
    A_set = _greedy_construction()
    
    # Phase 2: Local search to minimize peak
    A_set, peak, cov = _local_search(A_set, max_iters=1000000, deadline=deadline)
    
    if not cov:
        return list(range(64, 129))
    
    return sorted(A_set)
# EVOLVE-BLOCK-END
