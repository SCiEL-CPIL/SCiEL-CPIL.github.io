"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _eval_peak(A_set):
    """Compute the maximum unordered-pair representation multiplicity over [128,256]."""
    n = 128
    ind = np.zeros(2*n + 1, dtype=np.int32)
    for a in A_set:
        ind[a] = 1
    conv = np.convolve(ind, ind, mode='full')
    # conv[t] = ordered pair count for sum t
    targets = np.arange(128, 257)
    conv_t = conv[targets]  # ordered counts
    # self-pair indicator: t even and t//2 in A
    half = targets // 2
    is_even = (targets % 2 == 0)
    # Check if half is in A_set
    # half ranges from 64 to 128
    in_A = np.zeros(len(targets), dtype=np.int32)
    for i, h in enumerate(half):
        if is_even[i] and h in A_set:
            in_A[i] = 1
    unordered = (conv_t + in_A) // 2
    # Check coverage: all must be >= 1
    if np.any(unordered < 1):
        return 999999
    return int(np.max(unordered))

def _eval_peak_fast(A_set, ind):
    """Faster eval assuming ind is the indicator array."""
    conv = np.convolve(ind, ind, mode='full')
    targets = np.arange(128, 257)
    conv_t = conv[targets]
    half = targets // 2
    is_even = (targets % 2 == 0)
    in_A = np.zeros(len(targets), dtype=np.int32)
    for i, h in enumerate(half):
        if is_even[i] and h in A_set:
            in_A[i] = 1
    unordered = (conv_t + in_A) // 2
    if np.any(unordered < 1):
        return 999999
    return int(np.max(unordered))

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # safety margin before 60s limit
    
    n = 128
    # Start with a reasonable initial set
    # The seed is [64..128] which gives peak ~32
    # Try a denser set or a structured set
    A = set(range(64, 129))
    ind = np.zeros(2*n + 1, dtype=np.int32)
    for a in A:
        ind[a] = 1
    
    best_peak = _eval_peak(A)
    best_A = set(A)
    
    # Local search: try adding/removing elements
    # Elements can be in [0, 128]
    all_elements = set(range(0, 129))
    
    # First, try to improve by adding elements that help cover and reduce peak
    # Then try removing elements that don't hurt
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        
        # Try single additions
        for x in sorted(all_elements - A):
            if (time.time() - start_time) > DEADLINE:
                break
            A.add(x)
            ind[x] = 1
            peak = _eval_peak_fast(A, ind)
            ind[x] = 0
            A.discard(x)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
                A.add(x)
                ind[x] = 1
                improved = True
        
        # Try single removals
        for x in sorted(A):
            if (time.time() - start_time) > DEADLINE:
                break
            A.discard(x)
            ind[x] = 0
            peak = _eval_peak_fast(A, ind)
            ind[x] = 1
            A.add(x)
            if peak < best_peak:
                best_peak = peak
                best_A = set(A)
                A.discard(x)
                ind[x] = 0
                improved = True
    
    # Try more aggressive multi-move perturbations if time remains
    if (time.time() - start_time) < DEADLINE - 5:
        for iteration in range(20):
            if (time.time() - start_time) > DEADLINE - 3:
                break
            
            # Random perturbation: swap 1-3 elements
            A_copy = set(best_A)
            n_swaps = 1 + (iteration % 3)
            
            # Remove some random elements
            to_remove = list(A_copy)
            np.random.shuffle(to_remove)
            for x in to_remove[:n_swaps]:
                A_copy.discard(x)
            
            # Add some random elements
            candidates = list(all_elements - A_copy)
            np.random.shuffle(candidates)
            for x in candidates[:n_swaps]:
                A_copy.add(x)
            
            if not A_copy:
                continue
                
            ind2 = np.zeros(2*n + 1, dtype=np.int32)
            for a in A_copy:
                ind2[a] = 1
            peak = _eval_peak_fast(A_copy, ind2)
            
            if peak < best_peak:
                best_peak = peak
                best_A = A_copy
    
    return list(best_A)
# EVOLVE-BLOCK-END
