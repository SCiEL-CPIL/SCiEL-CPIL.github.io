"""Low discrepancy on all arithmetic progressions - Vectorized evaluation with Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns the max absolute sum.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    
    max_abs = 0
    # We iterate over step sizes d from 1 to n//2 (step > n//2 gives only length 1 or 2 progressions)
    # Actually, for step d, the maximum length is floor((n-1)/d) + 1
    # We need to check all starting positions i in [0, d) and all lengths L >= 1
    
    for d in range(1, n):
        # For step d, starting indices are 0, 1, ..., d-1
        # But starting index i is valid only if i < n
        # The progression is s[i], s[i+d], s[i+2d], ...
        # We can vectorize over starting indices
        
        # For a given d, we have d possible starting residues
        # For each residue r in 0..d-1, the sequence is s[r], s[r+d], s[r+2d], ...
        # We want max over all prefixes of these sequences of |sum|
        
        # Vectorized approach:
        # For each r, compute cumulative sums of s[r::d]
        # Then take max abs of cumulative sums
        
        # To vectorize, we can pad sequences to same length
        # Max length for step d is L_max = (n - 1) // d + 1
        # But different r have different lengths
        
        # Simpler: iterate over r in 0..d-1, but vectorize the cumulative sum computation
        # For each r, the indices are r, r+d, r+2d, ... < n
        # Number of elements: L_r = (n - 1 - r) // d + 1 if r < n else 0
        
        # For efficiency, let's process all r at once by creating a 2D array
        # But lengths vary, so let's just loop over r but use numpy for the cumsum
        
        for r in range(d):
            if r >= n:
                break
            # Get the subsequence
            sub = s[r::d]
            if len(sub) == 0:
                continue
            # Compute cumulative sums
            cumsums = np.cumsum(sub)
            # Find max absolute value
            m = np.max(np.abs(cumsums))
            if m > max_abs:
                max_abs = int(m)
                # Early termination: if max_abs is already high, we might want to stop
                # But for SA, we need exact value for comparison
                
    return max_abs

def _fast_max_ap_discrepancy(signs):
    """
    Optimized version using precomputed structures.
    For SA, we often only need to know if a move improves the current max.
    This version is slightly optimized for the common case.
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int32)
    
    max_abs = 0
    for d in range(1, n):
        for r in range(d):
            if r >= n:
                break
            sub = s[r::d]
            if len(sub) == 0:
                continue
            cumsums = np.cumsum(sub)
            m = np.max(np.abs(cumsums))
            if m > max_abs:
                max_abs = int(m)
                if max_abs > 10:  # Early exit if already bad
                    return max_abs
    return max_abs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 256
    rng = random.Random(42)
    
    # Initialize with random signs
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int32)
    current_cost = _max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.randrange(n)
        signs[idx] *= -1
        
        new_cost = _max_ap_discrepancy(signs)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_cost - current_cost) / T)
            delta = new_cost - current_cost
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() > prob:
                    # Reject
                    signs[idx] *= -1
            else:
                # Should not happen since new_cost >= current_cost in else branch
                pass
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_signs.tolist()

# EVOLVE-BLOCK-END
