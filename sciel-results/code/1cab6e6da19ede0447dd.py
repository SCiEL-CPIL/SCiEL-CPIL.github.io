"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute signed sum over all arithmetic progressions
    in [0, n-1].
    
    signs: numpy array of -1/1 of length n.
    """
    n = len(signs)
    max_disc = 1  # Singletons have sum +/-1
    
    # We iterate over all possible steps d
    # For a fixed step d, the progressions are determined by start index r in [0, d-1]
    # The sequence is signs[r], signs[r+d], signs[r+2d], ...
    # We need the max absolute prefix sum for each such sequence.
    
    # To be efficient, we can process all r for a given d.
    # However, looping d from 1 to n is O(n).
    # Inside, we have d sequences.
    # Total work is sum_{d=1}^n d * (n/d) = n^2 operations roughly?
    # Actually, for a fixed d, the total number of elements processed across all r is n.
    # So total complexity is O(n^2) for all d.
    # n=256, n^2 = 65536. This is very fast.
    
    for d in range(1, n + 1):
        # For each start r in [0, d-1] such that r < n
        # We look at the subsequence signs[r::d]
        # We need max abs prefix sum of this subsequence.
        
        # Optimization: If d is large, the subsequences are short.
        # If d is small, there are few subsequences but they are long.
        
        # We can vectorize the prefix sum calculation for a fixed d?
        # Not easily in pure numpy without reshaping tricks because lengths vary.
        # But n is small, so a python loop over r is fine.
        
        # Actually, we can just compute the cumulative sum for each r.
        for r in range(d):
            if r >= n:
                break
            subseq = signs[r::d]
            if len(subseq) == 0:
                continue
            # Compute cumulative sum
            csums = np.cumsum(subseq)
            # Find max absolute value
            current_max = np.max(np.abs(csums))
            if current_max > max_disc:
                max_disc = current_max
                
    return int(max_disc)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget, 10s safety margin
    
    n = 256
    rng = random.Random(42)
    
    # Initialize with random signs
    signs = np.array([rng.choice((-1, 1)) for _ in range(n)], dtype=np.int64)
    
    current_score = _max_abs_ap_sum(signs)
    best_signs = signs.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    
    # Pre-compute indices for faster flipping? No, just flip index.
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.randint(0, n - 1)
        
        # Flip
        signs[idx] *= -1
        
        # Evaluate
        new_score = _max_abs_ap_sum(signs)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = signs.copy()
        else:
            # Metropolis criterion: accept with probability exp(-(new_score - current_score) / T)
            # Since scores are integers, diff is at least 1.
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    # Reject: revert flip
                    signs[idx] *= -1
            # If delta <= 0, it would have been accepted in the first branch
            # But if delta == 0, we accept to explore? 
            # Usually SA accepts equal moves. Let's accept equal moves to move around plateau.
            elif delta == 0:
                current_score = new_score
            else:
                signs[idx] *= -1
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optional: Restart with slightly higher T if stuck? 
            # For now, keep cooling.
            
    # Return best found as list of ints
    return best_signs.tolist()

# EVOLVE-BLOCK-END
