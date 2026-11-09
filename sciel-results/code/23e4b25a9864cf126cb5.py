"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_progressions(n=256):
    """Precompute all arithmetic progressions in [1, n] as index arrays (0-indexed)."""
    progressions = []
    for step in range(1, n + 1):
        for start in range(0, n, step):
            indices = np.arange(start, n, step, dtype=np.int32)
            if len(indices) >= 1:
                progressions.append(indices)
    return progressions

def _max_discrepancy(signs, progressions):
    """Compute max |sum| over all arithmetic progressions."""
    max_disc = 0
    for idx in progressions:
        s = signs[idx].sum()
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def construct():
    n = 256
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Precompute all APs
    progressions = _precompute_progressions(n)
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    # Evaluate initial
    current_disc = _max_discrepancy(signs, progressions)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random position to flip
        pos = rng.randint(0, n - 1)
        old_sign = signs[pos]
        new_sign = -old_sign
        
        # Compute change in discrepancy efficiently
        # For each AP containing pos, the sum changes by 2*new_sign
        # We need to find which APs contain pos and update their sums
        # For efficiency, we'll just recompute the max discrepancy
        # But that's O(num_APs * avg_len) which is expensive
        # Instead, let's track sums for all APs
        
        # Actually, let's use a different approach:
        # Maintain an array of sums for each AP
        # When we flip position pos, for each AP that contains pos, 
        # the sum changes by 2*new_sign
        
        # Precompute for each position, which APs contain it
        # This is memory intensive but let's try
        
        # Alternative: just recompute from scratch but vectorized
        signs[pos] = new_sign
        
        new_disc = _max_discrepancy(signs, progressions)
        
        delta = new_disc - current_disc
        
        # Metropolis criterion
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            signs[pos] = old_sign
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Reset temperature occasionally to escape local minima
        if step % 5000 == 0:
            T = max(T, 5.0)
    
    return list(best_signs)
# EVOLVE-BLOCK-END
