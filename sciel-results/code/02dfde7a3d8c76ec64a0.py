"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_progressions(n=256):
    """Precompute all arithmetic progressions in [1, n] as index arrays."""
    progressions = []
    for step in range(1, n + 1):
        for start in range(1, n + 1, step):
            # AP: start, start+step, start+2*step, ... <= n
            indices = list(range(start - 1, n, step))  # 0-indexed
            if len(indices) >= 1:
                progressions.append(np.array(indices, dtype=np.int32))
    return progressions

def _max_discrepancy(signs, progressions):
    """Compute max |sum| over all arithmetic progressions."""
    max_disc = 0
    for idx in progressions:
        s = np.sum(signs[idx])
        abs_s = abs(s)
        if abs_s > max_disc:
            max_disc = abs_s
    return max_disc

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 50.0  # safety margin before 60s kill
    
    # Precompute progressions
    progressions = _precompute_progressions(n)
    
    # Initialize with random signs
    rng = random.Random(42)
    signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
    
    best_signs = signs.copy()
    best_disc = _max_discrepancy(signs, progressions)
    
    # Simulated annealing
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random position to flip
        pos = rng.randrange(n)
        signs[pos] = -signs[pos]
        
        new_disc = _max_discrepancy(signs, progressions)
        
        # Accept if better or with probability exp(-(new-old)/T)
        if new_disc < best_disc:
            best_disc = new_disc
            best_signs = signs.copy()
        elif new_disc <= best_disc or rng.random() < np.exp(-(new_disc - best_disc) / T):
            pass  # accept
        else:
            signs[pos] = -signs[pos]  # revert
        
        T *= cooling_rate
    
    return [int(x) for x in best_signs]
# EVOLVE-BLOCK-END
