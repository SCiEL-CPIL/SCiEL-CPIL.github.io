"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

DEADLINE = 115.0
START_TIME = time.time()


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def construct():
    # Strategy: Compare greedy constructions for weight-3 and weight-4 vectors.
    # Weight-3: Norm^2=3, need u.v <= 1.
    # Weight-4: Norm^2=4, need u.v <= 2.
    
    def _greedy_max_set(weight, max_dot):
        candidates = []
        for idxs in combinations(range(D), weight):
            for signs in product((1, -1), repeat=weight):
                v = np.zeros(D, dtype=np.int64)
                for idx, sign in zip(idxs, signs):
                    v[idx] = sign
                candidates.append(v)
        
        candidates = np.array(candidates, dtype=np.int64)
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        indices = np.arange(len(candidates))
        np.random.shuffle(indices)
        candidates = candidates[indices]
        
        selected = []
        
        for c in candidates:
            if _time_left() < 10.0:
                break
                
            if len(selected) == 0:
                selected.append(c)
                continue
                
            arr = np.array(selected)
            dots = arr @ c
            if np.max(dots) <= max_dot:
                selected.append(c)
                
        return np.array(selected, dtype=np.int64) if selected else np.empty((0, D), dtype=np.int64)

    # Run W3
    res_w3 = _greedy_max_set(3, 1)
    
    # Run W4
    res_w4 = _greedy_max_set(4, 2)
    
    # Return the best one
    if len(res_w3) >= len(res_w4):
        return res_w3
    else:
        return res_w4
# EVOLVE-BLOCK-END
