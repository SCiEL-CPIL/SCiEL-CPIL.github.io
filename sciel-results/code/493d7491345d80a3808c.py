"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def _dist2(a, b):
    return np.sum((a - b) ** 2)


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s budget with margin
    
    # Start with weight-2 vectors (known valid: min_dist²=4, max_norm²=2)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    best = vectors.copy()
    
    # Try to improve by searching for larger configurations
    # Strategy: try weight-3 vectors with constraint that min_dist² >= max_norm²
    # For pure weight-3: max_norm² = 3, need min_dist² >= 3
    # Two weight-3 vectors sharing 2+ positions with same signs have dist² = 2 < 3
    # So we need a set where no two share 2+ positions with same signs
    
    # Generate all weight-3 candidates
    candidates = []
    for idx in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(idx, signs):
                v[pos] = s
            candidates.append(v)
    candidates = np.array(candidates, dtype=np.int64)
    
    # Greedy: add candidates that maintain min_dist² >= 3 among all selected
    selected = []
    for c in candidates:
        if time.time() > deadline - 10:
            break
        if not selected:
            selected.append(c)
            continue
        # Check distance to all selected
        dists = np.sum((selected - c) ** 2, axis=1)
        if np.min(dists) >= 3:
            selected.append(c)
    
    if len(selected) > len(best):
        best = np.array(selected, dtype=np.int64)
    
    # Also try: keep weight-2 set and see if we can do better with mixed approach
    # For now, return the best found
    return best
# EVOLVE-BLOCK-END
