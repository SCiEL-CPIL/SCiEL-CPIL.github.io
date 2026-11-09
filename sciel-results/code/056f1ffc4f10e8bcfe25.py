# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the
    minimum pairwise squared distance is >= the maximum squared norm.
    
    Construction: 2*E8 (240 points) + 2*D3 (12 points) = 252 points.
    All points have norm squared = 8, min pairwise distance squared = 8.
    Then attempts greedy expansion with additional integer points.
    """
    deadline = time.time() + 540  # safety margin
    
    d = 11
    
    # --- Generate 2*E8 vectors (240 points) in first 8 dimensions ---
    e8_vectors = []
    
    # Weight-2 vectors: ±e_i ± e_j for 1 <= i < j <= 8
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 8
                v[i] = 2 * si
                v[j] = 2 * sj
                e8_vectors.append(v)
    
    # Weight-8 vectors: ±1 in all 8 coordinates, with even number of minus signs
    for signs in product([1, -1], repeat=8):
        if sum(1 for s in signs if s == -1) % 2 == 0:
            e8_vectors.append(list(signs))
    
    # Pad to 11D
    e8_scaled = []
    for v in e8_vectors:
        vec = v + [0] * 3
        e8_scaled.append(vec)
    
    # --- Generate 2*D3 vectors (12 points) in last 3 dimensions ---
    d3_vectors = []
    for i, j in combinations(range(3), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 3
                v[i] = 2 * si
                v[j] = 2 * sj
                d3_vectors.append(v)
    
    # Pad to 11D (first 8 dims are 0)
    d3_scaled = []
    for v in d3_vectors:
        vec = [0] * 8 + v
        d3_scaled.append(vec)
    
    # Combine all points
    all_points = e8_scaled + d3_scaled
    
    points = np.array(all_points, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
