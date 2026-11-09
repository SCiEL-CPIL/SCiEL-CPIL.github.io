import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a large set of integer points in Z^11 with squared norm 4
    and mutual squared distance >= 4, using a greedy independent set search
    over a large candidate pool.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Generate candidate pool: all integer vectors in Z^11 with squared norm 4
    # Types:
    #   Type A: (±2, ±2, 0, ...) - C(11,2)*4 = 220
    #   Type B: (±2, ±1, ±1, 0, ...) - C(11,3)*8 = 1320
    #   Type C: (±1, ±1, ±1, ±1, 0, ...) - C(11,4)*16 = 3300
    # Total: 4840 candidates
    points_list = []
    
    # Type A: two entries of ±2
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # Type B: one entry of ±2, two entries of ±1
    for i, j, k in combinations(range(d), 3):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = sj
                    p[k] = sk
                    points_list.append(p)
    
    # Type C: four entries of ±1
    for i, j, k, l in combinations(range(d), 4):
        for si in [1, -1]:
            for sj in [1, -1]:
                for sk in [1, -1]:
                    for sl in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = si
                        p[j] = sj
                        p[k] = sk
                        p[l] = sl
                        points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    n_candidates = points.shape[0]
    
    # Greedy independent set: select points such that no two have squared distance < 4
    # Conflict: dist_sq < 4 means points are "too close"
    # We want min pairwise dist_sq >= 4 (which equals the squared norm)
    
    # For speed, use a greedy approach: process candidates in a fixed order,
    # add a candidate if it's compatible with all already-selected points.
    # To improve the result, try multiple random orderings and keep the best.
    
    best_selected = None
    best_count = 0
    
    # First, try the natural order
    selected = []
    for idx in range(n_candidates):
        if time.time() > deadline - 30:
            break
        p = points[idx]
        compatible = True
        for s in selected:
            diff = p - s
            dist_sq = int(np.sum(diff**2))
            if dist_sq < 4:
                compatible = False
                break
        if compatible:
            selected.append(p)
    
    if len(selected) > best_count:
        best_count = len(selected)
        best_selected = list(selected)
    
    # Try a few random permutations to potentially find larger independent sets
    rng = np.random.default_rng(42)
    num_trials = 5
    for trial in range(num_trials):
        if time.time() > deadline - 30:
            break
        order = rng.permutation(n_candidates)
        selected = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            p = points[idx]
            compatible = True
            for s in selected:
                diff = p - s
                dist_sq = int(np.sum(diff**2))
                if dist_sq < 4:
                    compatible = False
                    break
            if compatible:
                selected.append(p)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = list(selected)
    
    # If we found a good set, use it
    if best_count >= 2 and best_selected is not None:
        result = np.array(best_selected, dtype=np.int64)
        k = result.shape[0]
        
        # Final validation
        norms_sq = np.sum(result**2, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        
        valid = True
        for i in range(k):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, k):
                diff = result[i] - result[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and k >= 2:
            return np.around(result).astype(np.int64)
    
    # Fallback: D11 construction (220 points, weight-2)
    points_list = []
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    k = points.shape[0]
    
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = int(np.max(norms_sq))
    
    valid = True
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if valid and k >= 2:
        return np.around(points).astype(np.int64)
    
    # Final fallback: 2-point solution
    return np.array([[1,0,0,0,0,0,0,0,0,0,0],
                     [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
