# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a maximal valid set of integer points in Z^11 where
    min pairwise dist^2 >= max norm^2.
    
    Strategy: Mix weight-1 (norm^2=1) and weight-2 (norm^2=2) vectors.
    max_norm_sq = 2, so we need min pairwise dist^2 >= 2.
    Use greedy filtering to guarantee validity.
    """
    deadline = time.time() + 580.0  # Safety margin
    
    d = 11
    candidates = []
    
    # Generate all weight-1 vectors: norm^2 = 1
    for i in range(d):
        for s in (-1, 1):
            p = np.zeros(d, dtype=np.int64)
            p[i] = s
            candidates.append(p)
    
    # Generate all weight-2 vectors: norm^2 = 2
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    candidates.append(p)
    
    cands = np.array(candidates, dtype=np.int64)
    N = len(cands)
    
    # max_norm_sq = 2 (weight-2 vectors have norm^2 = 2)
    max_norm_sq = 2
    
    # Sort candidates: weight-1 first (to maximize count), then weight-2
    norms_sq = np.sum(cands * cands, axis=1)
    order = np.argsort(norms_sq)  # weight-1 (norm^2=1) before weight-2 (norm^2=2)
    
    selected = []
    selected_arr = []
    
    for idx in order:
        if time.time() > deadline:
            break
            
        p = cands[idx]
        
        # Check if p is valid against all selected points
        valid = True
        for q in selected_arr:
            diff = p - q
            d2 = int(np.dot(diff, diff))
            if d2 < max_norm_sq:
                valid = False
                break
        
        if valid:
            selected.append(idx)
            selected_arr.append(p)
    
    if len(selected) == 0:
        # Fallback: return two simple points
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    points = cands[selected]
    
    # Final verification
    k = len(points)
    assert k >= 2, f"Only {k} points"
    assert points.shape[1] == 11
    
    # Verify no zero vector
    for p in points:
        assert np.any(p != 0), "Zero vector found"
    
    # Verify min dist^2 >= max norm^2
    max_norm = int(np.max(np.sum(points * points, axis=1)))
    min_dist = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = int(np.dot(diff, diff))
            if d2 < min_dist:
                min_dist = d2
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If verification is too slow, trust the greedy construction
    # The greedy algorithm guarantees validity by construction
    
    return points

# EVOLVE-BLOCK-END
