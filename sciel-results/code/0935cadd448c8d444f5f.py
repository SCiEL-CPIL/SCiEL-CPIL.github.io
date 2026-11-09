import numpy as np
from itertools import combinations
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using the D11 lattice roots to maximize the number of points.
    
    The D11 construction uses all vectors with exactly 2 non-zero entries (±1).
    C(11,2) * 4 = 55 * 4 = 220 points.
    Each point has norm squared = 2.
    The minimum pairwise distance squared between any two points is 2.
    Thus min_dist_sq = max_norm_sq = 2, satisfying the constraint.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for safety
    
    # D11 construction: all vectors with exactly 2 non-zero entries (±1)
    # C(11,2) * 4 = 55 * 4 = 220 points
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
    
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Verify: no zero vectors
    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        norms_sq = np.sum(points**2, axis=1)
    
    max_norm_sq = int(np.max(norms_sq))
    
    # Verify pairwise distances
    valid = True
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = int(np.sum(diff**2))
            if dist_sq < max_norm_sq:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: 2-point solution
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = np.around(points).astype(np.int64)
    return result
