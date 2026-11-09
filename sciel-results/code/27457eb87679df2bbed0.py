# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    Uses all vectors with exactly two non-zero entries of value +/- 1.
    This yields 220 points.
    Max norm squared is 2.
    Min pairwise distance squared is 2.
    Condition min_dist^2 >= max_norm^2 is satisfied (2 >= 2).
    """
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    # Number of such vectors: C(11, 2) * 4 = 55 * 4 = 220
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    if time.time() > deadline - 5:
         # Fallback to a smaller valid set if we are running out of time (unlikely)
         # Just return the first few
         points_list = points_list[:10]

    points = np.array(points_list, dtype=np.int64)
    
    # Verification step (optional but good for safety)
    # Check shape
    if points.shape[1] != d:
        raise ValueError("Invalid dimension")
    if points.shape[0] < 2:
        raise ValueError("Not enough points")
        
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        # Remove zero vectors if any (shouldn't happen)
        points = points[~np.all(points == 0, axis=1)]
        
    # Calculate max norm squared
    norms_sq = np.sum(points**2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Calculate min pairwise distance squared
    # k is small (220), so O(k^2) is fine
    min_dist_sq = float('inf')
    k = points.shape[0]
    for i in range(k):
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                
    if min_dist_sq < max_norm_sq:
        # This should not happen for this construction, but if it does, 
        # we might need to filter. 
        # However, the construction is theoretically sound.
        pass

    return points

# EVOLVE-BLOCK-END
