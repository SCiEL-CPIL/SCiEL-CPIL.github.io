# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    deadline = time.time() + 600 - 10  # safety margin
    
    # Generate all vectors with two non-zero entries being +/-1
    # There are C(11, 2) = 55 pairs
    # For each pair (i, j), i < j:
    # v1 = e_i + e_j  (norm^2 = 2)
    # v2 = -e_i - e_j (norm^2 = 2)
    # v3 = e_i - e_j  (norm^2 = 2)
    # v4 = -e_i + e_j (norm^2 = 2)
    
    points_list = []
    indices = list(range(d))
    
    for i in range(d):
        for j in range(i + 1, d):
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            points_list.append(v1)
            
            v2 = -v1
            points_list.append(v2)
            
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = 1
            v3[j] = -1
            points_list.append(v3)
            
            v4 = -v3
            points_list.append(v4)
    
    # Check validity: min squared distance >= max squared norm
    # For these points, all have norm^2 = 2
    # Distances: 
    #   Same vector: dist^2 = 0 (but we don't have duplicates)
    #   Opposite vectors (v and -v): dist^2 = 4*2 = 8
    #   Orthogonal vectors (disjoint supports): dist^2 = 2 + 2 = 4
    #   Vectors sharing one index: e.g., e_i+e_j and e_i-e_k, dist^2 = (1-1)^2 + (1-0)^2 + (0-(-1))^2 = 0 + 1 + 1 = 2? Let me recalculate
    
    # Actually let me just compute it properly
    points = np.array(points_list, dtype=np.int64)
    k = len(points)
    
    # Verify the condition
    max_norm_sq = 0
    for p in points:
        norm_sq = np.dot(p, p)
        if norm_sq > max_norm_sq:
            max_norm_sq = norm_sq
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    
    # Check if valid
    if min_dist_sq >= max_norm_sq:
        return points
    
    # If not valid, fall back to the seed
    points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    return points


# EVOLVE-BLOCK-END
