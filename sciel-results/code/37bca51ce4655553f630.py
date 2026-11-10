# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    using the E8 lattice roots construction to maximize the number of points.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    points_list = []
    
    # 2E8 ⊕ 2D3 lattice: 252 minimal vectors
    # E8 part (first 8 dims): 240 vectors of norm 2
    #   Type A: (±1, ±1, 0, 0, 0, 0, 0, 0) - C(8,2)*4 = 112 vectors
    #   Type B: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) even minuses - 128 vectors
    #     But we need integers, so scale by 2: norm becomes 4
    # D3 part (last 3 dims): 12 vectors of norm 2
    #   (±1, ±1, 0), (±1, 0, ±1), (0, ±1, ±1) - C(3,2)*4 = 12 vectors
    #
    # To have consistent norms, use:
    # E8 Type A scaled by 1: (±1, ±1, 0,...,0) norm 2, 112 vectors in dims 0-7
    # E8 Type B: can't use half-integers, skip or scale
    # D3: (0,...,0,±1,±1,0), etc. norm 2, 12 vectors in dims 8-10
    #
    # Total: 112 + 12 = 124. Not enough.
    #
    # Alternative: Use the full E8 root system with integer scaling
    # Scale E8 by 2: all vectors have even or odd integer coords
    # Type A: (±2, ±2, 0,...,0) norm 8, 112 vectors
    # Type B: (±1, ±1, ..., ±1) even minuses, norm 8, 128 vectors
    # Total E8: 240 vectors, all norm 8
    #
    # D3 scaled by 2: (±2, ±2, 0), etc. norm 8, 12 vectors
    # Total: 240 + 12 = 252 vectors, all norm 8
    #
    # Check distance: min distance between any two should be >= 8
    # Within E8: min dist^2 = 4 (between (1,1,0,...) and (1,-1,0,...) after scaling by 2: (2,2,0) vs (2,-2,0) dist^2=16)
    # Actually in E8, min distance between distinct roots is 2 (norm of difference is 2)
    # Scaled by 2: min distance^2 = 4*2 = 8. Good, equals max norm^2 = 8.
    
    # E8 Type A: (±2, ±2, 0, 0, 0, 0, 0, 0) in first 8 dims
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    # E8 Type B: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of -1s
    for mask in range(256):
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 0:
            p = np.zeros(d, dtype=np.int64)
            for k in range(8):
                if (mask >> k) & 1:
                    p[k] = -1
                else:
                    p[k] = 1
            points_list.append(p)
    
    # D3: (0, 0, 0, 0, 0, 0, 0, 0, ±2, ±2, 0), etc. in last 3 dims (indices 8,9,10)
    for i, j in combinations(range(8, 11), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the solution
    k = points.shape[0]
    if k < 2:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Check no zero vector
    norms_sq = np.sum(points**2, axis=1)
    if np.any(norms_sq == 0):
        mask = norms_sq > 0
        points = points[mask]
        k = points.shape[0]
        if k < 2:
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
        norms_sq = np.sum(points**2, axis=1)
    
    # Check pairwise distances using vectorized computation for speed
    min_dist_sq = float('inf')
    max_norm_sq = int(np.max(norms_sq))
    
    # For k=252, k(k-1)/2 = 31626 pairs. Use vectorized chunks for speed.
    valid = True
    chunk_size = 50
    for i in range(k):
        if time.time() > deadline:
            break
        for j_start in range(i+1, k, chunk_size):
            j_end = min(j_start + chunk_size, k)
            # Vectorized distance computation for points[i] vs points[j_start:j_end]
            diffs = points[i] - points[j_start:j_end]
            dists_sq = np.sum(diffs**2, axis=1)
            min_chunk = int(np.min(dists_sq))
            if min_chunk < min_dist_sq:
                min_dist_sq = min_chunk
            if min_chunk < max_norm_sq:
                valid = False
                break
        if not valid:
            break
        if time.time() > deadline:
            break
    
    # Check validity
    valid = valid and (min_dist_sq >= max_norm_sq)
    
    if not valid:
        # Fallback: use the D11 construction (220 points)
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
        
        min_dist_sq = float('inf')
        max_norm_sq = int(np.max(norms_sq))
        valid = True
        for i in range(k):
            for j in range(i+1, k):
                diff = points[i] - points[j]
                dist_sq = int(np.sum(diff**2))
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                if dist_sq < max_norm_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid or min_dist_sq < max_norm_sq:
            # Final fallback: 2-point solution
            return np.array([[1,0,0,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = np.around(points).astype(np.int64)
    return result


# EVOLVE-BLOCK-END
