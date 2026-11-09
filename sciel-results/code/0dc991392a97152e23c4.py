import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    d = 11
    pts = []
    
    # E8 construction: all integer vectors in Z^8 with even sum of coordinates,
    # norm_sq = 2. This gives 240 points.
    # E8 roots: weight 2 (28*4=112) + weight 4 (70*2=140) = 240 points
    
    # Weight-2 vectors in Z^8
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(8, dtype=np.int64)
                p[i] = si
                p[j] = sj
                pts.append(p)
    
    # Weight-4 vectors in Z^8 with even number of -1's (0, 2, or 4 negative signs)
    for quad in combinations(range(8), 4):
        for neg_count in [0, 2, 4]:
            for neg_positions in combinations(range(4), neg_count):
                p = np.zeros(8, dtype=np.int64)
                for idx, coord in enumerate(quad):
                    if idx in neg_positions:
                        p[coord] = -1
                    else:
                        p[coord] = 1
                pts.append(p)
    
    # Now embed into Z^11 by padding with 3 zeros, and also add D_3 components
    # Actually, let's think differently. We need points in Z^11.
    # 
    # Strategy: Use two orthogonal subspaces.
    # Subspace A: first 8 coordinates, last 3 are zero. Points from E8 (norm_sq=2).
    # Subspace B: last 3 coordinates, first 8 are zero. Points from D_3 or A_2.
    # 
    # But we need ALL pairwise distances >= max norm_sq.
    # If we use E8 points (norm_sq=2) in first 8 coords, and D_3 points (norm_sq=2) in last 3 coords,
    # then a point from E8 has norm_sq=2, a point from D_3 has norm_sq=2.
    # Distance between E8 point and D_3 point: sqrt(2+2)=2, so dist_sq=4 >= 2. Good.
    # Distance between two E8 points: min dist_sq in E8 is 2. Good.
    # Distance between two D_3 points: need to check.
    #
    # D_3 in Z^3: weight-2 vectors. C(3,2)*4 = 12 points. Norm_sq=2.
    # Two D_3 points: if disjoint support (impossible in 3D for weight-2, they always share at least 1),
    # if share 1 index: dist_sq = 2 (if same sign) or 6 (if different sign). Min is 2. Good.
    #
    # So: E8 (240 pts) in coords 0-7, D_3 (12 pts) in coords 8-10.
    # Total: 240 + 12 = 252 points. All have norm_sq=2.
    # Min pairwise dist_sq = 2. Valid!
    
    # Let me reconsider. The inspiration says "2*E8 + 2*D3 (252 pts)".
    # Maybe they mean something different. Let me just do E8 + D_3 = 240 + 12 = 252.
    
    # Clear pts and rebuild
    pts = []
    
    # E8 points in first 8 coordinates
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                pts.append(p)
    
    for quad in combinations(range(8), 4):
        for neg_count in [0, 2, 4]:
            for neg_positions in combinations(range(4), neg_count):
                p = np.zeros(d, dtype=np.int64)
                for idx, coord in enumerate(quad):
                    if idx in neg_positions:
                        p[coord] = -1
                    else:
                        p[coord] = 1
                pts.append(p)
    
    # D_3 points in last 3 coordinates (indices 8, 9, 10)
    for i, j in combinations([8, 9, 10], 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                pts.append(p)
    
    result = np.array(pts)
    
    # Verify validity
    k = result.shape[0]
    max_norm_sq = 0
    for p in result:
        n = np.sum(p**2)
        if n > max_norm_sq:
            max_norm_sq = n
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i+1, k):
            diff = result[i] - result[j]
            d2 = np.sum(diff**2)
            if d2 < min_dist_sq:
                min_dist_sq = d2
    
    if min_dist_sq >= max_norm_sq and max_norm_sq > 0:
        return np.around(result).astype(np.int64)
    else:
        # Fallback: D_11
        fallback = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    fallback.append(p)
        return np.array(fallback)
