import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590  # Safety margin
    
    # All integer vectors with squared norm 4:
    # Type A: one coordinate is +/-2, rest 0. (22 vectors)
    # Type B: two coordinates are +/-1, rest 0. (11*10*4 = 440 vectors)
    # Total: 462 vectors
    #
    # Distance check:
    # Same type, different vectors: dist_sq = 4 (e.g., (2,0,..) vs (-2,0,..) -> dist=4; (2,0,..) vs (0,2,..) -> dist=8; (1,1,..) vs (1,-1,..) -> dist=4)
    # Different types: 
    #   (2,0,..) and (1,1,..): if they share a coordinate, e.g., (2,0,..) at pos i and (1,1,..) at pos i,j: diff = (1,-1,..) or (1,1,..) -> dist_sq = 2
    #   (2,0,..) and (-1,-1,..): similar
    #   (1,1,..) and (2,0,..): dist_sq = 2 if share a coordinate
    #   
    # Wait, let me recheck. If v1 = (2,0,0,...) and v2 = (1,1,0,...), then v1-v2 = (1,-1,0,...) with norm^2 = 2.
    # max_norm = 4. So 2 < 4. INVALID.
    #
    # So we CANNOT mix types. We must pick either all Type A or all Type B.
    # 
    # Type A: 22 vectors. Min dist between any two: (2,0,..) and (-2,0,..) -> dist=4. (2,0,..) and (2,0,..) are same, not allowed. (2,0,..) and (0,2,..) -> dist=8.
    #   So min_dist = 4 = max_norm. VALID. Size = 22.
    #
    # Type B: 440 vectors. Min dist: (1,1,..) and (1,-1,..) -> dist=4. (1,1,..) and (-1,1,..) -> dist=4. (1,1,..) and (1,1,..) same.
    #   (1,1,..) and (0,0,..) not in set.
    #   (1,1,..) and (-1,-1,..) -> dist=8.
    #   So min_dist = 4 = max_norm. VALID. Size = 440.
    #
    # So Type B gives 440 points. That's much better than 145.
    
    # But can we do better by using vectors with larger norms?
    # If max_norm = N, we need min_dist >= N.
    # For norm 1: only 22 vectors (unit vectors). Min dist = 2. 2 >= 1. VALID. Size = 22.
    # For norm 4: as above, 440 vectors (Type B only).
    # For norm 9: (3,0,..) -> 22 vectors. (2,1,..) -> 11*10*4 = 440 vectors. (1,1,1,..) -> C(11,3)*8 = 165*8 = 1320 vectors.
    #   But mixing types: (3,0,..) and (2,1,..) share coordinate: dist_sq = (1,-1,..) = 2 < 9. INVALID.
    #   So we'd have to pick one type. Type C (1,1,1): 1320 vectors.
    #   Min dist between two (1,1,1) vectors: (1,1,1,..) and (1,1,-1,..) -> dist=4. 4 < 9. INVALID!
    #   So norm 9 doesn't work for Type C.
    #
    # What about norm 1? 22 points.
    # Norm 4: 440 points.
    # 
    # Can we mix norm 1 and norm 4? 
    # max_norm would be 4. Need min_dist >= 4.
    # (1,0,..) and (1,1,..): dist_sq = (0,-1,..) = 1 < 4. INVALID.
    # So no mixing.
    
    # So 440 seems to be a strong lower bound. Let's return all 440 Type B vectors.
    
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    pts_array = np.array(points, dtype=np.int64)
    
    # Verify
    n_pts = len(pts_array)
    norms = np.sum(pts_array ** 2, axis=1)
    max_norm = norms.max()
    
    min_dist = float('inf')
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            diff = pts_array[i] - pts_array[j]
            dist_sq = int(np.sum(diff ** 2))
            if dist_sq < min_dist:
                min_dist = dist_sq
    
    if min_dist < max_norm or n_pts < 2:
        # Fallback to the 22 unit vectors
        pts_array = np.array([np.eye(d, dtype=np.int64)[i] * s for i in range(d) for s in [1, -1]])
    
    return np.around(pts_array).astype(np.int64)
