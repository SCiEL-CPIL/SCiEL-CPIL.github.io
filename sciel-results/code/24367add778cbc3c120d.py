# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    
    Strategy:
    1. Include all 22 vectors of the form 2*e_i (norm^2 = 4).
    2. Include a large subset of vectors with exactly four non-zero entries +/-1 (norm^2 = 4).
       Condition: For any two distinct vectors u, v in the set, ||u-v||^2 >= 4.
       This implies u.v <= 2 for vectors of norm^2 4.
       We use a greedy algorithm to select the largest possible subset of the 5280 candidates.
       
    Max norm squared is 4.
    Min dist squared is 4 (between A and C, or C and C).
    Condition 4 >= 4 holds.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    points = []
    
    # Type A: Vectors with one non-zero entry +/-2
    # There are 11 * 2 = 22 such vectors.
    for i in range(d):
        p1 = np.zeros(d, dtype=np.int64)
        p1[i] = 2
        points.append(p1)
        
        p2 = np.zeros(d, dtype=np.int64)
        p2[i] = -2
        points.append(p2)
        
        if time.time() > deadline:
            break
    
    # Type C: Vectors with exactly four non-zero entries +/-1
    # There are C(11, 4) * 2^4 = 330 * 16 = 5280 such vectors.
    candidates = []
    for indices in itertools.combinations(range(d), 4):
        i, j, k, l = indices
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    for s4 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        p[k] = s3
                        p[l] = s4
                        candidates.append(p)
                        
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    if time.time() > deadline:
        break

    candidates = np.array(candidates, dtype=np.int64)
    
    # Greedy selection from candidates
    # We need u.v <= 2 for all selected u, v.
    selected = []
    selected_arr = [] # list of np arrays for fast dot product if needed, or just check against points
    
    # Note: Points in 'points' (Type A) are already compatible with all Type C.
    # We only need to check compatibility among Type C candidates.
    
    for c in candidates:
        if time.time() > deadline:
            break
        
        # Check against all selected Type C points
        compatible = True
        for s in selected:
            # dot product
            dot = np.dot(c, s)
            if dot > 2:
                compatible = False
                break
        
        if compatible:
            selected.append(c)
            
    # Combine Type A and selected Type C
    if selected:
        selected_arr = np.array(selected, dtype=np.int64)
        pts = np.vstack((np.array(points, dtype=np.int64), selected_arr))
    else:
        pts = np.array(points, dtype=np.int64)
        
    # Verification step
    if len(pts) > 0:
        norms_sq = np.sum(pts**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        k = len(pts)
        for i in range(k):
            for j in range(i + 1, k):
                diff = pts[i] - pts[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        if min_dist_sq < max_norm_sq:
            # Fallback to seed if somehow invalid
            return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
            
    return pts

# EVOLVE-BLOCK-END
