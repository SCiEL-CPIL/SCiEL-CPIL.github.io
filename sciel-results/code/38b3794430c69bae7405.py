import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    
    # Fallback: D_11 roots (220 points)
    def get_d11_points():
        pts = []
        for i, j in combinations(range(d), 2):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    pts.append(p)
        return np.array(pts)
    
    fallback_pts = get_d11_points()
    best_pts = fallback_pts
    best_k = fallback_pts.shape[0]
    
    # Try to build a larger set from weight-3 vectors
    # Vectors with exactly 3 non-zero entries, each +/- 1
    # Total: C(11, 3) * 2^3 = 165 * 8 = 1320
    
    all_vecs = []
    for indices in combinations(range(d), 3):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[indices[0]] = s1
                    p[indices[1]] = s2
                    p[indices[2]] = s3
                    all_vecs.append(p)
    
    all_vecs = np.array(all_vecs)
    
    # Greedy selection
    # We want max subset where min_dist_sq >= 3
    # Note: Max norm sq is 3. So condition is dist_sq >= 3.
    
    selected = []
    selected_set = [] # List of tuples for fast lookup? No, just list of arrays.
    
    # To speed up, we can use a set of "forbidden" pairs or just check against selected list.
    # With k ~ 500, checks are 500 * 1320 ~ 660k. Fast.
    
    # Shuffle to avoid bias? Or just take in order.
    # Taking in order might be fine.
    
    for p in all_vecs:
        if time.time() > deadline:
            break
            
        valid = True
        # Check against all selected points
        # Optimization: if selected is large, this is O(k).
        for q in selected:
            diff = p - q
            dist_sq = np.sum(diff**2)
            if dist_sq < 3:
                valid = False
                break
                
        if valid:
            selected.append(p)
            
    if len(selected) > best_k:
        best_pts = np.array(selected)
        best_k = len(selected)
        
    # Final verification and return
    if best_pts.shape[1] != d:
        return get_d11_points()
        
    # Ensure no zero vectors (should be true for weight 2 and 3)
    # Ensure min dist >= max norm
    
    max_norm_sq = np.max(np.sum(best_pts**2, axis=1))
    min_dist_sq = float('inf')
    
    k = best_pts.shape[0]
    for i in range(k):
        for j in range(i+1, k):
            diff = best_pts[i] - best_pts[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm_sq:
                # Invalid, return fallback
                return fallback_pts
                
    return best_pts
