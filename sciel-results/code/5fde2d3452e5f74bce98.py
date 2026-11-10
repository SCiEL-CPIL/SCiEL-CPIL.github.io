import numpy as np
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    best_points = []
    best_k = 0
    
    # Try different weights
    for w in [3, 4, 5]:
        # Determine max allowed intersection
        # Condition: min_dist_sq >= max_norm_sq = w
        # dist_sq = 2w - 2 * dot
        # 2w - 2 * dot >= w  => w >= 2 * dot => dot <= w/2
        # dot is sum of products on intersection of size k.
        # Max dot for intersection k is k.
        # So we need k <= w/2.
        max_intersect = w // 2
        
        # Generate all w-subsets
        subsets = list(combinations(range(d), w))
        
        # Greedy selection
        selected = []
        for s in subsets:
            s_set = set(s)
            compatible = True
            for s_prev in selected:
                if len(s_set.intersection(s_prev)) > max_intersect:
                    compatible = False
                    break
            if compatible:
                selected.append(s)
        
        # Generate points
        points = []
        for s in selected:
            # Generate all 2^w sign vectors
            for mask in range(1 << w):
                vec = np.zeros(d, dtype=np.int64)
                for i, idx in enumerate(s):
                    if (mask >> i) & 1:
                        vec[idx] = 1
                    else:
                        vec[idx] = -1
                points.append(vec)
        
        if len(points) > best_k:
            best_k = len(points)
            best_points = points
            
    if not best_points:
        # Fallback to weight 2
        for i in range(d):
            for j in range(i+1, d):
                for s1 in [-1, 1]:
                    for s2 in [-1, 1]:
                        vec = np.zeros(d, dtype=np.int64)
                        vec[i] = s1
                        vec[j] = s2
                        best_points.append(vec)
        
    return np.array(best_points, dtype=np.int64)
