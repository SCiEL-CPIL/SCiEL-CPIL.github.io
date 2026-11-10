import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Generate all 4-subsets of {0, ..., 10}
    four_subsets = list(combinations(range(d), 4))
    n_subsets = len(four_subsets)  # 330
    
    # Precompute compatibility masks
    compat = [0] * n_subsets
    for i in range(n_subsets):
        si = set(four_subsets[i])
        mask = 0
        for j in range(n_subsets):
            sj = set(four_subsets[j])
            if len(si & sj) <= 2:
                mask |= (1 << j)
        compat[i] = mask
        
    best_clique = 0
    best_size = 0
    
    def solve(avail_mask, current_clique, current_size):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return

        # Pruning
        if current_size + avail_mask.bit_count() <= best_size:
            return

        if avail_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique
            return

        # Pick a vertex to branch on
        # Use the vertex with the lowest index for determinism
        lsb = avail_mask & -avail_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        new_avail_include = (avail_mask & compat[v]) & ~lsb
        solve(new_avail_include, current_clique | lsb, current_size + 1)
        
        # Branch 2: Exclude v
        new_avail_exclude = avail_mask & ~lsb
        solve(new_avail_exclude, current_clique, current_size)
        
    # Start search
    initial_mask = (1 << n_subsets) - 1
    solve(initial_mask, 0, 0)
    
    # Construct points
    points = []
    for idx in range(n_subsets):
        if (best_clique >> idx) & 1:
            support = four_subsets[idx]
            for signs in range(16):
                v = np.zeros(d, dtype=np.int64)
                for k, pos in enumerate(support):
                    if (signs >> k) & 1:
                        v[pos] = 1
                    else:
                        v[pos] = -1
                points.append(v)
    
    if len(points) < 2:
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        
    pts_array = np.array(points, dtype=np.int64)
    
    # Validation
    n_pts = len(pts_array)
    if n_pts > 0:
        norms = np.sum(pts_array ** 2, axis=1)
        max_norm = norms.max()
        min_dist = float('inf')
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = pts_array[i] - pts_array[j]
                dist_sq = int(np.sum(diff ** 2))
                if dist_sq < min_dist:
                    min_dist = dist_sq
        
        if min_dist < max_norm:
            points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
            pts_array = np.array(points, dtype=np.int64)
            
    return np.around(pts_array).astype(np.int64)
