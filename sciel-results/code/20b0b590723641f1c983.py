"""Optimized construction of large 4-uniform sunflower-free family."""

def construct():
    from itertools import combinations
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    # Generate all 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820

    # Convert to bitmasks for fast operations
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)

    # Precompute pairwise intersections between all sets
    # inter[i][j] = masks[i] & masks[j]
    inter = np.zeros((n_total, n_total), dtype=np.int64)
    for i in range(n_total):
        inter[i, i+1:] = masks[i] & masks[i+1:]
        inter[i+1:, i] = inter[i, i+1:]

    # Key insight: For candidate c and existing family F:
    # c conflicts with pair (a,b) in F if inter[a][b] == inter[a][c] == inter[b][c]
    
    # Efficient check using vectorized operations:
    # For candidate c, group existing sets by their intersection with c.
    # For each intersection value v, let G_v = {a in F : inter[a][c] == v}
    # A conflict exists if there are a,b in G_v (a != b) such that inter[a][b] == v

    def can_add(c_idx, family_indices):
        """Check if set c_idx can be added to family_indices without creating a sunflower."""
        if len(family_indices) < 2:
            return True
        
        cm = masks[c_idx]
        fam_arr = np.array(family_indices, dtype=np.int64)
        
        # Group family members by their intersection with c
        # inter_with_c = inter[family_indices, c_idx]
        inter_with_c = inter[fam_arr, c_idx]
        
        # For each unique intersection value, check if any pair in that group has inter == v
        unique_vals = np.unique(inter_with_c)
        
        for v in unique_vals:
            # Get indices in family_indices where inter_with_c == v
            mask_v = (inter_with_c == v)
            if np.sum(mask_v) < 2:
                continue
            
            # Get the actual family indices
            group_idx = fam_arr[mask_v]
            n_group = len(group_idx)
            
            if n_group < 2:
                continue
            
            # Check all pairs in this group
            # For pairs (a, b) in group, check if inter[a][b] == v
            group_inter = inter[np.ix_(group_idx, group_idx)]
            
            # Check upper triangle (excluding diagonal)
            iu = np.triu_indices(n_group, k=1)
            if np.any(group_inter[iu] == v):
                return False
        
        return True

    # Greedy construction with multiple orderings
    best_family = []
    best_size = 0
    
    # Try several different orderings
    orderings = []
    
    # Natural order
    orderings.append(list(range(n_total)))
    
    # Random orderings
    rng = random.Random(42)
    for _ in range(8):
        perm = list(range(n_total))
        rng.shuffle(perm)
        orderings.append(perm)
    
    # Heuristic: order by "popularity" - sets that intersect many others in small ways first
    # Sort by average intersection size (smaller first)
    avg_inter = np.zeros(n_total, dtype=np.float64)
    for i in range(n_total):
        total = np.sum(np.bitwise_count(inter[i, :]) if hasattr(np, 'bitwise_count') else 
                       [bin(x).count('1') for x in inter[i, :]])
        avg_inter[i] = total / (n_total - 1)
    
    by_avg = sorted(range(n_total), key=lambda x: avg_inter[x])
    orderings.append(by_avg)
    
    # Sort by reverse
    orderings.append(list(reversed(range(n_total))))

    # Conflict-graph based ordering: nodes with fewer "sunflower conflicts" first
    # A set conflicts with another if they could form a sunflower with some third set
    # Precompute conflict degrees
    conflict_degree = np.zeros(n_total, dtype=np.int64)
    for i in range(n_total):
        mi = masks[i]
        for j in range(i + 1, n_total):
            mj = masks[j]
            inter_ij = mi & mj
            # Check if there exists k such that inter[i][k] == inter_ij and inter[j][k] == inter_ij
            # This is expensive, so approximate by counting sets with same intersection pattern
            conflict_degree[i] += 1
            conflict_degree[j] += 1
    
    by_conflict = sorted(range(n_total), key=lambda x: conflict_degree[x])
    orderings.append(by_conflict)

    for order in orderings:
        if time.time() > deadline - 15:
            break
        
        family = []
        family_set = set()
        
        for c_idx in order:
            if time.time() > deadline - 15:
                break
            if c_idx in family_set:
                continue
            
            if can_add(c_idx, family):
                family.append(c_idx)
                family_set.add(c_idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Local search: try to improve by adding sets
    if time.time() < deadline - 5:
        improved = True
        family_set = set(best_family)
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try to add a set
            for c_idx in range(n_total):
                if time.time() > deadline - 5:
                    break
                if c_idx in family_set:
                    continue
                if can_add(c_idx, best_family):
                    best_family.append(c_idx)
                    family_set.add(c_idx)
                    improved = True
                    break
    
    return [sorted(all_sets[i]) for i in best_family]
