import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    edge_order = P['edge_order']
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute masks
    masks = []
    for triple in edge_order:
        m = 0
        for v in triple:
            m |= (1 << v)
        masks.append(m)
    
    # Check if a perfect matching exists using only the first K triples
    def feasible(K):
        sub_masks = masks[:K]
        unique_masks = list(set(sub_masks))
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(used_mask, remaining):
            if remaining == 0:
                return used_mask == full_mask
            if remaining < 0:
                return False
            unused = full_mask ^ used_mask
            if bin(unused).count('1') < 3 * remaining:
                return False
            for m in unique_masks:
                if m & used_mask:
                    continue
                if dfs(used_mask | m, remaining - 1):
                    return True
            return False
        
        return dfs(0, 6)
    
    # Binary search for minimum K
    lo, hi = 1, len(edge_order)
    best_K = hi
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if feasible(mid):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Find actual matching for best_K
    sub_masks = masks[:best_K]
    unique_masks = list(set(sub_masks))
    
    # Map mask to first index
    mask_to_first_idx = {}
    for i in range(best_K):
        m = masks[i]
        if m not in mask_to_first_idx:
            mask_to_first_idx[m] = i
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def find_matching(used_mask, remaining):
        if remaining == 0:
            if used_mask == full_mask:
                return ()
            else:
                return None
        if remaining < 0:
            return None
        unused = full_mask ^ used_mask
        if bin(unused).count('1') < 3 * remaining:
            return None
        for m in unique_masks:
            if m & used_mask:
                continue
            result = find_matching(used_mask | m, remaining - 1)
            if result is not None:
                idx = mask_to_first_idx[m]
                return (idx,) + result
        return None
    
    result = find_matching(0, 6)
    
    if result is None:
        return [list(range(i, i+3)) for i in range(0, P['n'], 3)]
    
    matching = [list(edge_order[i]) for i in result]
    return matching

