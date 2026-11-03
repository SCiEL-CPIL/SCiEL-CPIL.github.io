    candidates = []
    # Generate splits
    def get_splits(d):
        # Simple splits: all 2s and 1s, and some mixed
        splits = []
        # Pattern 1: 2s then 1
        s = []
        rem = d
        while rem > 2:
            s.append(2)
            rem -= 2
        s.append(rem)
        splits.append(s)
        
        # Pattern 2: 1 then 2s
        s = [1]
        rem = d - 1
        while rem > 2:
            s.append(2)
            rem -= 2
        s.append(rem)
        splits.append(s)
        
        # Pattern 3: If d is small, try specific
        if d == 3:
            splits.append([1, 2])
            splits.append([2, 1])
            splits.append([3])
        if d == 4:
            splits.append([1, 3])
            splits.append([3, 1])
            splits.append([2, 2])
            splits.append([4])
        if d == 7:
            splits.append([1, 6])
            splits.append([6, 1])
            splits.append([3, 4])
            splits.append([4, 3])
            
        # Remove duplicates
        unique_splits = []
        for s in splits:
            if s not in unique_splits:
                unique_splits.append(s)
        return unique_splits

    splits_n = get_splits(n)
    splits_m = get_splits(m)
    splits_p = get_splits(p)
    
    best_result = None
    best_rank = n * m * p
    
    for sn in splits_n:
        for sm in splits_m:
            for sp in splits_p:
                if time.time() > deadline - 10:
                    break
                try:
                    res = _try_strassen_block_decomposition_with_splits(n, m, p, sn, sm, sp)
                    if _verify(res["U"], res["V"], res["W"], n, m, p):
                        rank = len(res["U"][0])
                        if rank < best_rank:
                            best_rank = rank
                            best_result = res
                except Exception:
                    pass
        if time.time() > deadline - 10:
            break
    if time.time() > deadline - 10:
        break
        
    if best_result:
        return best_result
