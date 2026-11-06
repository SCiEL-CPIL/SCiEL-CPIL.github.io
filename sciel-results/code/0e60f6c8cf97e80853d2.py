def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts: conflict[i] contains j if lcm(nums[i], nums[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            ai, aj = nums[i], nums[j]
            g = gcd(ai, aj)
            if ai * aj // g <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Use integer weights: weight[i] = LCM / nums[i] where LCM is common denominator
    # For comparing sums, we use common denominator = product of all nums, but that's huge.
    # Instead, use floating point for comparison but track exact via fractions at the end.
    # Actually, let's use a large common denominator approach.
    # Better: use the fact that we're maximizing sum of 1/nums[i], so sort by 1/nums[i] descending.
    
    # Strategy: 
    # 1. Enumerate subsets of "critical" small numbers (2-20) that are independent.
    # 2. For each valid small subset, solve MWIS on remaining compatible large numbers.
    
    # Expand small range to include numbers up to 20 (index 18 for num=20)
    small_range = list(range(0, 19))  # indices for nums 2-20
    large_range = list(range(19, n))  # indices for nums 21-256
    
    best_sum_float = 0.0
    best_set = set()
    
    # Cache for large MWIS results
    large_cache = {}
    
    def solve_large_mwis(compatible_frozenset):
        """Solve MWIS on large numbers compatible with a given set of small numbers."""
        if compatible_frozenset in large_cache:
            return large_cache[compatible_frozenset]
        
        compatible_large = set(compatible_frozenset)
        if not compatible_large:
            large_cache[compatible_frozenset] = (set(), 0.0)
            return set(), 0.0
        
        # Greedy initialization: sort by 1/x descending, add if no conflict
        candidates = sorted(compatible_large, key=lambda x: -1.0/nums[x])
        selected = set()
        sel_sum = 0.0
        for j in candidates:
            if not (conflict[j] & selected):
                selected.add(j)
                sel_sum += 1.0 / nums[j]
        
        # Local search with multiple passes
        for iteration in range(10):
            if time.time() > deadline:
                break
            
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                # Try adding any compatible node
                for j in compatible_large:
                    if j not in selected and not (conflict[j] & selected):
                        selected.add(j)
                        sel_sum += 1.0 / nums[j]
                        improved = True
                
                # Try 1-for-1 swaps
                for i in list(selected):
                    s_without = selected - {i}
                    base_sum = sel_sum - 1.0 / nums[i]
                    for j in compatible_large:
                        if j not in s_without and not (conflict[j] & s_without):
                            new_sum = base_sum + 1.0 / nums[j]
                            if new_sum > sel_sum + 1e-15:
                                selected = s_without | {j}
                                sel_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                
                # Try 2-for-1 swaps (remove 2, add 1)
                if not improved and len(selected) >= 2:
                    sel_list = list(selected)
                    for a_idx in range(len(sel_list)):
                        if time.time() > deadline:
                            break
                        for b_idx in range(a_idx+1, len(sel_list)):
                            i, j = sel_list[a_idx], sel_list[b_idx]
                            s_without = selected - {i, j}
                            base_sum = sel_sum - 1.0 / nums[i] - 1.0 / nums[j]
                            for k in compatible_large:
                                if k not in s_without and not (conflict[k] & s_without):
                                    new_sum = base_sum + 1.0 / nums[k]
                                    if new_sum > sel_sum + 1e-15:
                                        selected = s_without | {k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try 1-for-2 swaps (remove 1, add 2)
                if not improved:
                    for i in list(selected):
                        if time.time() > deadline:
                            break
                        s_without = selected - {i}
                        base_sum = sel_sum - 1.0 / nums[i]
                        # Find two nodes that can both be added
                        addable = [j for j in compatible_large if j not in s_without and not (conflict[j] & s_without)]
                        for a_idx in range(len(addable)):
                            for b_idx in range(a_idx+1, len(addable)):
                                j, k = addable[a_idx], addable[b_idx]
                                if not (conflict[j] & conflict[k]) and not (conflict[j] & s_without) and not (conflict[k] & s_without):
                                    new_sum = base_sum + 1.0 / nums[j] + 1.0 / nums[k]
                                    if new_sum > sel_sum + 1e-15:
                                        selected = s_without | {j, k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
        
        result = (selected, sel_sum)
        large_cache[compatible_frozenset] = result
        return selected, sel_sum
    
    # Enumerate subsets of small numbers
    # 2^19 = 524288 is manageable with fast independence check
    small_count = len(small_range)
    
    # Precompute conflict masks for small numbers for faster independence check
    small_conflict_mask = [0] * small_count
    for i in range(small_count):
        idx = small_range[i]
        for j in range(small_count):
            if j != i and small_range[j] in conflict[idx]:
                small_conflict_mask[i] |= (1 << j)
    
    for mask in range(1 << small_count):
        if time.time() > deadline:
            break
        
        # Fast independence check using bit operations
        valid = True
        s_small = set()
        for i in range(small_count):
            if mask & (1 << i):
                if small_conflict_mask[i] & mask:
                    valid = False
                    break
                s_small.add(small_range[i])
        
        if not valid:
            continue
        
        # Find compatible large numbers
        compatible_large = frozenset(j for j in large_range if not (conflict[j] & s_small))
        
        large_set, large_sum = solve_large_mwis(compatible_large)
        
        small_sum = sum(1.0 / nums[x] for x in s_small)
        total_sum = small_sum + large_sum
        
        if total_sum > best_sum_float:
            best_sum_float = total_sum
            best_set = s_small | large_set
    
    # Final local search on the complete solution
    if best_set and time.time() < deadline:
        selected = set(best_set)
        sel_sum = sum(1.0 / nums[i] for i in selected)
        
        for iteration in range(5):
            if time.time() > deadline:
                break
            
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                # Try adding any compatible node
                for j in range(n):
                    if j not in selected and not (conflict[j] & selected):
                        selected.add(j)
                        sel_sum += 1.0 / nums[j]
                        improved = True
                
                # Try 1-for-1 swaps
                for i in list(selected):
                    s_without = selected - {i}
                    base_sum = sel_sum - 1.0 / nums[i]
                    for j in range(n):
                        if j not in s_without and not (conflict[j] & s_without):
                            new_sum = base_sum + 1.0 / nums[j]
                            if new_sum > sel_sum + 1e-15:
                                selected = s_without | {j}
                                sel_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                
                # Try 2-for-1 swaps
                if not improved and len(selected) >= 2:
                    sel_list = list(selected)
                    for a_idx in range(len(sel_list)):
                        if time.time() > deadline:
                            break
                        for b_idx in range(a_idx+1, len(sel_list)):
                            i, j = sel_list[a_idx], sel_list[b_idx]
                            s_without = selected - {i, j}
                            base_sum = sel_sum - 1.0 / nums[i] - 1.0 / nums[j]
                            for k in range(n):
                                if k not in s_without and not (conflict[k] & s_without):
                                    new_sum = base_sum + 1.0 / nums[k]
                                    if new_sum > sel_sum + 1e-15:
                                        selected = s_without | {k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try 1-for-2 swaps
                if not improved:
                    for i in list(selected):
                        if time.time() > deadline:
                            break
                        s_without = selected - {i}
                        base_sum = sel_sum - 1.0 / nums[i]
                        addable = [j for j in range(n) if j not in s_without and not (conflict[j] & s_without)]
                        for a_idx in range(len(addable)):
                            for b_idx in range(a_idx+1, len(addable)):
                                j, k = addable[a_idx], addable[b_idx]
                                if not (conflict[j] & conflict[k]) and not (conflict[j] & s_without) and not (conflict[k] & s_without):
                                    new_sum = base_sum + 1.0 / nums[j] + 1.0 / nums[k]
                                    if new_sum > sel_sum + 1e-15:
                                        selected = s_without | {j, k}
                                        sel_sum = new_sum
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
        
        if sel_sum > best_sum_float:
            best_set = selected
    
    # Convert to list of actual numbers
    result = sorted(nums[i] for i in best_set)
    if not result:
        return [2]
    return result
