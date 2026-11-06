import time

def construct():
    start_time = time.time()
    
    # 1. Generate 5-smooth numbers up to 128
    smooth_nums = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_nums.append(x)
    
    is_smooth = set(smooth_nums)
    n_smooths = len(smooth_nums)
    
    # best[n] will store the tuple of numbers representing the minimum term count
    best = [None] * 129
    
    # Precompute divisibility conflicts for fast checking
    # conflict[i][j] is True if smooth_nums[i] divides smooth_nums[j] or vice versa
    # We only care about distinct numbers.
    # Since we iterate in sorted order, we can just check on the fly.
    # But for speed, we can precompute a matrix or just use modulo.
    
    # Backtracking to find all valid antichains with sum <= 128
    # State: current_sum, current_tuple, next_index_to_consider
    # We want to minimize the length of the tuple for each sum.
    
    # Optimization: If current_sum is already covered by a solution with <= current_length, 
    # we might still want to extend it to cover larger sums, but we can prune if 
    # current_length >= best[current_sum] and we are only looking for improvements for current_sum?
    # No, a longer set for a smaller sum might lead to a valid set for a larger sum 
    # that has no other representation. 
    # However, since we are iterating all subsets, we will find all valid antichains.
    # The number of such antichains is small.
    
    def dfs(current_sum, current_set, start_idx):
        # Update best for current_sum if current_set is better
        if current_sum > 0:
            if best[current_sum] is None or len(current_set) < len(best[current_sum]):
                best[current_sum] = current_set
        
        # Pruning: if current_sum + min_possible_next > 128, we can stop?
        # min_possible_next is smooth_nums[start_idx] if start_idx < n_smooths else inf
        
        for i in range(start_idx, n_smooths):
            val = smooth_nums[i]
            new_sum = current_sum + val
            if new_sum > 128:
                break
            
            # Check antichain property
            valid = True
            for existing in current_set:
                if existing < val:
                    if val % existing == 0:
                        valid = False
                        break
                else:
                    if existing % val == 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Recurse
            # Convert tuple to list for appending, then back to tuple? 
            # Tuples are immutable and hashable, good for storage.
            # current_set is a tuple.
            dfs(new_sum, current_set + (val,), i + 1)
            
            # Time check
            if time.time() - start_time > 50.0:
                return

    dfs(0, (), 0)
    
    # Fallback for any n that wasn't found (should not happen for 1..128)
    # If not found, we need a valid representation.
    # The problem states "All targets require valid representations".
    # If our search is complete, all should be found.
    # If timeout occurred, we might have gaps.
    # Let's ensure we have a fallback strategy if some best[n] is None.
    
    # Simple fallback: use the current program's logic for any missing n?
    # Or just trust the DP. The DP is exhaustive for sum <= 128.
    # The number of antichains is finite and small.
    
    # Construct the final list
    result = []
    for n in range(1, 129):
        if best[n] is not None:
            result.append(list(best[n]))
        else:
            # This should not happen, but to be safe, try a simple greedy or reuse seed logic?
            # For now, if it happens, it's a bug in my assumption or timeout.
            # Let's try to find one quickly using the seed's method for just this n.
            # But since we are in a tight loop, let's just raise an error or use a dummy?
            # The task says "invalid result ... scores 0".
            # We must ensure validity.
            # Given the small range, the DP should find them all well within 50s.
            # Let's add a simple check: if n is smooth, [n] is valid.
            if n in is_smooth:
                result.append([n])
            else:
                # Try 2 terms
                found = False
                for a in smooth_nums:
                    if a >= n: break
                    b = n - a
                    if b <= a: continue
                    if b in is_smooth:
                        if b % a != 0 and a % b != 0:
                            result.append([a, b])
                            found = True
                            break
                if not found:
                    # Try 3 terms
                    found = False
                    for a in smooth_nums:
                        if 3*a >= n: break
                        for b in smooth_nums:
                            if b <= a: continue
                            if a + 2*b >= n: break
                            c = n - a - b
                            if c <= b: continue
                            if c in is_smooth:
                                # Check antichain
                                if (c % a != 0 and a % c != 0 and 
                                    c % b != 0 and b % c != 0 and 
                                    b % a != 0 and a % b != 0):
                                    result.append([a, b, c])
                                    found = True
                                    break
                        if found: break
                    if not found:
                        # This is a catastrophic failure case
                        # Append a dummy to prevent crash, but it will be invalid
                        result.append([n]) 
                
    return result
