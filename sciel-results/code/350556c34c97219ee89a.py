def construct():
    import random
    from math import gcd
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
            
    # Precompute bad_triple_mask[a][b]
    # bad_triple_mask[a][b] is a bitmask where bit k (0-indexed, representing k+1) is 1
    # if {a, b, k+1} is a bad triple.
    bad_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            mask = 0
            for w in range(1, N + 1):
                if w == a or w == b:
                    continue
                if gcd_mat[a][w] == g_ab and gcd_mat[b][w] == g_ab:
                    mask |= (1 << (w - 1))
            bad_mask[a][b] = mask
            bad_mask[b][a] = mask

    def greedy(order):
        selected = []
        # forbidden[i] counts how many pairs in 'selected' forbid i
        forbidden = [0] * (N + 1)
        
        for x in order:
            if forbidden[x] == 0:
                # Add x
                selected.append(x)
                # Update forbidden counts for all y
                for a in selected[:-1]: # All previously selected
                    m = bad_mask[x][a]
                    # Iterate through set bits in m
                    mm = m
                    while mm:
                        # Get lowest set bit
                        lsb = mm & -mm
                        idx = lsb.bit_length() # 1-indexed position
                        y = idx
                        forbidden[y] += 1
                        mm ^= lsb
        
        return selected

    def local_improve(selected):
        # Returns improved selected list
        # We use the forbidden array state
        forbidden = [0] * (N + 1)
        sel_set = set(selected)
        
        # Initialize forbidden array for current selected
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                a, b = selected[i], selected[j]
                m = bad_mask[a][b]
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length()
                    y = idx
                    if y not in sel_set: # Only care about unselected? 
                        # Actually, we need forbidden for ALL y to check addability.
                        # But if y is in sel_set, it doesn't matter if it's "forbidden" 
                        # because we won't add it. 
                        # However, for swap logic, we need accurate counts.
                        forbidden[y] += 1
                    mm ^= lsb

        improved = selected[:]
        sel_set = set(improved)
        improved_flag = True
        
        while improved_flag:
            improved_flag = False
            
            # 1. Try to add any x not in sel_set with forbidden[x] == 0
            added = False
            for x in range(1, N + 1):
                if x not in sel_set and forbidden[x] == 0:
                    # Add x
                    improved.append(x)
                    sel_set.add(x)
                    # Update forbidden
                    for a in improved[:-1]:
                        m = bad_mask[x][a]
                        mm = m
                        while mm:
                            lsb = mm & -mm
                            idx = lsb.bit_length()
                            y = idx
                            if y not in sel_set:
                                forbidden[y] += 1
                            mm ^= lsb
                    added = True
                    improved_flag = True
                    break
            
            if added:
                continue

            # 2. Try 1-for-1 swap
            # For each y in improved, remove it, check if any x can be added
            # We need to calculate the "relaxed" forbidden array if y is removed.
            
            best_swap = None
            best_gain = -1 # We want to swap if it allows adding, size stays same, 
                           # but maybe allows further additions? 
                           # Actually, if we can add x after removing y, size is same.
                           # We should only swap if it leads to a state where we CAN add later?
                           # Or just swap to break deadlock?
                           # A simple hill climb on size won't work with 1-for-1 swaps if size doesn't change.
                           # We need to look ahead or use simulated annealing.
                           # Let's try: Remove y, Add x. If valid, do it. 
                           # Then try to add more.
                           
            for y in improved:
                # Calculate forbidden if y is removed
                # Pairs involving y: (y, a) for a in improved, a != y
                # These pairs contributed to forbidden.
                # We need to subtract their contribution.
                
                # To avoid O(N^2) per y, we can just recompute or use a temporary array.
                # Given N=128, O(N^2) is 16k. 128 * 16k = 2M ops. Fine.
                
                temp_forbidden = forbidden[:]
                # Subtract contributions of pairs (y, a)
                for a in improved:
                    if a == y:
                        continue
                    m = bad_mask[y][a]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length()
                        z = idx
                        temp_forbidden[z] -= 1
                        mm ^= lsb
                
                # Now check if any x not in improved (and x != y) has temp_forbidden[x] == 0
                # Note: y is still in 'improved' list but we are simulating its removal.
                # x must not be in improved \ {y}.
                
                for x in range(1, N + 1):
                    if x == y:
                        continue
                    if x in sel_set:
                        continue
                    if temp_forbidden[x] == 0:
                        # Valid swap: remove y, add x
                        # Apply change
                        # 1. Remove y from improved
                        improved.remove(y)
                        sel_set.remove(y)
                        
                        # Update forbidden: 
                        # First, apply the subtraction for removing y (we already calculated temp_forbidden)
                        # But we need to update the main 'forbidden' array to match the new state.
                        # New state has pairs (x, a) for a in new_improved.
                        # Old state had pairs (y, a) and (a, b).
                        # 
                        # Let's just recompute forbidden from scratch for the new set?
                        # O(|S|^2 * N). 128^2 * 128 = 2M. 
                        # If we do this for every swap, it might be slow if many swaps.
                        # But swaps should be rare if we are near optimal.
                        
                        # Let's do the incremental update properly.
                        # Current 'forbidden' corresponds to 'improved' (with y).
                        # We want 'forbidden' to correspond to 'improved' - {y} + {x}.
                        
                        # Step 1: Remove y.
                        # Subtract contributions of (y, a) for all a in improved (excluding y).
                        for a in improved: # 'improved' still contains y? No, we removed it above?
                             # Wait, I did improved.remove(y) before this block?
                             # Let's restructure.
                             pass
                        
                        # Easier: Recompute forbidden for the new set.
                        # It's safe and fast enough for N=128.
                        new_selected = improved[:] # After removing y
                        new_selected.append(x)
                        sel_set.add(x)
                        
                        # Recompute forbidden
                        forbidden = [0] * (N + 1)
                        ns = new_selected
                        for i in range(len(ns)):
                            for j in range(i + 1, len(ns)):
                                a, b = ns[i], ns[j]
                                m = bad_mask[a][b]
                                mm = m
                                while mm:
                                    lsb = mm & -mm
                                    idx = lsb.bit_length()
                                    z = idx
                                    if z not in sel_set:
                                        forbidden[z] += 1
                                    mm ^= lsb
                        
                        improved_flag = True
                        best_swap = (y, x)
                        break
                if best_swap:
                    break
            
        return improved

    # Generate initial solutions
    best = []
    
    # Strategies
    strategies = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: x.bit_count()),
        sorted(range(1, N + 1), key=lambda x: x.bit_count(), reverse=True),
    ]
    
    # Add prime factor count strategy
    def num_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    strategies.append(sorted(range(1, N + 1), key=num_prime_factors))
    strategies.append(sorted(range(1, N + 1), key=num_prime_factors, reverse=True))

    for strat in strategies:
        r = greedy(strat)
        r = local_improve(r)
        if len(r) > len(best):
            best = r

    # Random restarts
    random.seed(42)
    for _ in range(2000):
        shuffled = nums[:]
        random.shuffle(shuffled)
        r = greedy(shuffled)
        r = local_improve(r)
        if len(r) > len(best):
            best = r

    return best
