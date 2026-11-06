def construct():
    from itertools import combinations
    from math import gcd
    import random
    import time
    
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
            
    # Precompute forbidden triples and conflict structures
    # conflict_pairs_for_x[x] = set of pairs (u, v) with u < v such that {u, v, x} is forbidden
    conflict_pairs_for_x = [set() for _ in range(n + 1)]
    
    # We also need to know, for a pair (u, v), which x are forbidden.
    # This is useful for local search: if we remove x, we might be able to add y.
    # But the main check is: given S, can we add x?
    # Check: Are there any pairs (u, v) in S such that (u, v) in conflict_pairs_for_x[x]?
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    # Triple (a, b, c) is forbidden
                    # For x=a, pairs (b,c) conflict
                    # For x=b, pairs (a,c) conflict
                    # For x=c, pairs (a,b) conflict
                    
                    pair_bc = (b, c)
                    pair_ac = (a, c)
                    pair_ab = (a, b)
                    
                    conflict_pairs_for_x[a].add(pair_bc)
                    conflict_pairs_for_x[b].add(pair_ac)
                    conflict_pairs_for_x[c].add(pair_ab)

    def is_valid_addition(selected_pairs, x):
        """
        Check if x can be added given the set of pairs currently in the selection.
        selected_pairs: set of tuples (u, v) with u < v
        x: integer
        """
        conflicts = conflict_pairs_for_x[x]
        # If any pair in selected_pairs is in conflicts, we cannot add x
        # Optimization: iterate over the smaller set
        if len(conflicts) < len(selected_pairs):
            for p in conflicts:
                if p in selected_pairs:
                    return False
        else:
            for p in selected_pairs:
                if p in conflicts:
                    return False
        return True

    def greedy_construct(order):
        selected = []
        selected_pairs = set()
        for x in order:
            if is_valid_addition(selected_pairs, x):
                # Add x
                selected.append(x)
                # Update pairs
                for u in selected[:-1]:
                    pair = (min(u, x), max(u, x))
                    selected_pairs.add(pair)
        return selected, selected_pairs

    def local_search(selected, selected_pairs):
        """
        Try to improve the set by swapping elements.
        Returns improved set and pairs.
        """
        improved = True
        while improved:
            improved = False
            for i in range(len(selected)):
                x = selected[i]
                # Remove x
                new_selected = selected[:i] + selected[i+1:]
                new_pairs = set()
                # Rebuild pairs is O(N^2), but N is small.
                # Or we can just remove pairs involving x.
                # Let's just rebuild for simplicity and correctness, N<=128.
                for a_idx in range(len(new_selected)):
                    for b_idx in range(a_idx + 1, len(new_selected)):
                        u = new_selected[a_idx]
                        v = new_selected[b_idx]
                        new_pairs.add((min(u, v), max(u, v)))
                
                # Try to add any y not in new_selected
                for y in nums:
                    if y not in new_selected:
                        if is_valid_addition(new_pairs, y):
                            # Check if we actually gain size (we removed 1, added 1, so size is same)
                            # Wait, the goal is to MAXIMIZE cardinality.
                            # A 1-swap keeps size same. We need 1-remove, 2-add or 2-remove, 3-add etc.
                            # Or, we can try to add y WITHOUT removing x first?
                            # If we can add y to current set, we do it.
                            pass
                
                # Actually, let's do a simpler local search:
                # 1. Try to add any y not in selected.
                # 2. If no y can be added, try to remove x and add y, z.
                
                # Let's restructure local search.
                break
        
        # The above loop was messy. Let's implement a cleaner one.
        return selected, selected_pairs

    # Better Local Search
    def improve(selected, selected_pairs):
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any element
            for y in nums:
                if y not in selected:
                    if is_valid_addition(selected_pairs, y):
                        selected.append(y)
                        for u in selected[:-1]:
                            pair = (min(u, y), max(u, y))
                            selected_pairs.add(pair)
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try 1-swap: remove x, add y (if y is valid in S\{x})
            # This doesn't increase size, but might allow future additions?
            # Actually, if we can't add any y to S, maybe removing x allows adding y and z?
            # Let's try 2-add-1-remove.
            
            # For each x in selected:
            for i in range(len(selected)):
                x = selected[i]
                # Temporarily remove x
                temp_selected = selected[:i] + selected[i+1:]
                temp_pairs = set()
                for a_idx in range(len(temp_selected)):
                    for b_idx in range(a_idx + 1, len(temp_selected)):
                        u = temp_selected[a_idx]
                        v = temp_selected[b_idx]
                        temp_pairs.add((min(u, v), max(u, v)))
                
                # Try to add two elements y, z
                # To keep it fast, iterate y, then z
                # Since N=128, pairs (y,z) is ~8000.
                # For each pair, check validity.
                # This might be slow if done for every x.
                
                # Heuristic: only try y, z that are "compatible" with most of temp_selected
                # Just try all y
                added = False
                for y in nums:
                    if y in temp_selected:
                        continue
                    if not is_valid_addition(temp_pairs, y):
                        continue
                    
                    # y is valid. Now try to add z to temp_selected + y
                    # Update temp_pairs with y
                    temp_pairs_with_y = set(temp_pairs)
                    for u in temp_selected:
                        temp_pairs_with_y.add((min(u, y), max(u, y)))
                    
                    for z in nums:
                        if z in temp_selected or z == y:
                            continue
                        if is_valid_addition(temp_pairs_with_y, z):
                            # We can add y and z
                            # New set is temp_selected + [y, z]
                            # Size increases by 1
                            new_selected = temp_selected + [y, z]
                            new_pairs = set(temp_pairs_with_y)
                            # Add pairs for z
                            for u in temp_selected:
                                new_pairs.add((min(u, z), max(u, z)))
                            new_pairs.add((min(y, z), max(y, z)))
                            
                            selected = new_selected
                            selected_pairs = new_pairs
                            improved = True
                            added = True
                            break
                    if added:
                        break
                if added:
                    break
        return selected, selected_pairs

    # Main Loop
    best_selected = []
    best_pairs = set()
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s margin
    
    # Initial Greedy
    # Order by number of conflicts (fewest conflicts first)
    conflict_counts = [len(conflict_pairs_for_x[i]) for i in range(n + 1)]
    initial_order = sorted(nums, key=lambda x: conflict_counts[x])
    best_selected, best_pairs = greedy_construct(initial_order)
    
    # Local Search
    best_selected, best_pairs = improve(best_selected, best_pairs)
    
    # Random Restarts
    while time.time() - start_time < max_time:
        # Random ordering
        random_order = nums[:]
        random.shuffle(random_order)
        
        # Greedy
        curr_selected, curr_pairs = greedy_construct(random_order)
        
        # Local Search
        curr_selected, curr_pairs = improve(curr_selected, curr_pairs)
        
        if len(curr_selected) > len(best_selected):
            best_selected = curr_selected
            best_pairs = curr_pairs
    
    return best_selected
