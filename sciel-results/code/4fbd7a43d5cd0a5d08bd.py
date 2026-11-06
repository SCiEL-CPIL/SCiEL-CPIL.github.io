def construct():
    import time
    import random
    
    n = 128
    start_time = time.time()
    timeout = 55.0  # Leave 5s buffer
    
    # Precompute all 4-term APs
    # AP is (a, a+d, a+2d, a+3d)
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            if a + 3 * d > n:
                break
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # Map position to list of AP indices containing it
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    def solve(num_colors):
        colors = [-1] * n
        # Heuristic: order positions by degree (number of APs) descending?
        # Or just 0 to n-1.
        # Let's try 0 to n-1 first.
        
        def is_consistent(pos, color):
            # Check all APs containing pos
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Collect colors of other positions
                other_colors = []
                unassigned_count = 0
                for p in ap:
                    if p == pos:
                        continue
                    if colors[p] == -1:
                        unassigned_count += 1
                    else:
                        other_colors.append(colors[p])
                
                # Case 1: All others assigned (unassigned_count == 0)
                if unassigned_count == 0:
                    all_cols = other_colors + [color]
                    if len(set(all_cols)) < 3:
                        return False
                
                # Case 2: One other unassigned
                elif unassigned_count == 1:
                    # The 3 assigned colors are other_colors + [color]
                    assigned_set = set(other_colors + [color])
                    if len(assigned_set) == 1:
                        # 3 same colors. Last one must make size >= 3. Impossible.
                        return False
                    elif len(assigned_set) == 2:
                        # 2 distinct colors assigned.
                        # The last one must be a 3rd color.
                        # This is a constraint on the future, not a direct failure here,
                        # UNLESS we are checking if this assignment is "safe" in a way that 
                        # we can detect impossibility later? 
                        # For backtracking, we just record that the last one is restricted.
                        # But we don't store constraints dynamically in this simple version.
                        # However, if we are just checking if THIS assignment is valid 
                        # given current state, it is valid IF there EXISTS a color for the last one.
                        # Since num_colors >= 3, there is always a 3rd color available.
                        # So this is OK.
                        pass
                    else:
                        # 3 distinct colors assigned. Last one can be anything.
                        pass
                
                # Case 3: Two others unassigned
                elif unassigned_count == 2:
                    assigned_set = set(other_colors + [color])
                    if len(assigned_set) == 1:
                        # 2 same colors assigned (pos and one other).
                        # The other two must be distinct and != this color.
                        # Is it possible? Yes, if num_colors >= 3.
                        # e.g. c, c, x, y with x!=y, x!=c, y!=c.
                        # So this is OK.
                        pass
                    elif len(assigned_set) == 2:
                        # 2 distinct colors assigned.
                        # The other two must not both be in this set.
                        # Possible? Yes.
                        pass
                    else:
                        # 3 distinct colors assigned.
                        # OK.
                        pass
                        
                # Case 4: Three others unassigned
                else:
                    # Only 1 color assigned (pos).
                    # Always possible to extend.
                    pass
            
            return True

        def backtrack(idx):
            if time.time() - start_time > timeout:
                return False, "timeout"
            
            if idx == n:
                return True, None
            
            pos = idx
            # Try colors
            # Randomize order?
            color_order = list(range(num_colors))
            random.shuffle(color_order)
            
            for c in color_order:
                if is_consistent(pos, c):
                    colors[pos] = c
                    success, reason = backtrack(idx + 1)
                    if success:
                        return True, None
                    if reason == "timeout":
                        return False, reason
                    colors[pos] = -1
            
            return False, None

        success, reason = backtrack(0)
        if success:
            return colors
        return None

    # Try 3 colors
    random.seed(42)
    sol = solve(3)
    if sol is not None:
        # Verify
        valid = True
        for ap in aps:
            if len(set(sol[p] for p in ap)) < 3:
                valid = False
                break
        if valid:
            return sol
    
    # Try 4 colors
    random.seed(43)
    sol = solve(4)
    if sol is not None:
        valid = True
        for ap in aps:
            if len(set(sol[p] for p in ap)) < 3:
                valid = False
                break
        if valid:
            return sol
    
    # Fallback: Return a simple coloring (might be invalid or use many colors)
    # Just return 0..127 to be safe? No, must be valid.
    # Use a greedy that might fail but is better than crashing?
    # The task requires a valid coloring.
    # If solve fails, we must ensure we return something valid.
    # Let's try a larger K or a different heuristic.
    
    # For now, if 3 and 4 fail, try 5.
    random.seed(44)
    sol = solve(5)
    if sol is not None:
        return sol
        
    # If all else fails, return a random permutation (uses 128 colors, definitely valid? No.)
    # A permutation of 0..127 ensures all 4 terms in any AP are distinct?
    # If colors are all distinct, then any 4 terms have 4 distinct colors.
    # So a permutation of 0..127 is always valid.
    # It uses 128 colors. Not minimal, but valid.
    # The evaluator minimizes the number of labels used.
    # So we want to minimize this.
    # But if we can't find a better one, 128 is a valid upper bound.
    return list(range(128))
