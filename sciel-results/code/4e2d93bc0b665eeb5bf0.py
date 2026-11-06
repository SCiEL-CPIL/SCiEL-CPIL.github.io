<<<<<<< SEARCH
def _is_3_colorable_mrv(adj_masks, n):
    """Fast 3-colorability check using iterative DP with bitmasks and memoization."""
    full_mask = (1 << n) - 1
    
    # Memoization: key = (uncolored_mask, cc0, cc1)
    # cc2 is implicitly full_mask ^ uncolored_mask ^ cc0 ^ cc1
    memo = {}
    
    def dp(uncolored_mask, cc0, cc1):
        if uncolored_mask == 0:
            return True
        
        key = (uncolored_mask, cc0, cc1)
        if key in memo:
            return memo[key]
        
        cc2 = full_mask ^ uncolored_mask ^ cc0 ^ cc1
        
        # Select vertex with fewest available colors (MRV heuristic)
        best_v = -1
        best_avail = 4
        m = uncolored_mask
        while m:
            v = (m & (-m)).bit_length() - 1
            m &= m - 1
            nb = adj_masks[v]
            avail = 3
            if nb & cc0:
                avail -= 1
            if nb & cc1:
                avail -= 1
            if nb & cc2:
                avail -= 1
            if avail == 0:
                memo[key] = False
                return False
            if avail < best_avail:
                best_avail = avail
                best_v = v
                if avail == 1:
                    break
        
        v = best_v
        nb = adj_masks[v]
        
        # Try each available color
        for c in range(3):
            if c == 0:
                if nb & cc0:
                    continue
                new_cc0 = cc0 | (1 << v)
                new_cc1 = cc1
            elif c == 1:
                if nb & cc1:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1 | (1 << v)
            else:
                if nb & cc2:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1
            
            if dp(uncolored_mask & ~(1 << v), new_cc0, new_cc1):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    return dp(full_mask, 0, 0)
=======
def _is_3_colorable_mrv(adj_masks, n):
    """Fast 3-colorability check using iterative DFS with bitmasks and memoization."""
    full_mask = (1 << n) - 1
    memo = {}
    
    # Stack-based DFS to avoid recursion overhead
    # Each stack entry: (uncolored_mask, cc0, cc1, next_color_to_try)
    # We use a state machine approach
    
    stack = [(full_mask, 0, 0, 0)]
    # path_memo: for each state we've visited, store whether it led to success
    # Actually, let's use a simpler approach: memoize states directly
    
    def check_state(uncolored_mask, cc0, cc1):
        """Check if a state is solvable. Returns True/False/None (undetermined)."""
        key = (uncolored_mask, cc0, cc1)
        if key in memo:
            return memo[key]
        if uncolored_mask == 0:
            return True
        
        cc2 = full_mask ^ uncolored_mask ^ cc0 ^ cc1
        
        # MRV: find vertex with fewest available colors
        best_v = -1
        best_avail = 4
        m = uncolored_mask
        while m:
            v = (m & (-m)).bit_length() - 1
            m &= m - 1
            nb = adj_masks[v]
            avail = 3
            if nb & cc0:
                avail -= 1
            if nb & cc1:
                avail -= 1
            if nb & cc2:
                avail -= 1
            if avail == 0:
                return False
            if avail < best_avail:
                best_avail = avail
                best_v = v
                if avail == 1:
                    break
        
        return (best_v, best_avail)
    
    # Iterative backtracking with explicit stack
    # Stack elements: (uncolored_mask, cc0, cc1, next_color_idx)
    # We also need to track the MRV vertex for each state
    
    stack = []
    # Start state
    uncolored = full_mask
    cc0 = 0
    cc1 = 0
    cc2 = full_mask
    
    # Compute MRV for start
    best_v = -1
    best_avail = 4
    m = uncolored
    while m:
        v = (m & (-m)).bit_length() - 1
        m &= m - 1
        nb = adj_masks[v]
        avail = 3
        if nb & cc0:
            avail -= 1
        if nb & cc1:
            avail -= 1
        if nb & cc2:
            avail -= 1
        if avail == 0:
            return False
        if avail < best_avail:
            best_avail = avail
            best_v = v
            if avail == 1:
                break
    
    stack.append((uncolored, cc0, cc1, best_v, 0))  # next_color_to_try = 0
    
    while stack:
        uncolored, cc0, cc1, v, next_c = stack[-1]
        
        if v == -1:
            # No uncolored vertices left -> success
            stack.pop()
            if not stack:
                return True
            # Backtrack: parent found a solution
            return True
        
        # Try colors from next_c onwards
        nb = adj_masks[v]
        found = False
        for c in range(next_c, 3):
            if c == 0:
                if nb & cc0:
                    continue
                new_cc0 = cc0 | (1 << v)
                new_cc1 = cc1
                new_cc2 = full_mask ^ (uncolored & ~(1 << v)) ^ new_cc0 ^ new_cc1
            elif c == 1:
                if nb & cc1:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1 | (1 << v)
                new_cc2 = full_mask ^ (uncolored & ~(1 << v)) ^ new_cc0 ^ new_cc1
            else:
                if nb & cc2:
                    continue
                new_cc0 = cc0
                new_cc1 = cc1
                new_cc2 = full_mask ^ (uncolored & ~(1 << v)) ^ new_cc0 ^ new_cc1
            
            new_uncolored = uncolored & ~(1 << v)
            
            if new_uncolored == 0:
                return True
            
            # Check memo
            key = (new_uncolored, new_cc0, new_cc1)
            if key in memo:
                if memo[key]:
                    return True
                continue
            
            # Compute MRV for new state
            new_best_v = -1
            new_best_avail = 4
            m = new_uncolored
            while m:
                vv = (m & (-m)).bit_length() - 1
                m &= m - 1
                nbb = adj_masks[vv]
                avail = 3
                if nbb & new_cc0:
                    avail -= 1
                if nbb & new_cc1:
                    avail -= 1
                if nbb & new_cc2:
                    avail -= 1
                if avail == 0:
                    memo[key] = False
                    continue
                if avail < new_best_avail:
                    new_best_avail = avail
                    new_best_v = vv
                    if avail == 1:
                        break
            
            if new_best_v == -1 and new_best_avail == 4:
                # All vertices have 0 available colors (shouldn't happen if avail>0 check passed)
                memo[key] = False
                continue
            
            # Push new state
            stack[-1] = (uncolored, cc0, cc1, v, c + 1)  # Update parent to try next color
            stack.append((new_uncolored, new_cc0, new_cc1, new_best_v, 0))
            found = True
            break
        
        if not found:
            # All colors exhausted for this vertex
            memo[(uncolored, cc0, cc1)] = False
            stack.pop()
            if not stack:
                return False
            # Backtrack: increment parent's next_color
            parent = stack[-1]
            stack[-1] = (parent[0], parent[1], parent[2], parent[3], parent[4] + 1)
    
    return False
>>>>>>> REPLACE

