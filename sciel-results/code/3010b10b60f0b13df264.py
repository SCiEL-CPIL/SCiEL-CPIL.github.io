    def solve(mask, current_size):
        nonlocal best_size, best_mask
        
        if time.time() > DEADLINE:
            return

        if current_size + bin(mask).count('1') <= best_size:
            return

        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = mask
            return

        # Quick Coloring Bound
        # We'll use a simple greedy coloring on the subgraph.
        # To make it fast, we don't sort.
        # This bound is valid.
        
        # If the mask is small, we can be more precise.
        # For now, let's just use the simple bound and hope the branching is good.
        # If not, I'll add the coloring bound in the next iteration.
        
        # Find max degree vertex
        max_deg = -1
        best_v = -1
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = bin(adj[v] & mask).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            return

        # Branch 1: Include
        new_mask_include = mask & ~((1 << best_v) | adj[best_v])
        solve(new_mask_include, current_size + 1)
        
        # Branch 2: Exclude
        new_mask_exclude = mask & ~(1 << best_v)
        solve(new_mask_exclude, current_size)
