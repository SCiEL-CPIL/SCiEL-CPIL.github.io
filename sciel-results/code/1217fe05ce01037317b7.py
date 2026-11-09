    def try_add_vertex_fast(mat, n):
        """
        Attempt to add a vertex to the graph defined by mat (size n).
        Returns a list of colors for the new vertex (length n) or None.
        Uses bitmasks for speed and random restarts.
        """
        # Precompute neighbor masks for each vertex and color
        neigh_mask = [[0]*4 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                c = mat[i][j]
                if c >= 0:
                    neigh_mask[i][c] |= (1 << j)
        
        assign = [-1] * n
        assigned_mask = [0]*4
        
        def backtrack_order(idx, order):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            for c in [0, 1, 2, 3]:
                if neigh_mask[i][c] & assigned_mask[c]:
                    continue
                
                assign[i] = c
                assigned_mask[c] |= (1 << i)
                
                if backtrack_order(idx + 1, order):
                    return True
                
                assigned_mask[c] &= ~(1 << i)
                assign[i] = -1
                
            return False

        # 1. Try degree-based order
        order_deg = sorted(range(n), key=lambda x: -sum(1 for j in range(n) if mat[x][j] >= 0))
        if backtrack_order(0, order_deg):
            return assign[:]

        # 2. Try random shuffles
        # Determine budget for restarts based on remaining time
        # If we have lots of time, try more.
        remaining = deadline - time.time()
        # Estimate cost of one restart? Hard to say.
        # Let's just try up to 50 restarts, guarded by deadline.
        for _ in range(50):
            if time.time() > deadline - 0.5:
                break
            order_rand = list(range(n))
            random.shuffle(order_rand)
            if backtrack_order(0, order_rand):
                return assign[:]
                
        return None
