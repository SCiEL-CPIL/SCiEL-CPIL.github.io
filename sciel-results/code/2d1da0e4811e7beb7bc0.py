import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 35s safety margin
    
    def get_idx(n, i, j):
        """Get index for edge (i,j) where i<j in lexicographic order."""
        return i * (n - 1) - i * (i - 1) // 2 + j - i - 1

    def has_c6_fast(adj, n):
        """
        Checks if graph with adjacency bitsets contains a 6-cycle.
        Returns True if found, False otherwise.
        Uses early termination for speed.
        """
        for v0 in range(n):
            mask_v0 = ~(1 << v0)
            neigh_v0 = adj[v0]
            while neigh_v0:
                v1 = (neigh_v0 & -neigh_v0).bit_length() - 1
                neigh_v0 &= neigh_v0 - 1
                
                # v1 must be > v0 to avoid duplicates in undirected graph
                if v1 <= v0:
                    continue
                    
                mask_excl_v0 = mask_v0
                neigh_v1 = adj[v1] & mask_excl_v0
                
                while neigh_v1:
                    v2 = (neigh_v1 & -neigh_v1).bit_length() - 1
                    neigh_v1 &= neigh_v1 - 1
                    
                    if v2 == v0:
                        continue
                        
                    mask_excl_v0_v1 = mask_excl_v0 & ~(1 << v1)
                    neigh_v2 = adj[v2] & mask_excl_v0_v1
                    
                    while neigh_v2:
                        v3 = (neigh_v2 & -neigh_v2).bit_length() - 1
                        neigh_v2 &= neigh_v2 - 1
                        
                        if v3 in (v0, v1):
                            continue
                        
                        mask_excl_v0_v1_v2 = mask_excl_v0_v1 & ~(1 << v2)
                        neigh_v3 = adj[v3] & mask_excl_v0_v1_v2
                        
                        while neigh_v3:
                            v4 = (neigh_v3 & -neigh_v3).bit_length() - 1
                            neigh_v3 &= neigh_v3 - 1
                            
                            if v4 in (v0, v1, v2):
                                continue
                            
                            mask_excl_v0_v1_v2_v3 = mask_excl_v0_v1_v2 & ~(1 << v3)
                            neigh_v4 = adj[v4] & mask_excl_v0_v1_v2_v3
                            
                            while neigh_v4:
                                v5 = (neigh_v4 & -neigh_v4).bit_length() - 1
                                neigh_v4 &= neigh_v4 - 1
                                
                                if v5 in (v0, v1, v2, v3):
                                    continue
                                
                                if (adj[v5] >> v0) & 1:
                                    return True
        return False

    def count_c6s(adj, n):
        """Count approximate number of C6s for cost function."""
        count = 0
        for v0 in range(n):
            mask_v0 = ~(1 << v0)
            neigh_v0 = adj[v0]
            while neigh_v0:
                v1 = (neigh_v0 & -neigh_v0).bit_length() - 1
                neigh_v0 &= neigh_v0 - 1
                if v1 <= v0:
                    continue
                mask_excl_v0 = mask_v0
                neigh_v1 = adj[v1] & mask_excl_v0
                while neigh_v1:
                    v2 = (neigh_v1 & -neigh_v1).bit_length() - 1
                    neigh_v1 &= neigh_v1 - 1
                    if v2 == v0:
                        continue
                    mask_excl_v0_v1 = mask_excl_v0 & ~(1 << v1)
                    neigh_v2 = adj[v2] & mask_excl_v0_v1
                    while neigh_v2:
                        v3 = (neigh_v2 & -neigh_v2).bit_length() - 1
                        neigh_v2 &= neigh_v2 - 1
                        if v3 in (v0, v1):
                            continue
                        mask_excl_v0_v1_v2 = mask_excl_v0_v1 & ~(1 << v2)
                        neigh_v3 = adj[v3] & mask_excl_v0_v1_v2
                        while neigh_v3:
                            v4 = (neigh_v3 & -neigh_v3).bit_length() - 1
                            neigh_v3 &= neigh_v3 - 1
                            if v4 in (v0, v1, v2):
                                continue
                            mask_excl_v0_v1_v2_v3 = mask_excl_v0_v1_v2 & ~(1 << v3)
                            neigh_v4 = adj[v4] & mask_excl_v0_v1_v2_v3
                            while neigh_v4:
                                v5 = (neigh_v4 & -neigh_v4).bit_length() - 1
                                neigh_v4 &= neigh_v4 - 1
                                if v5 in (v0, v1, v2, v3):
                                    continue
                                if (adj[v5] >> v0) & 1:
                                    count += 1
        return count

    def try_find_coloring(n, max_time, initial_colors=None):
        """
        Simulated annealing to find valid 4-coloring of K_n.
        Returns colors list or None.
        """
        if time.time() > max_time:
            return None
            
        num_edges = n * (n - 1) // 2
        
        if initial_colors is None:
            # Use a structured initial coloring to start closer to valid
            colors = [random.randint(0, 3) for _ in range(num_edges)]
        else:
            colors = initial_colors[:]
        
        # Build adjacency bitsets
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(n, i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        def update_adj(i, j, old_c, new_c):
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)

        # Simulated annealing parameters
        T0 = 1.0
        T_min = 0.001
        cooling = 0.995
        T = T0
        
        best_colors = colors[:]
        best_cost = float('inf')
        
        # Initial cost
        initial_cost = 0
        for c in range(4):
            if has_c6_fast(adj[c], n):
                initial_cost += 1
        best_cost = initial_cost
        
        while time.time() < max_time:
            if T < T_min:
                T = T0  # Restart with high temperature
            
            # Pick a random edge to recolor
            idx = random.randint(0, num_edges - 1)
            # Find (i,j) from idx
            # Inverse of get_idx
            # For given idx, find i such that sum_{k=0}^{i-1}(n-1-k) <= idx < sum_{k=0}^{i}(n-1-k)
            i = 0
            cum = 0
            while i < n - 1:
                count_i = n - 1 - i
                if cum + count_i > idx:
                    break
                cum += count_i
                i += 1
            j = i + 1 + (idx - cum)
            
            old_c = colors[idx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Apply move
            colors[idx] = new_c
            update_adj(i, j, old_c, new_c)
            
            # Compute new cost
            new_cost = 0
            for c in range(4):
                if has_c6_fast(adj[c], n):
                    new_cost += 1
            
            delta = new_cost - best_cost if best_cost == initial_cost else new_cost - (0 if best_cost == 0 else 1)
            # Actually, let's track current cost properly
            # Recompute current cost for comparison
            cur_cost = 0
            for c in range(4):
                if has_c6_fast(adj[c], n):
                    cur_cost += 1
            
            delta = new_cost - cur_cost if False else 0  # This is getting complex
            # Simplify: just use new_cost vs 0 (valid) or accept with probability
            # Better: track current cost explicitly
            break  # Let me restructure
            
        # Restructure: simpler approach
        # Let me just do a clean implementation
        
        return None  # Placeholder, will fix

    # Let me rewrite try_find_coloring properly
    def try_find_coloring_v2(n, max_time, initial_colors=None):
        if time.time() > max_time:
            return None
            
        num_edges = n * (n - 1) // 2
        
        if initial_colors is None:
            colors = [random.randint(0, 3) for _ in range(num_edges)]
        else:
            colors = initial_colors[:]
        
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(n, i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

        def update_adj(i, j, old_c, new_c):
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)

        def compute_cost():
            cost = 0
            for c in range(4):
                if has_c6_fast(adj[c], n):
                    cost += 1
            return cost

        cur_cost = compute_cost()
        best_colors = colors[:]
        best_cost = cur_cost
        
        T = 1.0
        cooling = 0.997
        T_min = 0.001
        
        while time.time() < max_time:
            if T < T_min:
                T = 1.0  # Restart annealing
            
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            # Inverse index
            i = 0
            cum = 0
            while i < n - 1:
                count_i = n - 1 - i
                if cum + count_i > idx:
                    break
                cum += count_i
                i += 1
            j = i + 1 + (idx - cum)
            
            old_c = colors[idx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Apply
            colors[idx] = new_c
            update_adj(i, j, old_c, new_c)
            
            new_cost = compute_cost()
            delta = new_cost - cur_cost
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                # Revert
                colors[idx] = old_c
                update_adj(i, j, new_c, old_c)
            
            T *= cooling
        
        if best_cost == 0:
            return best_colors
        return None

    # Try n=17 first
    max_time_for_17 = 80
    result_17 = try_find_coloring_v2(17, start_time + max_time_for_17)
    if result_17:
        # Verify
        n = 17
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(n, i, j)
                c = result_17[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        valid = True
        for c in range(4):
            if has_c6_fast(adj[c], n):
                valid = False
                break
        
        if valid:
            return {'n': 17, 'colors': result_17}

    # Try n=18 if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        if remaining > 15:
            result_18 = try_find_coloring_v2(18, deadline)
            if result_18:
                n = 18
                adj = [[0] * n for _ in range(4)]
                for i in range(n):
                    for j in range(i + 1, n):
                        idx = get_idx(n, i, j)
                        c = result_18[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                
                valid = True
                for c in range(4):
                    if has_c6_fast(adj[c], n):
                        valid = False
                        break
                
                if valid:
                    return {'n': 18, 'colors': result_18}

    # Fallback: try to find any valid coloring for smaller n
    for n_try in [16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 5:
            break
        result = try_find_coloring_v2(n_try, deadline)
        if result:
            adj = [[0] * n_try for _ in range(4)]
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    idx = get_idx(n_try, i, j)
                    c = result[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            
            valid = True
            for c in range(4):
                if has_c6_fast(adj[c], n_try):
                    valid = False
                    break
            
            if valid:
                return {'n': n_try, 'colors': result}

    # Final fallback: trivial valid coloring for small n
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}
