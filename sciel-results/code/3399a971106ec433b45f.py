import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Known valid 4-coloring of K_17 from literature/previous runs
    # This serves as the baseline and a warm-start for n=18
    def get_known_17():
        n = 17
        # Cyclic coloring distances 1..8
        # Assignment: [0, 1, 2, 3, 1, 0, 3, 2]
        dist_assign = [0, 1, 2, 3, 1, 0, 3, 2]
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                d_min = min(d, n - d)
                c = dist_assign[d_min - 1]
                colors.append(c)
        if verify_coloring_fast(n, colors):
            return {'n': n, 'colors': colors}
        # Fallback if verification fails (should not happen)
        n = 9
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = j - i
                d_min = min(d, n - d)
                colors.append((d_min - 1) % 4)
        return {'n': n, 'colors': colors}

    def verify_coloring_fast(n, colors):
        """Fast verification using bitset C6 detection."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        for c in range(4):
            adj = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if colors[idx] == c:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    idx += 1
            
            if has_c6_bitset(adj, n):
                return False
        return True

    def has_c6_bitset(adj, n):
        """Check if graph with bitmask adjacency contains a C6."""
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
                neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
                
                a_list = [i for i in range(n) if (neighbors_u >> i) & 1]
                d_list = [i for i in range(n) if (neighbors_v >> i) & 1]
                
                for a in a_list:
                    for d in d_list:
                        if a == d:
                            continue
                        neighbors_a = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        neighbors_d = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                        
                        b_list = [i for i in range(n) if (neighbors_a >> i) & 1]
                        for b in b_list:
                            if b == d:
                                continue
                            c_mask = neighbors_d & adj[b]
                            if c_mask:
                                return True
        return False

    def count_c6s(adj, n):
        """Count the total number of C6s in the graph defined by adj."""
        count = 0
        # For each edge (u,v), count paths of length 4 between u and v
        # that do not use u or v internally.
        # This counts each C6 exactly twice (once for each direction? No, C6 is undirected cycle).
        # A C6 contains 6 edges. If we iterate over edges and count paths, 
        # each C6 will be counted 6 times (once for each edge as the "chord" closing the cycle? No).
        # Let's be precise:
        # A cycle v0-v1-v2-v3-v4-v5-v0.
        # If we pick edge (v0, v1), we look for path v0-...-v1 of length 4.
        # The path is v0-v5-v4-v3-v2-v1. This is a valid path.
        # So for edge (v0,v1), this C6 contributes 1 to the count.
        # For edge (v1,v2), path v1-v0-v5-v4-v3-v2. Contributes 1.
        # ...
        # So each C6 is counted exactly 6 times (once for each edge in the cycle).
        # To get the number of distinct C6s, we divide by 6.
        
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                # We need paths u - a - b - c - d - v
                # a in N(u)\{v}, d in N(v)\{u}, b in N(a)\{u,v,d}, c in N(d)\{u,v,a}
                # and b-c edge.
                
                # Optimization: Use bitsets
                neighbors_u = adj[u] & ~(1 << v) & ~(1 << u)
                neighbors_v = adj[v] & ~(1 << u) & ~(1 << v)
                
                # Iterate a
                a_list = [i for i in range(n) if (neighbors_u >> i) & 1]
                d_list = [i for i in range(n) if (neighbors_v >> i) & 1]
                
                for a in a_list:
                    for d in d_list:
                        if a == d:
                            continue
                        # b in N(a) \ {u, v, d}
                        # c in N(d) \ {u, v, a}
                        # b-c edge, b != c
                        neighbors_a = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                        neighbors_d = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << d) & ~(1 << a)
                        
                        # b must be in neighbors_a
                        # c must be in neighbors_d
                        # and adj[b] & (1 << c) must be set
                        # and b != c (implied if we check intersection carefully? No, b and c are distinct vertices in the path)
                        
                        # Intersection of neighbors_a and neighbors_d gives candidate pairs?
                        # No, we need b in N(a) and c in N(d) and (b,c) in E.
                        # So for each b in N(a), check if there is c in N(d) such that (b,c) in E.
                        
                        b_list = [i for i in range(n) if (neighbors_a >> i) & 1]
                        for b in b_list:
                            # c must be in neighbors_d AND adjacent to b
                            c_mask = neighbors_d & adj[b]
                            # c cannot be b (simple graph, no self loops, so adj[b] doesn't have b)
                            # c must be distinct from a, d, u, v. 
                            # neighbors_d already excludes u, v, d, a.
                            # So any set bit in c_mask is a valid c.
                            count += c_mask.bit_count()
        
        return count // 6

    def anneal(n, initial_colors, time_limit, seed):
        """
        Simulated Annealing to minimize total C6s across 4 color classes.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_edges = len(edges)
        
        # State: colors array
        colors = initial_colors[:] if initial_colors else [random.randint(0, 3) for _ in range(num_edges)]
        
        # Precompute edge indices for fast access
        # adj[c] = list of bitmasks for color c
        adj = [[0] * n for _ in range(4)]
        
        def update_adj(idx, old_c, new_c):
            i, j = edges[idx]
            bit_i = 1 << i
            bit_j = 1 << j
            adj[old_c][i] &= ~bit_j
            adj[old_c][j] &= ~bit_i
            adj[new_c][i] |= bit_j
            adj[new_c][j] |= bit_i

        # Build initial adj
        for idx, c in enumerate(colors):
            i, j = edges[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            
        def total_energy():
            e = 0
            for c in range(4):
                e += count_c6s(adj[c], n)
            return e
            
        current_energy = total_energy()
        best_energy = current_energy
        best_colors = colors[:]
        
        T = 10.0
        cooling_rate = 0.9995
        steps = 0
        
        while time.time() < time_limit:
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            old_c = colors[idx]
            
            # Pick a new color different from old_c
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Apply move
            update_adj(idx, old_c, new_c)
            colors[idx] = new_c
            
            # Calculate delta energy
            # Instead of recalculating full energy, we could optimize, 
            # but count_c6s is O(n^3) or so. For n=18, it's fast enough.
            # Let's just recalculate for simplicity and correctness.
            new_energy = total_energy()
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    if best_energy == 0:
                        return best_colors
            else:
                # Revert
                update_adj(idx, new_c, old_c)
                colors[idx] = old_c
            
            T *= cooling_rate
            steps += 1
            
            # If T gets too small, reheat occasionally to escape local minima
            if T < 0.1 and steps % 10000 == 0:
                T = 1.0
                
        return best_colors

    # Strategy:
    # 1. Get known n=17
    # 2. Try to extend to n=18 using Simulated Annealing
    #    - Start with n=17 valid coloring + 1 random vertex
    #    - Or start with random n=18 coloring
    # 3. If n=18 fails, return n=17
    
    result_17 = get_known_17()
    if result_17['n'] < 17:
        return result_17
        
    # Try n=18
    n_target = 18
    time_for_18 = min(60, deadline - time.time() - 5)
    
    if time_for_18 > 10:
        # Attempt 1: Warm start from n=17
        # Add vertex 17. Connect to 0..16.
        # We can try to find colors for these 17 edges that don't create C6s.
        # This is a sub-problem. But SA on full n=18 is more robust.
        
        # Let's run SA on n=18 with a few different seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
            
            # Random initialization
            init_colors = None # Let SA generate random
            best_colors = anneal(n_target, init_colors, time.time() + min(15, remaining), seed)
            
            if best_colors is not None:
                if verify_coloring_fast(n_target, best_colors):
                    return {'n': n_target, 'colors': best_colors}
        
        # If random init failed, try warm start
        # Construct a partial coloring for n=18 based on n=17
        # Take n=17 colors, add edges for vertex 17.
        # This is complex to set up correctly for SA state, so we'll stick to random or simple heuristics.
        
    # Return n=17
    return result_17
