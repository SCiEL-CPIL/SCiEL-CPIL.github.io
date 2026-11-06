import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Fallback: Seed program result (n=32)
    def seed_construct():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
    
    best_result = seed_construct()
    best_n = 32

    # Try to find a valid coloring for n=50
    target_n = 50
    
    # Helper functions for bitset operations
    def _compute_index(n, i, j):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def _build_adj(colors, n):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def _count_mono_triangles(adj, n):
        count = 0
        for c in range(4):
            for i in range(n):
                # Only consider neighbors j > i to avoid double counting? 
                # Actually, for delta calculation in SA, we just need the count.
                # For total count:
                # Sum over edges (i,j) with i<j of |N(i) cap N(j) cap {k > j}| ? No.
                # Standard way: 
                # For each color c, for each vertex i, 
                # count pairs (j,k) in N_c(i) such that j<k and (j,k) is an edge of color c.
                # This is equivalent to summing (1/2) * sum_i deg_c(i) * (deg_c(i)-1) is NOT correct because it counts non-edges.
                # Correct way:
                # For each color c, iterate i < j.
                # Count common neighbors k > j.
                # This counts each triangle exactly once.
                for i in range(n):
                    neigh_i = adj[c][i]
                    # Iterate over j in neigh_i such that j > i
                    j_mask = neigh_i & ~((1 << (i + 1)) - 1)
                    while j_mask:
                        j_bit = j_mask & (-j_mask)
                        j = j_bit.bit_length() - 1
                        j_mask ^= j_bit
                        # Common neighbors of i and j, restricted to k > j
                        common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                        count += common.bit_count()
        return count

    def _greedy_init(n):
        m = n * (n - 1) // 2
        colors = [0] * m
        adj = [[0] * n for _ in range(4)]
        
        for v in range(1, n):
            for u in range(v):
                best_c = 0
                min_cost = float('inf')
                for c in range(4):
                    # Cost: number of triangles formed if we color (u,v) with c
                    # A triangle (u, v, w) is formed if (u,w) and (v,w) are also color c.
                    # Since we are adding v, edges (u,w) for w < u are already fixed.
                    # Edges (v,w) for w < u are being added now? No, we are adding edge (u,v).
                    # The neighbors of u and v among {0..u-1} are known.
                    # If we color (u,v) with c, we form a triangle for every w in N_c(u) intersect N_c(v) where w < u.
                    # N_c(u) contains neighbors of u with color c.
                    # N_c(v) currently only contains neighbors of v with color c among {0..u-1} (since we're adding v, and u < v, we've processed all w < u for v? No.)
                    
                    # Let's re-evaluate greedy step.
                    # We are adding vertex v. We assign colors to edges (u, v) for u=0..v-1.
                    # When assigning color c to edge (u, v):
                    # We create a triangle (u, v, w) if edge (u, w) is color c AND edge (v, w) is color c.
                    # At this point, edges (v, w) for w < u have already been assigned.
                    # Edges (v, w) for w >= u and w < v are NOT yet assigned.
                    # So we only check triangles involving w < u.
                    
                    # Wait, the standard greedy is:
                    # For u from 0 to v-1:
                    #   For each color c:
                    #     count = |{ w < u : (u,w) is c AND (v,w) is c }|
                    #     This is | N_c(u) \cap N_c(v) \cap {0..u-1} |
                    
                    # N_c(u) is known.
                    # N_c(v) is partially known (for w < u).
                    
                    # So:
                    # cost = (adj[c][u] & adj[c][v] & ((1 << u) - 1)).bit_count()
                    
                    cost = (adj[c][u] & adj[c][v] & ((1 << u) - 1)).bit_count()
                    if cost < min_cost:
                        min_cost = cost
                        best_c = c
                        
                idx = _compute_index(n, u, v)
                colors[idx] = best_c
                adj[best_c][u] |= (1 << v)
                adj[best_c][v] |= (1 << u)
                
        return colors

    def _sa_search(n, time_budget):
        m = n * (n - 1) // 2
        start_time = time.time()
        
        # Precompute index-to-edge mapping for speed
        idx_to_edge = [None] * m
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_to_edge[idx] = (i, j)
                idx += 1
                
        best_tri = float('inf')
        best_colors = None
        
        # Try a few restarts
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() - start_time > time_budget:
                break
                
            colors = _greedy_init(n)
            adj = _build_adj(colors, n)
            tri_count = _count_mono_triangles(adj, n)
            
            if tri_count < best_tri:
                best_tri = tri_count
                best_colors = colors[:]
                
            if tri_count == 0:
                return colors
            
            T = 2.0
            T_min = 0.01
            cooling = 0.999
            steps_since_improve = 0
            max_steps_no_improve = 5000
            
            while True:
                if time.time() - start_time > time_budget:
                    break
                if steps_since_improve > max_steps_no_improve:
                    break
                    
                ri = random.randint(0, m - 1)
                old_c = colors[ri]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                u, v = idx_to_edge[ri]
                
                # Delta calculation
                # Triangles lost in old_c: common neighbors of u and v in old_c
                removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                # Triangles gained in new_c: common neighbors of u and v in new_c
                added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                delta = added - removed
                
                if delta <= 0:
                    # Accept
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += delta
                    steps_since_improve = 0
                else:
                    if T > T_min:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            colors[ri] = new_c
                            adj[old_c][u] &= ~(1 << v)
                            adj[old_c][v] &= ~(1 << u)
                            adj[new_c][u] |= (1 << v)
                            adj[new_c][v] |= (1 << u)
                            tri_count += delta
                            steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                        
                if tri_count < best_tri:
                    best_tri = tri_count
                    best_colors = colors[:]
                    if tri_count == 0:
                        return colors
                        
                T *= cooling
                
        return best_colors if best_tri == 0 else None

    # Main search loop
    # We have about 140 seconds.
    # Let's allocate time to try n=50.
    # If we find it, great.
    # If not, we might try to extend the seed to n=33, 34, etc.
    # But the biggest jump is 32 -> 50.
    
    # Time budget for n=50 search
    time_for_50 = 130.0
    t0 = time.time()
    
    res_50 = _sa_search(target_n, time_for_50)
    
    if res_50 is not None:
        # Verify quickly
        adj = _build_adj(res_50, target_n)
        if _count_mono_triangles(adj, target_n) == 0:
            return {'n': target_n, 'colors': res_50}
            
    # If we didn't find 50, can we do better than 32?
    # Try to extend the seed coloring to n=33, 34, ...
    # This is a different problem structure, but let's see if we have time.
    time_left = deadline - time.time()
    if time_left > 10:
        # Try to extend seed to n=33
        # This is tricky because the seed structure is specific.
        # A simpler approach: Run SA for n=33, 34, ... starting from random or greedy.
        # But SA for n=33 is easier than n=50.
        
        for n_try in range(33, 40):
            if time.time() > deadline - 5:
                break
            t_rem = deadline - time.time() - 2.0
            if t_rem < 5:
                break
            res = _sa_search(n_try, t_rem)
            if res is not None:
                adj = _build_adj(res, n_try)
                if _count_mono_triangles(adj, n_try) == 0:
                    if n_try > best_n:
                        best_n = n_try
                        best_result = {'n': n_try, 'colors': res}
                    break # If we found a valid one, we can stop or try higher? 
                         # Actually, if we found n=33, maybe we can find 34?
                         # But time is tight. Let's just return the best found.
                         # Since 50 is the main target, and we failed, 33 is a small improvement.
                         # However, the score is value/seed. 33/32 = 1.03. 50/32 = 1.56.
                         # So 50 is much better.
                         # If we can't get 50, 33 is better than 32.

    return best_result

