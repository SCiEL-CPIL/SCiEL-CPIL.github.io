import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: n=1 is always valid (no edges)
    best_n = 1
    best_colors = []

    # Known algebraic construction for n=26 (Lazebnik-Woldar style)
    # We'll try to build it directly.
    # For n=26, we can use the 5-coloring based on the 25-element affine plane + 1 point.
    # Actually, a simpler approach: use the known n=25 construction and try to extend.
    # But let's just use the known result: there exists a 5-coloring of K_26 with no mono C4.
    # We'll construct it using the method: vertices are Z_5 x Z_5 plus infinity.
    # Actually, the standard construction is:
    # Vertices: {0,1,...,24} corresponding to Z_5^2, and 25 = infinity.
    # For each pair, assign color based on... this is complex.
    # Let's instead use a well-known explicit construction if available, 
    # or rely on the optimizer to find n>=15 which is already an improvement.
    
    # Strategy:
    # 1. Try to find a valid coloring for n=12..26 using local search.
    # 2. The key insight from the knowledge: incremental O(n) energy update enables ~25x more steps.
    # 3. We use a vectorized approach with NumPy for speed.
    
    def try_local_search(n, max_time_per_n=20):
        """Try to find a valid 5-coloring of K_n using local search."""
        time_start = time.time()
        max_iter = 1000000  # large, will be limited by time
        
        num_edges = n * (n - 1) // 2
        
        # Precompute edge index
        edge_index = np.zeros((n, n), dtype=np.int32)
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[i, j] = idx
                edge_index[j, i] = idx
                endpoints.append((i, j))
                idx += 1
        
        # Precompute edge_index as a flat array for faster access
        # For each edge e = (u,v), we want to quickly find all edges (u,x) and (v,x) for x != u,v
        # We'll precompute for each edge e, the list of edges sharing u and edges sharing v.
        # But this is memory intensive. Instead, we'll use the edge_index 2D array.
        
        for restart in range(20):
            if time.time() > deadline - 5:
                return None
            if time.time() - time_start > max_time_per_n:
                break
            
            colors = np.random.randint(0, 5, size=num_edges, dtype=np.int8)
            
            # Build CN matrices
            CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
            
            # Build neighbor sets for each color
            # neighbors[c][u] = list of v such that (u,v) has color c
            neighbors = [[set() for _ in range(n)] for _ in range(5)]
            
            for e in range(num_edges):
                u, v = endpoints[e]
                c = int(colors[e])
                neighbors[c][u].add(v)
                neighbors[c][v].add(u)
            
            # Compute initial CN
            for c in range(5):
                for u in range(n):
                    nbrs = list(neighbors[c][u])
                    L = len(nbrs)
                    for i in range(L):
                        for j in range(i+1, L):
                            v1 = nbrs[i]
                            v2 = nbrs[j]
                            CN[c][v1, v2] += 1
                            CN[c][v2, v1] += 1
            
            # Initial cost: number of mono C4s
            # A C4 in color c is a pair (a,b) with CN[c][a,b] >= 2
            # Actually, each pair of common neighbors forms a C4 if the 4 vertices are distinct.
            # For a fixed pair (u,v) with CN[c][u,v] = k, there are C(k,2) C4s containing edge... wait.
            # No: a C4 is four vertices a,b,c,d with edges (a,b), (b,c), (c,d), (d,a) all in color c.
            # This corresponds to two disjoint edges (a,c) and (b,d)? No.
            # Actually, a C4 in color c means: vertices u,v,w,x such that (u,v), (v,w), (w,x), (x,u) all in color c.
            # This is equivalent to: u and w have at least 2 common neighbors in color c (v and x), 
            # AND v and x are not adjacent in color c (otherwise it's a triangle).
            # Wait, the task says "no two vertices with two common neighbours in one colour".
            # So the condition is simply: for every color c and every pair (u,w), CN[c][u,w] <= 1.
            # This is exactly the condition. A pair (u,w) with CN[c][u,w] >= 2 forms a C4 (u,v,w,x)
            # where v,x are two common neighbors. The edges (u,v), (v,w), (w,x), (x,u) are all in color c.
            # So yes, the cost is the number of pairs (u,w) with CN[c][u,w] >= 2.
            
            def compute_cost(CN):
                cost = 0
                for c in range(5):
                    tri_u, tri_v = np.triu_indices(n, 1)
                    vals = CN[c][tri_u, tri_v]
                    # For each pair, if CN >= 2, add C(CN, 2) C4s
                    # But actually, each pair with CN = k contributes C(k,2) to the number of C4s
                    # through that pair. But each C4 is counted twice (once for each diagonal pair).
                    # Wait, a C4 has two diagonals. So if we sum C(CN[u,w], 2) over all pairs (u,w),
                    # each C4 is counted twice.
                    # But the task only requires that NO pair has CN >= 2.
                    # So the cost we care about is: number of pairs (u,w) with CN[c][u,w] >= 2.
                    # Let's define cost = sum over c of sum over pairs (u,w) of [CN[c][u,w] >= 2].
                    cost += int(np.sum((vals >= 2).astype(np.int64)))
                return cost
            
            current_cost = compute_cost(CN)
            
            if current_cost == 0:
                return colors.tolist()
            
            # Local search: pick a random edge, try to change its color to reduce cost.
            # For speed, we'll use a simulated annealing-like approach with occasional random restarts.
            
            best_cost = current_cost
            best_colors = colors.copy()
            
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    return None
                if time.time() - time_start > max_time_per_n:
                    break
                
                if current_cost == 0:
                    return colors.tolist()
                
                # Pick a random edge
                e_idx = np.random.randint(num_edges)
                u, v = endpoints[e_idx]
                old_c = int(colors[e_idx])
                
                # Try all 4 other colors
                best_delta = 0
                best_new_c = old_c
                
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    
                    delta = 0
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        
                        c_vx = int(colors[edge_index[v, x]])
                        c_ux = int(colors[edge_index[u, x]])
                        
                        # When we change (u,v) from old_c to new_c:
                        # For old_c: 
                        #   If (v,x) is in old_c, then u and x had a common neighbor v in old_c.
                        #   After change, u and x no longer have v as common neighbor in old_c.
                        #   Similarly if (u,x) is in old_c, then v and x lose u as common neighbor in old_c.
                        #   So CN[old_c][u,x] decreases by 1 if (v,x) in old_c.
                        #   CN[old_c][v,x] decreases by 1 if (u,x) in old_c.
                        #   The cost change for pair (u,x) in old_c: 
                        #     old_CN = CN[old_c][u,x]
                        #     new_CN = old_CN - (1 if (v,x) in old_c else 0)
                        #     Cost contribution: [old_CN >= 2] - [new_CN >= 2]
                        #     This is -1 if old_CN == 2 and (v,x) in old_c, else 0.
                        #     More generally: [old_CN >= 2] - [new_CN >= 2] = 
                        #       1 if old_CN >= 2 and new_CN < 2, else 0
                        #       = 1 if old_CN == 2 and (v,x) in old_c, else 0
                        #   Similarly for pair (v,x) in old_c.
                        #   For new_c:
                        #     CN[new_c][u,x] increases by 1 if (v,x) in new_c.
                        #     Cost change: [new_CN >= 2] - [old_CN >= 2] = 
                        #       1 if old_CN == 1 and (v,x) in new_c, else 0
                        #     Similarly for pair (v,x) in new_c.
                        
                        if c_vx == old_c:
                            # Pair (u,x) in old_c loses common neighbor v
                            old_cn = int(CN[old_c][u, x])
                            if old_cn == 2:
                                delta -= 1  # cost decreases
                        if c_ux == old_c:
                            # Pair (v,x) in old_c loses common neighbor u
                            old_cn = int(CN[old_c][v, x])
                            if old_cn == 2:
                                delta -= 1
                        if c_vx == new_c:
                            # Pair (u,x) in new_c gains common neighbor v
                            old_cn = int(CN[new_c][u, x])
                            if old_cn == 1:
                                delta += 1  # cost increases
                        if c_ux == new_c:
                            # Pair (v,x) in new_c gains common neighbor u
                            old_cn = int(CN[new_c][v, x])
                            if old_cn == 1:
                                delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                
                if best_delta >= 0:
                    # No improvement, try a random change with small probability
                    if random.random() < 0.1:
                        new_c = random.choice([c for c in range(5) if c != old_c])
                        best_delta = 0  # we'll compute the actual delta
                        # Recompute delta for this random choice
                        # For simplicity, just accept and recompute cost
                        colors[e_idx] = new_c
                        # Update CN
                        for x in range(n):
                            if x == u or x == v:
                                continue
                            c_vx = int(colors[edge_index[v, x]])
                            c_ux = int(colors[edge_index[u, x]])
                            if c_vx == old_c:
                                CN[old_c][u, x] -= 1
                                CN[old_c][x, u] -= 1
                            if c_ux == old_c:
                                CN[old_c][v, x] -= 1
                                CN[old_c][x, v] -= 1
                            if c_vx == new_c:
                                CN[new_c][u, x] += 1
                                CN[new_c][x, u] += 1
                            if c_ux == new_c:
                                CN[new_c][v, x] += 1
                                CN[new_c][x, v] += 1
                        # Update neighbors
                        neighbors[old_c][u].discard(v)
                        neighbors[old_c][v].discard(u)
                        neighbors[new_c][u].add(v)
                        neighbors[new_c][v].add(u)
                        current_cost = compute_cost(CN)
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
                    continue
                
                # Accept the best move
                colors[e_idx] = best_new_c
                current_cost += best_delta
                
                # Update CN
                for x in range(n):
                    if x == u or x == v:
                        continue
                    c_vx = int(colors[edge_index[v, x]])
                    c_ux = int(colors[edge_index[u, x]])
                    if c_vx == old_c:
                        CN[old_c][u, x] -= 1
                        CN[old_c][x, u] -= 1
                    if c_ux == old_c:
                        CN[old_c][v, x] -= 1
                        CN[old_c][x, v] -= 1
                    if c_vx == best_new_c:
                        CN[best_new_c][u, x] += 1
                        CN[best_new_c][x, u] += 1
                    if c_ux == best_new_c:
                        CN[best_new_c][v, x] += 1
                        CN[best_new_c][x, v] += 1
                
                # Update neighbors
                neighbors[old_c][u].discard(v)
                neighbors[old_c][v].discard(u)
                neighbors[best_new_c][u].add(v)
                neighbors[best_new_c][v].add(u)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        
        return best_colors.tolist() if best_cost == 0 else None
    
    # Try increasing n
    for n in range(12, 27):
        if time.time() > deadline - 10:
            break
        result = try_local_search(n)
        if result is not None:
            best_n = n
            best_colors = result
    
    return {'n': best_n, 'colors': best_colors}
