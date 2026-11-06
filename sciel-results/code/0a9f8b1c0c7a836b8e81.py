import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper: Fast MIS using Branch and Bound
    def max_independent_set_size(n, adj, time_limit):
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')
            
        def branch(candidates_mask, current_size):
            if time.time() > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + popcount(candidates)
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    # Helper: Get one maximum independent set
    def get_max_independent_set(n, adj):
        best_set = [0]
        best_size = [0]
        
        def popcount(x):
            return bin(x).count('1')
            
        def branch(candidates_mask, current_size, current_set):
            if candidates_mask == 0:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    best_set[0] = current_set
                return
            
            if current_size + popcount(candidates_mask) <= best_size[0]:
                return
            
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Include
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1, current_set | v_bit)
            
            # Exclude
            branch(candidates_mask & ~v_bit, current_size, current_set)

        branch((1 << n) - 1, 0, 0)
        return best_set[0]

    # --- Step 1: Construct PG(2, 7) Polarity Graph ---
    q = 7
    # Field F7 is just integers mod 7
    # Points in PG(2, 7): (x, y, z) not all 0, normalized
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, q)
                    points.append((1, (y * inv_x) % q, (z * inv_x) % q))
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    points.append((0, 1, (z * inv_y) % q))
                else:
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_full = len(points) # 57
    
    # Bilinear form: u0*v2 + u1*v1 + u2*v0
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % q

    adj_full = [0] * n_full
    for i in range(n_full):
        for j in range(i + 1, n_full):
            if bilinear(points[i], points[j]) == 0:
                adj_full[i] |= (1 << j)
                adj_full[j] |= (1 << i)
                
    # --- Step 2: Greedy Vertex Deletion ---
    # We want to find a subset S of vertices such that alpha(G[S]) <= 10.
    # We start with all vertices and remove them one by one.
    
    current_vertices = list(range(n_full))
    n_current = n_full
    adj_current = adj_full[:] # We will maintain adj for the induced subgraph
    
    # To make deletion easier, we can just keep a mask of active vertices
    # But updating adj masks is O(n).
    # Let's just keep a list of active vertex indices and rebuild adj if needed, or use a mask.
    # Since n <= 57, we can use bitmasks.
    
    active_mask = (1 << n_full) - 1
    
    # We iterate until alpha <= 10
    # We want to maximize the number of vertices kept.
    # So we remove the "least important" vertex.
    # Heuristic: Remove vertex from MIS with lowest degree in G[active].
    
    while time.time() < deadline - 5:
        # Compute MIS on active subgraph
        # We need adj restricted to active_mask
        # adj_active[i] = adj_full[i] & active_mask
        
        # Fast check: if popcount(active_mask) <= 38, we might still try to grow? 
        # No, we are shrinking.
        
        # Get MIS size
        # We can use the branch and bound.
        # To speed up, we can use a greedy coloring bound?
        # For now, B&B is fast enough for n=57.
        
        # Note: The B&B function expects adj list for the full n, but we can mask it.
        # Let's create a wrapper that masks adj.
        
        def get_mis_size_and_set(n, adj, mask, time_limit):
            best_size = [0]
            best_set = [0]
            
            def popcount(x):
                return bin(x).count('1')
                
            def branch(candidates_mask, current_size, current_set):
                if time.time() > time_limit:
                    return
                if candidates_mask == 0:
                    if current_size > best_size[0]:
                        best_size[0] = current_size
                        best_set[0] = current_set
                    return
                
                if current_size + popcount(candidates_mask) <= best_size[0]:
                    return
                
                best_v = -1
                best_deg = -1
                m = candidates_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    # Degree in candidate subgraph, masked by active_mask (which is already in candidates_mask)
                    deg = popcount(candidates_mask & (adj[v] & mask))
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    m -= lsb
                
                if best_v == -1:
                    return

                v_bit = 1 << best_v
                
                # Include
                new_candidates = candidates_mask & ~(adj[best_v] & mask) & ~v_bit
                branch(new_candidates, current_size + 1, current_set | v_bit)
                
                # Exclude
                branch(candidates_mask & ~v_bit, current_size, current_set)

            branch(mask, 0, 0)
            return best_size[0], best_set[0]

        alpha, mis_set = get_mis_size_and_set(n_full, adj_full, active_mask, time.time() + 10)
        
        if alpha <= 10:
            # We found a valid graph
            n_final = bin(active_mask).count('1')
            if n_final >= 39:
                # Extract edges
                edge_list = []
                for i in range(n_full):
                    if not (active_mask & (1 << i)):
                        continue
                    for j in range(i + 1, n_full):
                        if not (active_mask & (1 << j)):
                            continue
                        if adj_full[i] & (1 << j):
                            edge_list.append([i, j])
                return {'n': n_final, 'edges': edge_list}
            else:
                # Failed to keep 39 vertices
                # Try a different heuristic or break
                break
        
        # alpha > 10, need to remove a vertex
        # Get vertices in MIS
        mis_vertices = []
        m = mis_set
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            mis_vertices.append(v)
            m -= lsb
            
        # Choose vertex to remove
        # Heuristic: Min degree in active graph
        min_deg_v = -1
        min_deg = 1000
        for v in mis_vertices:
            deg = bin(adj_full[v] & active_mask).count('1')
            if deg < min_deg:
                min_deg = deg
                min_deg_v = v
                
        # Remove min_deg_v
        active_mask &= ~(1 << min_deg_v)
        
    # Fallback: Return best known or a smaller valid graph
    # If we didn't find one with n>=39, try to return whatever we have if valid, or a known 39-graph?
    # The previous program found 39. This one should too.
    # If it fails, return a small valid graph to avoid 0 score.
    # Let's construct a simple C4-free graph with alpha < 11.
    # E.g., a clique of size 10? No, alpha=1. n=10. Score low.
    # Better: Return the PG(2,5) graph (n=31) which is C4-free and has small alpha.
    
    # Construct PG(2,5) as fallback
    q5 = 5
    points5 = []
    for x in range(q5):
        for y in range(q5):
            for z in range(q5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, q5)
                    points5.append((1, (y * inv_x) % q5, (z * inv_x) % q5))
                elif y != 0:
                    inv_y = pow(y, -1, q5)
                    points5.append((0, 1, (z * inv_y) % q5))
                else:
                    points5.append((0, 0, 1))
    points5 = list(set(points5))
    n5 = len(points5)
    adj5 = [0] * n5
    for i in range(n5):
        for j in range(i + 1, n5):
            if (points5[i][0]*points5[j][2] + points5[i][1]*points5[j][1] + points5[i][2]*points5[j][0]) % q5 == 0:
                adj5[i] |= (1 << j)
                adj5[j] |= (1 << i)
                
    edge_list = []
    for i in range(n5):
        for j in range(i + 1, n5):
            if adj5[i] & (1 << j):
                edge_list.append([i, j])
                
    return {'n': n5, 'edges': edge_list}
