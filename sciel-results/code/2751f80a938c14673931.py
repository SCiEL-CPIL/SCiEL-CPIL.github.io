import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Construct the polarity graph of the projective plane of order 5 (n=31).
    2. Iteratively add vertices using a conflict graph to ensure C4-freeness.
    3. Check independence number < 11.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # --- Helper: Fast Alpha Solver using Bitmasks ---
    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        """
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')

        def branch(candidates_mask, current_size):
            if time.time() > deadline - 5:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            # To speed up, we can just pick the first one, but degree heuristic is better.
            # For n<=43, iterating through bits is fast.
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                # Degree in candidate subgraph
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            # New candidates: those in candidates_mask not adjacent to best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    # --- Step 1: Construct Polarity Graph of PG(2,5) ---
    # Points in PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero component 1
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, z * inv_y % 5))
                else:
                    inv_z = pow(z, -1, 5)
                    points.append((0, 0, 1))
    
    # Remove duplicates (normalization should handle it, but let's be safe)
    points = list(set(points))
    n_base = len(points) # Should be 31
    
    # Bilinear form B(u,v) = u0*v2 + u1*v1 + u2*v0
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5

    # Build adjacency
    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Verify base graph
    if max_independent_set_size(n_base, adj) >= 11:
        # Fallback if something went wrong, but it shouldn't
        return {'n': 30, 'edges': []}

    best_n = n_base
    best_adj = adj[:]

    # --- Step 2: Iterative Growth ---
    # Conflict Graph: u, v conflict if N(u) & N(v) != 0
    # We maintain adj as bitmasks.
    
    for n in range(n_base + 1, 44):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random restarts for choosing the independent set S in Conflict Graph
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Build Conflict Graph for current n-1 vertices
            # Conflict[i] is bitmask of vertices that conflict with i
            # i, j conflict if adj[i] & adj[j] != 0
            conflict = [0] * (n - 1)
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    if adj[i] & adj[j]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            
            # Find an independent set S in Conflict Graph
            # Greedy with randomization
            random.seed(12345 + attempt)
            candidates = (1 << (n - 1)) - 1
            S_mask = 0
            
            while candidates:
                # Pick a vertex from candidates
                # Heuristic: pick one with fewest conflicts in remaining candidates
                best_v = -1
                min_conflicts = n
                m = candidates
                # Shuffle to add randomness
                verts = []
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m -= lsb
                random.shuffle(verts)
                
                for v in verts:
                    if not (candidates & (1 << v)):
                        continue
                    # Count conflicts in candidates
                    conf_count = bin(conflict[v] & candidates).count('1')
                    if conf_count < min_conflicts:
                        min_conflicts = conf_count
                        best_v = v
                        if min_conflicts == 0:
                            break
                
                if best_v == -1:
                    break
                    
                S_mask |= (1 << best_v)
                # Remove best_v and its neighbors from candidates
                candidates &= ~(1 << best_v)
                candidates &= ~conflict[best_v]
                
            # S_mask is the set of neighbors for the new vertex n-1
            # Add vertex n-1
            new_adj = adj[:] + [0]
            new_adj[n-1] = S_mask
            for i in range(n - 1):
                if S_mask & (1 << i):
                    new_adj[i] |= (1 << (n-1))
            
            # Check Alpha
            alpha = max_independent_set_size(n, new_adj)
            if alpha < 11:
                adj = new_adj
                best_n = n
                best_adj = adj[:]
                found = True
                break
        
        if not found:
            break

    # Convert best_adj to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
