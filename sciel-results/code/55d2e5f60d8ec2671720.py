import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Construct the polarity graph of the projective plane of order 5 (n=31).
    2. Iteratively add vertices using a conflict graph to ensure C4-freeness.
    3. Check independence number < 11 with fast bit-parallel solver.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # --- Helper: Fast Alpha Solver using Bitmasks with better pruning ---
    def max_independent_set_size(n, adj, time_limit):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        Returns best found (may be lower bound if time runs out).
        """
        best = [0]
        
        def popcount(x):
            return x.bit_count()

        def branch(candidates_mask, current_size):
            if time.time() > time_limit:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current + number of candidates
            remaining = popcount(candidates_mask)
            if current_size + remaining <= best[0]:
                return
            
            # Quick check: if we can already beat best with just current
            if current_size > best[0]:
                best[0] = current_size
            
            # Pick vertex with highest degree in candidate subgraph
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
                m ^= lsb
            
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

    # --- Step 1: Construct Polarity Graph of PG(2,5) ---
    # Points in PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, z * inv_y % 5))
                else:
                    inv_z = pow(z, -1, 5)
                    points.append((0, 0, 1))
    
    points = list(set(points))
    n_base = len(points)  # Should be 31
    
    def bilinear(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5

    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    if max_independent_set_size(n_base, adj, deadline - 15) >= 11:
        return {'n': 30, 'edges': []}

    best_n = n_base
    best_adj = adj[:]

    # --- Step 2: Iterative Growth ---
    for n in range(n_base + 1, 44):
        if time.time() > deadline - 25:
            break
        
        found = False
        
        # Build conflict graph once per n
        conflict = [0] * (n - 1)
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                if adj[i] & adj[j]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # Try multiple strategies with different random seeds
        for attempt in range(300):
            if time.time() > deadline - 10:
                break
            
            random.seed(42 + attempt * 13)
            candidates = (1 << (n - 1)) - 1
            S_mask = 0
            
            # Strategy varies based on attempt number
            strategy = attempt % 3
            
            while candidates:
                best_v = -1
                min_conflicts = n + 1
                m = candidates
                verts = []
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m ^= lsb
                
                if strategy == 0:
                    random.shuffle(verts)
                elif strategy == 1:
                    # Sort by degree in conflict graph (descending)
                    verts.sort(key=lambda v: (conflict[v] & candidates).bit_count(), reverse=True)
                else:
                    # Sort by degree in original graph (ascending) - prefer low degree
                    verts.sort(key=lambda v: adj[v].bit_count())
                
                for v in verts:
                    if not (candidates & (1 << v)):
                        continue
                    conf_count = (conflict[v] & candidates).bit_count()
                    if conf_count < min_conflicts:
                        min_conflicts = conf_count
                        best_v = v
                        if min_conflicts == 0:
                            break
                
                if best_v == -1:
                    break
                    
                S_mask |= (1 << best_v)
                candidates &= ~(1 << best_v)
                candidates &= ~conflict[best_v]
            
            # Add vertex n-1
            new_adj = adj[:] + [0]
            new_adj[n-1] = S_mask
            for i in range(n - 1):
                if S_mask & (1 << i):
                    new_adj[i] |= (1 << (n-1))
            
            # Quick alpha check with tight time limit
            alpha = max_independent_set_size(n, new_adj, deadline - 5)
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
