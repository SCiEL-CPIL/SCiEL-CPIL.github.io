import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Try to construct a subgraph of the Polarity Graph of PG(2,8) with n=43.
    2. If that fails (e.g., alpha >= 11), fallback to PG(2,5) growth.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # --- Helper: Fast Alpha Solver using Bitmasks ---
    def max_independent_set_size(n, adj, time_limit):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        """
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
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
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

    # --- Step 1: Try PG(2,8) Polarity Graph Subset ---
    try:
        # Field F8 arithmetic
        # Let alpha^3 = alpha + 1
        # Elements: 0, 1, a, a+1, a^2, a^2+a, a^2+a+1, a^2+1
        # Map:
        # 0: 0
        # 1: 1
        # a: 2
        # a+1: 3
        # a^2: 4
        # a^2+a: 5
        # a^2+a+1: 6
        # a^2+1: 7
        
        # Precompute addition and multiplication tables
        # We represent elements as ints 0-7.
        # We need a mapping from int to polynomial coefficients (c0, c1, c2) where val = c0 + c1*a + c2*a^2
        def int_to_poly(x):
            if x == 0: return (0,0,0)
            if x == 1: return (1,0,0)
            if x == 2: return (0,1,0)
            if x == 3: return (1,1,0)
            if x == 4: return (0,0,1)
            if x == 5: return (0,1,1)
            if x == 6: return (1,1,1)
            if x == 7: return (1,0,1)
            return (0,0,0)

        def poly_to_int(c):
            c0, c1, c2 = c
            if c0 == 0 and c1 == 0 and c2 == 0: return 0
            if c0 == 1 and c1 == 0 and c2 == 0: return 1
            if c0 == 0 and c1 == 1 and c2 == 0: return 2
            if c0 == 1 and c1 == 1 and c2 == 0: return 3
            if c0 == 0 and c1 == 0 and c2 == 1: return 4
            if c0 == 0 and c1 == 1 and c2 == 1: return 5
            if c0 == 1 and c1 == 1 and c2 == 1: return 6
            if c0 == 1 and c1 == 0 and c2 == 1: return 7
            return 0

        add_table = [[0]*8 for _ in range(8)]
        mul_table = [[0]*8 for _ in range(8)]
        
        for i in range(8):
            for j in range(8):
                # Addition
                pi = int_to_poly(i)
                pj = int_to_poly(j)
                sum_c = ((pi[0]^pj[0]), (pi[1]^pj[1]), (pi[2]^pj[2]))
                add_table[i][j] = poly_to_int(sum_c)
                
                # Multiplication
                # (c0 + c1 a + c2 a^2) * (d0 + d1 a + d2 a^2)
                # a^3 = a + 1
                # a^4 = a(a+1) = a^2 + a
                # a^5 = a(a^2+a) = a^3 + a^2 = (a+1) + a^2 = a^2 + a + 1
                # a^6 = a(a^2+a+1) = a^3 + a^2 + a = (a+1) + a^2 + a = a^2 + 1
                # a^7 = a(a^2+1) = a^3 + a = (a+1) + a = 1
                
                c0, c1, c2 = pi
                d0, d1, d2 = pj
                
                # Coeffs of product before reduction
                # c0*d0 + (c0*d1 + c1*d0)a + (c0*d2 + c1*d1 + c2*d0)a^2 + (c1*d2 + c2*d1)a^3 + c2*d2 a^4
                # a^3 -> a+1
                # a^4 -> a^2+a
                
                # Constant term:
                # from c0*d0
                # from a^3 term: (c1*d2 + c2*d1) * 1
                # from a^4 term: c2*d2 * 0 (constant part of a^2+a is 0)
                res_c0 = (c0*d0 + c1*d2 + c2*d1) % 2
                
                # a term:
                # from (c0*d1 + c1*d0)
                # from a^3 term: (c1*d2 + c2*d1) * 1
                # from a^4 term: c2*d2 * 1
                res_c1 = (c0*d1 + c1*d0 + c1*d2 + c2*d1 + c2*d2) % 2
                
                # a^2 term:
                # from (c0*d2 + c1*d1 + c2*d0)
                # from a^4 term: c2*d2 * 1
                res_c2 = (c0*d2 + c1*d1 + c2*d0 + c2*d2) % 2
                
                mul_table[i][j] = poly_to_int((res_c0, res_c1, res_c2))

        # Generate points in PG(2,8)
        points = []
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        # inv_x
                        # Find k such that mul_table[x][k] == 1
                        inv_x = 0
                        for k in range(8):
                            if mul_table[x][k] == 1:
                                inv_x = k
                                break
                        points.append((1, add_table[add_table[y, inv_x], 0], add_table[add_table[z, inv_x], 0]))
                    elif y != 0:
                        inv_y = 0
                        for k in range(8):
                            if mul_table[y][k] == 1:
                                inv_y = k
                                break
                        points.append((0, 1, add_table[add_table[z, inv_y], 0]))
                    else:
                        points.append((0, 0, 1))
        
        points = list(set(points))
        n_pg8 = len(points) # Should be 73
        
        # Bilinear form B(u,v) = u0*v2 + u1*v1 + u2*v0
        def bilinear_f8(u, v):
            # u0*v2
            term1 = mul_table[u[0]][v[2]]
            # u1*v1
            term2 = mul_table[u[1]][v[1]]
            # u2*v0
            term3 = mul_table[u[2]][v[0]]
            # Sum
            return add_table[add_table[term1, term2], term3]

        adj_pg8 = [0] * n_pg8
        for i in range(n_pg8):
            for j in range(i + 1, n_pg8):
                if bilinear_f8(points[i], points[j]) == 0:
                    adj_pg8[i] |= (1 << j)
                    adj_pg8[j] |= (1 << i)
        
        # Check alpha of the full 73-vertex graph
        # If alpha < 11, we can take any 43 vertices.
        # If alpha >= 11, we might still find a subset of 43 with alpha < 11, but it's harder.
        # Let's check alpha first.
        alpha_full = max_independent_set_size(n_pg8, adj_pg8, deadline - 20)
        
        if alpha_full < 11:
            # Take first 43 vertices
            n_target = 43
            # Remap vertices 0..42
            adj_target = [0] * n_target
            for i in range(n_target):
                # Mask out bits >= 43
                mask = (1 << n_target) - 1
                adj_target[i] = adj_pg8[i] & mask
                # Ensure symmetry (should be fine if we mask both)
            
            # Verify C4 and Alpha for the subset
            # C4 is inherited.
            # Check Alpha
            alpha_sub = max_independent_set_size(n_target, adj_target, deadline - 5)
            if alpha_sub < 11:
                edge_list = []
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj_target[i] & (1 << j):
                            edge_list.append([i, j])
                return {'n': n_target, 'edges': edge_list}
                
    except Exception as e:
        pass # Fallback to PG(2,5)

    # --- Fallback: PG(2,5) Growth ---
    # ... (Keep existing PG(2,5) code) ...
    
    # Re-include the PG(2,5) logic here if PG(2,8) fails
    # To save space, I will just copy the relevant parts from the original program
    # or assume the original code is still there if I only replace the top part.
    # But I must provide a complete construct().
    
    # Let's refactor to keep the PG(2,5) code as a fallback function.
    # Actually, I will just append the PG(2,5) code after the PG(2,8) block.
    
    # --- Step 2: Construct Polarity Graph of PG(2,5) ---
    points5 = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points5.append((1, y * inv_x % 5, z * inv_x % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points5.append((0, 1, z * inv_y % 5))
                else:
                    points5.append((0, 0, 1))
    
    points5 = list(set(points5))
    n_base = len(points5)
    
    def bilinear5(u, v):
        return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5

    adj = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear5(points5[i], points5[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_n = n_base
    best_adj = adj[:]

    # --- Step 3: Iterative Growth ---
    for n in range(n_base + 1, 44):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            conflict = [0] * (n - 1)
            for i in range(n - 1):
                for j in range(i + 1, n - 1):
                    if adj[i] & adj[j]:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            
            random.seed(12345 + attempt)
            candidates = (1 << (n - 1)) - 1
            S_mask = 0
            
            while candidates:
                best_v = -1
                min_conflicts = n
                m = candidates
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
                    conf_count = bin(conflict[v] & candidates).count('1')
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
                
            new_adj = adj[:] + [0]
            new_adj[n-1] = S_mask
            for i in range(n - 1):
                if S_mask & (1 << i):
                    new_adj[i] |= (1 << (n-1))
            
            alpha = max_independent_set_size(n, new_adj, deadline - 5)
            if alpha < 11:
                adj = new_adj
                best_n = n
                best_adj = adj[:]
                found = True
                break
        
        if not found:
            break

    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
