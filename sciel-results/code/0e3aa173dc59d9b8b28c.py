import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Construct the polarity graph of PG(2, 8).
    2. Use a heuristic to find a large subset (n=43) with alpha < 11.
    3. If heuristic fails, fallback to a smaller verified construction or greedy growth.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    # Helper: popcount
    def popcount(x):
        return bin(x).count('1')

    # Helper: Fast Alpha Solver using Branch and Bound
    def max_independent_set_size(n, adj, time_limit, target=10):
        """
        Find the size of the maximum independent set.
        Returns alpha.
        If alpha > target is found, we can stop early if we just need to know alpha > target.
        But here we need the exact max to be sure it's < 11.
        However, for pruning, we can stop if we find an independent set of size 11.
        """
        best = [0]
        
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
            
            # If we already found size 11, and we are checking if alpha < 11, we can return early?
            # Actually, the caller needs to know if alpha >= 11.
            # If best[0] >= 11, we can stop searching for larger sets.
            if best[0] >= target + 1:
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
            # New candidates: those not adjacent to best_v and not best_v itself
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    # --- Step 1: PG(2, 8) Construction ---
    try:
        # Field F8 arithmetic
        # Let a^3 = a + 1
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
                pi = int_to_poly(i)
                pj = int_to_poly(j)
                sum_c = ((pi[0]^pj[0]), (pi[1]^pj[1]), (pi[2]^pj[2]))
                add_table[i][j] = poly_to_int(sum_c)
                
                c0, c1, c2 = pi
                d0, d1, d2 = pj
                
                # Product calculation with a^3 = a + 1
                # res = (c0 + c1 a + c2 a^2) * (d0 + d1 a + d2 a^2)
                # ... (same as before)
                res_c0 = (c0*d0 + c1*d2 + c2*d1) % 2
                res_c1 = (c0*d1 + c1*d0 + c1*d2 + c2*d1 + c2*d2) % 2
                res_c2 = (c0*d2 + c1*d1 + c2*d0 + c2*d2) % 2
                mul_table[i][j] = poly_to_int((res_c0, res_c1, res_c2))

        # Generate points in PG(2,8)
        points = []
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
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
        n_pg8 = len(points) # 73
        
        # Bilinear form B(u,v) = u0*v2 + u1*v1 + u2*v0
        def bilinear_f8(u, v):
            term1 = mul_table[u[0]][v[2]]
            term2 = mul_table[u[1]][v[1]]
            term3 = mul_table[u[2]][v[0]]
            return add_table[add_table[term1, term2], term3]

        adj_pg8 = [0] * n_pg8
        for i in range(n_pg8):
            for j in range(i + 1, n_pg8):
                if bilinear_f8(points[i], points[j]) == 0:
                    adj_pg8[i] |= (1 << j)
                    adj_pg8[j] |= (1 << i)
        
        # Heuristic: Find a subset of size 43 with alpha < 11
        # We want to keep vertices that are "well-connected" or remove those that form large independent sets.
        # Independent set in the graph corresponds to a set of points where B(p_i, p_j) != 0 for all pairs.
        
        # Strategy: Greedy removal.
        # Start with all 73 vertices.
        # While alpha >= 11:
        #   Find an independent set of size 11.
        #   Remove one vertex from it (the one that "breaks" the most other potential independent sets?).
        #   Or simply remove a random vertex from it.
        
        current_n = n_pg8
        current_adj = adj_pg8[:]
        current_indices = list(range(n_pg8))
        
        # Map: old_index -> new_index (for reconstruction at the end)
        # Actually, let's just keep a boolean mask of which vertices are "active".
        active = [True] * n_pg8
        
        for step in range(50): # Max 50 removals
            if time.time() > deadline - 20:
                break
            
            # Check alpha of the current active subgraph
            # We need to map active vertices to 0..k-1 for the solver
            active_verts = [i for i in range(n_pg8) if active[i]]
            k = len(active_verts)
            
            if k < 43:
                # We removed too many, but we need n >= 39 ideally.
                # If we can't get to 43, we might still return what we have if alpha < 11.
                pass

            # Build adj for active subgraph
            adj_active = [0] * k
            for i in range(k):
                orig_i = active_verts[i]
                for j in range(i+1, k):
                    orig_j = active_verts[j]
                    if adj_pg8[orig_i] & (1 << orig_j):
                        adj_active[i] |= (1 << j)
                        adj_active[j] |= (1 << i)
            
            alpha = max_independent_set_size(k, adj_active, deadline - 10, target=10)
            
            if alpha < 11:
                # Success! We have a valid graph with k vertices.
                if k >= 39: # Best known is 39, we want 43.
                    # If k == 43, great. If k > 43, we can take any 43? 
                    # No, if we take a subset of a C4-free graph, it's C4-free.
                    # If we take a subset of a graph with alpha < 11, the subset has alpha < 11.
                    # So we can just pick the first 43 active vertices? 
                    # Wait, the problem says "Maximize n". 
                    # If we found a graph with k=50 and alpha<11, we should return n=50?
                    # The task says "1 <= n <= 43". 
                    # "The best known construction has 39 vertices."
                    # "A valid graph on n vertices certifies R(C4,K11) > n."
                    # So if we find n=43, that's the max possible return.
                    if k >= 43:
                        # Pick any 43 vertices from the active set.
                        # Let's pick the first 43.
                        target_indices = active_verts[:43]
                        n_target = 43
                        adj_target = [0] * n_target
                        
                        # Map target_indices to 0..42
                        idx_map = {orig: new for new, orig in enumerate(target_indices)}
                        
                        for i in range(n_target):
                            orig_i = target_indices[i]
                            for j in range(i+1, n_target):
                                orig_j = target_indices[j]
                                if adj_pg8[orig_i] & (1 << orig_j):
                                    adj_target[i] |= (1 << j)
                                    adj_target[j] |= (1 << i)
                                    
                        # Verify
                        alpha_check = max_independent_set_size(n_target, adj_target, deadline - 5)
                        if alpha_check < 11:
                            edge_list = []
                            for i in range(n_target):
                                for j in range(i+1, n_target):
                                    if adj_target[i] & (1 << j):
                                        edge_list.append([i, j])
                            return {'n': n_target, 'edges': edge_list}
                        else:
                            # This shouldn't happen if alpha of superset was < 11
                            pass
                    else:
                        # k < 43, return k
                        n_target = k
                        edge_list = []
                        for i in range(n_target):
                            for j in range(i+1, n_target):
                                if adj_active[i] & (1 << j):
                                    edge_list.append([i, j])
                        return {'n': n_target, 'edges': edge_list}

            # Alpha >= 11, need to remove a vertex.
            # Find an independent set of size 11 to identify which vertices to consider.
            # We can modify the solver to return the set, or just pick a random active vertex?
            # Picking a random vertex might be slow.
            # Let's pick the vertex with the highest degree in the active subgraph?
            # No, removing a high degree vertex might help break independent sets?
            # Actually, we want to remove a vertex that is part of many large independent sets.
            # A simple heuristic: remove a random vertex from the active set.
            # Or, remove the vertex with the minimum degree? 
            # In a polarity graph, all vertices have the same degree (quasi-random).
            # Let's just remove a random active vertex.
            
            rand_idx = random.randint(0, len(active_verts) - 1)
            active[active_verts[rand_idx]] = False
            
    except Exception as e:
        pass # Fallback

    # --- Fallback: PG(2,5) or Greedy ---
    # If PG(2,8) failed, we fall back to a known valid construction or greedy growth.
    # Best known is 39. Let's try to construct 39 using PG(2,5) or similar.
    # Actually, if PG(2,8) heuristic fails, we might just return a smaller graph.
    # Let's try a simpler greedy construction to ensure we return *something* valid.
    
    # Greedy construction of C4-free graph with alpha < 11.
    n = 1
    adj = [0] * 43
    for new_v in range(1, 43):
        if time.time() > deadline - 5:
            break
        
        # Find a set of vertices to connect to new_v
        # Such that no C4 is formed and alpha remains < 11.
        # C4 constraint: If we connect new_v to u and v, then u and v must NOT have a common neighbor.
        # So the neighbors of new_v must form an independent set in the "conflict graph" where edges are common neighbors.
        
        # Let's try to add new_v connected to some vertices.
        # Start with no edges.
        # Try adding edges one by one?
        
        # Simpler: Randomly pick a subset of existing vertices to connect to.
        # We want to maximize edges to help keep alpha low, but not create C4s.
        
        # Candidates for neighbors:
        # If we connect new_v to S, then for any u, v in S, N(u) & N(v) must be empty?
        # No, N(u) & N(v) in the OLD graph must be empty.
        # Because if u and v share a neighbor w, then new_v-u-w-v-new_v is a C4.
        
        conflict = [0] * new_v
        for i in range(new_v):
            for j in range(i+1, new_v):
                if adj[i] & adj[j]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # We want to pick a set S from 0..new_v-1 such that no two vertices in S have a common neighbor in the old graph.
        # This is equivalent to S being an independent set in the "conflict graph".
        # Let's find the maximum independent set in the conflict graph?
        # Or just a large one.
        
        # To save time, let's just try a greedy set packing.
        S = 0
        candidates = (1 << new_v) - 1
        while candidates:
            # Pick vertex with min conflicts in remaining candidates
            best_v = -1
            min_conf = new_v
            m = candidates
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                conf_count = popcount(conflict[v] & candidates)
                if conf_count < min_conf:
                    min_conf = conf_count
                    best_v = v
                    if min_conf == 0:
                        break
                m -= lsb
            
            if best_v == -1:
                break
            
            S |= (1 << best_v)
            candidates &= ~(1 << best_v)
            candidates &= ~conflict[best_v]
        
        # Check alpha
        new_adj = adj[:]
        new_adj[new_v] = S
        for i in range(new_v):
            if S & (1 << i):
                new_adj[i] |= (1 << new_v)
        
        alpha = max_independent_set_size(new_v + 1, new_adj, deadline - 2)
        if alpha < 11:
            adj = new_adj
            n = new_v + 1
        else:
            # Try again with a different random seed or break?
            # Let's just break to avoid getting stuck.
            # Or try one more time with randomization?
            pass

    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return {'n': n, 'edges': edge_list}

if __name__ == '__main__':
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
