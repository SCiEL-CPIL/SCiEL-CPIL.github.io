import time
import random
import math
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    q = 5
    n = 31
    
    # 1. Generate 31 points of PG(2,5)
    # Standard representation of points in PG(2, q)
    points = []
    # (1, x, y) for x, y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # (0, 1, z) for z in GF(q)
    for z in range(q):
        points.append((0, 1, z))
    # (0, 0, 1)
    points.append((0, 0, 1))
    
    # 2. Construct Polarity Graph
    # Edge if dot product is 0 mod q
    adj = [set() for _ in range(n)]
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            # Dot product in GF(5)
            dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q
            if dot == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.append((i, j))
                
    target_edges = 91
    current_edges_count = len(base_edges)
    edges_to_add = target_edges - current_edges_count
    
    # For q=5, current_edges_count should be 90, so edges_to_add = 1.
    # If this assumption holds, we can solve it exactly and quickly.
    
    if edges_to_add == 1:
        # Identify non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    non_edges.append((i, j))
                    
        best_edge = None
        min_cost = float('inf')
        
        # Calculate marginal cost for adding each non-edge
        # Cost = C(|N(u) intersect N(v)|, 2)
        for u, v in non_edges:
            common = len(adj[u] & adj[v])
            # Number of new C4s formed by adding (u,v)
            # Each pair of common neighbors (w, x) forms a C4 u-w-v-x-u
            cost = common * (common - 1) // 2
            
            if cost < min_cost:
                min_cost = cost
                best_edge = (u, v)
                
        if best_edge is None:
            # Fallback if something went wrong
            best_edge = non_edges[0] if non_edges else (0, 1)
            
        # Add the best edge
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        base_edges.append((u, v))
        
        # Final verification count (using the fast method from current program logic)
        # Just to be absolutely sure, though the incremental calculation is exact.
        # We can skip full recount if we trust the math, but let's do a quick sanity check
        # or just return. The incremental math is rigorous.
        
        final_edges = []
        for u, v in base_edges:
            final_edges.append([u, v])
            
        return final_edges

    else:
        # Fallback for unexpected edge counts: Greedy + SA
        # This path is unlikely to be taken for q=5, n=31, m=91
        
        # Precompute combos_4 for fast counting
        combos_4 = []
        for c in itertools.combinations(range(n), 4):
            combos_4.append(c)
            
        def count_total_c4s(adj_sets):
            total = 0
            # Map edge pairs to indices in the 4-combo
            # For a combo (a,b,c,d), edges are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
            # We need a fast lookup.
            # Since n is small, we can just check membership.
            for a, b, c, d in combos_4:
                mask = 0
                if b in adj_sets[a]: mask |= 1
                if c in adj_sets[a]: mask |= 2
                if d in adj_sets[a]: mask |= 4
                if c in adj_sets[b]: mask |= 8
                if d in adj_sets[b]: mask |= 16
                if d in adj_sets[c]: mask |= 32
                # C4s in K4:
                # 1-2-3-4-1 (edges 0,3,5,2 -> bits 0,3,5,2? No, indices 0,2,3,5 in 0-5 mapping)
                # Let's stick to the bitmask logic from the seed
                # Bits: 0:ab, 1:ac, 2:ad, 3:bc, 4:bd, 5:cd
                # C4 1: a-b-c-d-a -> ab, bc, cd, da (bits 0, 3, 5, 2)
                # C4 2: a-b-d-c-a -> ab, bd, dc, ca (bits 0, 4, 5, 1)
                # C4 3: a-c-b-d-a -> ac, cb, bd, da (bits 1, 3, 4, 2)
                
                if (mask & 1) and (mask & 8) and (mask & 32) and (mask & 4): count_total_c4s.c4_count += 1 # Placeholder
                # Actually, let's just use the common neighbor formula for this fallback, it's simpler
                pass
            
            # Re-implementing count_total_c4s properly for fallback
            count = 0
            for u in range(n):
                for v in range(u+1, n):
                    common = len(adj_sets[u] & adj_sets[v])
                    if common >= 2:
                        count += common * (common - 1) // 2
            return count // 2
        
        # Greedy initialization
        added_edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    non_edges.append((i, j))
                    
        current_adj = [set(s) for s in adj]
        
        while len(added_edges) < edges_to_add:
            best_edge = None
            min_c = float('inf')
            for e in non_edges:
                if e in added_edges: continue
                u, v = e
                c = len(current_adj[u] & current_adj[v])
                if c < min_c:
                    min_c = c
                    best_edge = e
            if best_edge is None: break
            added_edges.append(best_edge)
            u, v = best_edge
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        # SA Refinement
        current_added_set = set(added_edges)
        current_adj = [set(s) for s in adj]
        for u, v in current_added_set:
            current_adj[u].add(v)
            current_adj[v].add(u)
            
        current_c4s = count_total_c4s(current_adj)
        best_c4s = current_c4s
        best_added_set = set(current_added_set)
        
        T = 50.0
        T_min = 1.0
        alpha = 0.99
        
        while time.time() < deadline:
            if T < T_min: break
            if not current_added_set: break
            
            e_out = random.choice(list(current_added_set))
            e_in = None
            while e_in is None or e_in in current_added_set:
                e_in = random.choice(non_edges)
                
            u_out, v_out = e_out
            u_in, v_in = e_in
            
            current_adj[u_out].remove(v_out)
            current_adj[v_out].remove(u_out)
            current_added_set.remove(e_out)
            
            current_adj[u_in].add(v_in)
            current_adj[v_in].add(u_in)
            current_added_set.add(e_in)
            
            new_c4s = count_total_c4s(current_adj)
            delta = new_c4s - current_c4s
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_c4s = new_c4s
                if new_c4s < best_c4s:
                    best_c4s = new_c4s
                    best_added_set = set(current_added_set)
            else:
                current_adj[u_in].remove(v_in)
                current_adj[v_in].remove(u_in)
                current_added_set.remove(e_in)
                
                current_adj[u_out].add(v_out)
                current_adj[v_out].add(u_out)
                current_added_set.add(e_out)
                
            T *= alpha
            
        final_edges = []
        for u, v in base_edges:
            final_edges.append([u, v])
        for u, v in best_added_set:
            final_edges.append([u, v])
            
        return final_edges
