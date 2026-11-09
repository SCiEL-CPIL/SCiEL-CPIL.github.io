import random
import time
import itertools

def construct():
    # Fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    fallback_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    fallback = {'n': 17, 'edges': fallback_edges}
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    n = 35
    
    def is_valid(adj, n):
        # Check for K4
        # adj is a list of bitmasks
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] >> b) & 1:
                    continue
                # common neighbors of a and b
                common_ab = adj[a] & adj[b]
                # remove a and b from common
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                # iterate over pairs in common_ab
                # if any pair (c,d) in common_ab has edge c-d, then K4
                c_list = [i for i in range(n) if (common_ab >> i) & 1]
                for i in range(len(c_list)):
                    for j in range(i+1, len(c_list)):
                        c = c_list[i]
                        d = c_list[j]
                        if (adj[c] >> d) & 1:
                            return False
                            
        # Check for Independent Set of 6
        # Equivalent to K6 in the complement graph
        # Complement adjacency
        # comp_adj[u] has bits set for v where NOT adj[u][v] (and u!=v)
        comp_adj = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if u != v and not ((adj[u] >> v) & 1):
                    mask |= (1 << v)
            comp_adj[u] = mask
            
        # Find K6 in comp_adj
        for a in range(n):
            for b in range(a+1, n):
                if not (comp_adj[a] >> b) & 1:
                    continue
                common_ab = comp_adj[a] & comp_adj[b]
                common_ab &= ~(1 << a)
                common_ab &= ~(1 << b)
                c_list = [i for i in range(n) if (common_ab >> i) & 1]
                for i in range(len(c_list)):
                    for j in range(i+1, len(c_list)):
                        c = c_list[i]
                        d = c_list[j]
                        if not (comp_adj[c] >> d) & 1:
                            continue
                        # common neighbors of c and d in comp
                        common_cd = comp_adj[c] & comp_adj[d]
                        common_cd &= ~(1 << c)
                        common_cd &= ~(1 << d)
                        # We need one more vertex e such that e is connected to a,b,c,d in comp
                        # e must be in common_ab AND common_cd
                        # and e not in {a,b,c,d}
                        potential_e = common_ab & common_cd
                        potential_e &= ~(1 << a)
                        potential_e &= ~(1 << b)
                        potential_e &= ~(1 << c)
                        potential_e &= ~(1 << d)
                        if potential_e:
                            return False
        return True

    def get_edges_from_adj(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Local Search
    best_graph = None
    
    while time.time() < deadline:
        # Random initialization
        adj = [0] * n
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to find a violation and fix it
            # Randomly pick a K4 or Ind6?
            # For simplicity, just pick a random edge and see if flipping it reduces violations?
            # That's expensive.
            
            # Heuristic: Pick a random vertex, check its neighborhood for K4s or Ind6s?
            # Let's just do a random walk with acceptance if it doesn't create NEW violations?
            # Actually, we need to REDUCE violations.
            
            # Let's try flipping a random edge and checking if the number of violations decreases.
            # Counting violations is expensive.
            
            # Alternative: Greedy repair.
            # Find ONE K4 or ONE Ind6. If found, flip an edge in it.
            
            found_violation = False
            # Check for K4
            # Sample some vertices to find K4 quickly?
            # Or just iterate?
            # n=35, K4 check is fast.
            
            # Let's iterate through all K4 candidates
            k4_found = None
            for a in range(n):
                for b in range(a+1, n):
                    if not (adj[a] >> b) & 1: continue
                    common_ab = adj[a] & adj[b]
                    common_ab &= ~(1 << a)
                    common_ab &= ~(1 << b)
                    c_list = [i for i in range(n) if (common_ab >> i) & 1]
                    for i in range(len(c_list)):
                        for j in range(i+1, len(c_list)):
                            c = c_list[i]
                            d = c_list[j]
                            if (adj[c] >> d) & 1:
                                k4_found = [a, b, c, d]
                                break
                        if k4_found: break
                    if k4_found: break
                if k4_found: break
            
            if k4_found:
                # Flip a random edge in the K4
                verts = k4_found
                edge_idx = random.randint(0, 5)
                pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
                u, v = verts[pairs[edge_idx][0]], verts[pairs[edge_idx][1]]
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                improved = True
                found_violation = True
                continue
            
            # Check for Ind6 (K6 in complement)
            ind6_found = None
            # Reuse logic or simplified
            # To save time, we can just check a few random 6-sets?
            # No, we need to be sure.
            # But if K4 is gone, we check Ind6.
            
            # Efficient Ind6 check
            # We can use the same logic as is_valid but stop at first found.
            comp_adj = [0] * n
            for u in range(n):
                mask = 0
                for v in range(n):
                    if u != v and not ((adj[u] >> v) & 1):
                        mask |= (1 << v)
                comp_adj[u] = mask
                
            for a in range(n):
                for b in range(a+1, n):
                    if not (comp_adj[a] >> b) & 1: continue
                    common_ab = comp_adj[a] & comp_adj[b]
                    common_ab &= ~(1 << a)
                    common_ab &= ~(1 << b)
                    c_list = [i for i in range(n) if (common_ab >> i) & 1]
                    for i in range(len(c_list)):
                        for j in range(i+1, len(c_list)):
                            c = c_list[i]
                            d = c_list[j]
                            if not (comp_adj[c] >> d) & 1: continue
                            common_cd = comp_adj[c] & comp_adj[d]
                            common_cd &= ~(1 << c)
                            common_cd &= ~(1 << d)
                            potential_e = common_ab & common_cd
                            potential_e &= ~(1 << a)
                            potential_e &= ~(1 << b)
                            potential_e &= ~(1 << c)
                            potential_e &= ~(1 << d)
                            if potential_e:
                                # Find the vertex e
                                e = -1
                                for k in range(n):
                                    if (potential_e >> k) & 1:
                                        e = k
                                        break
                                ind6_found = [a, b, c, d, e]
                                break
                        if ind6_found: break
                    if ind6_found: break
                if ind6_found: break
            
            if ind6_found:
                # Flip a random edge in the Ind6 (which is a non-edge in G)
                # Wait, if it's an independent set in G, all pairs are non-edges.
                # We want to ADD an edge to break it.
                verts = ind6_found
                edge_idx = random.randint(0, 9) # 5 choose 2 = 10 edges
                pairs = list(itertools.combinations(range(5), 2))
                u, v = verts[pairs[edge_idx][0]], verts[pairs[edge_idx][1]]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                improved = True
                found_violation = True
                continue
            
            # If no violation found, we have a valid graph!
            if not found_violation:
                # Verify fully just in case
                if is_valid(adj, n):
                    best_graph = adj
                    break
            
            # If we didn't find a violation but also didn't break out, it means the loop above finished without setting found_violation?
            # The logic above breaks if found. If not found, found_violation remains False.
            # So if not found_violation, we break.
            break
            
        if best_graph:
            break
        # If loop finishes without best_graph, restart with new random seed
    
    if best_graph:
        return {'n': n, 'edges': get_edges_from_adj(best_graph, n)}
    else:
        return fallback
