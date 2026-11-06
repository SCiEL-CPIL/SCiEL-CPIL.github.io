import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def check_k4(n, adj):
        # adj is list of ints (bitsets)
        # Check for K4
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    # Check if common has an edge
                    # Iterate through bits in common
                    c = common
                    while c:
                        # Get lowest set bit
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c -= lsb
                        # Check if w is connected to any other in common
                        # common without w
                        common_wo_w = common & ~(1 << w)
                        if adj[w] & common_wo_w:
                            return (u, v, w) # Return vertices of K4 (partial)
        return None

    def check_ind6(n, adj):
        # Check for independent set of size 6
        # Equivalent to K6 in complement
        # Build complement adjacency
        mask = (1 << n) - 1
        comp = [0] * n
        for i in range(n):
            # Complement of adj[i] within mask, excluding self
            comp[i] = (mask ^ adj[i]) & ~(1 << i)
        
        # Check for K6 in comp
        # Simple check: iterate 6-subsets? 
        # For n=36, C(36,6) ~ 1.9M. 
        # Each check is O(1) with bitsets if we are clever?
        # Actually, checking if a specific 6-set is a clique:
        # Pick v0, v1, v2, v3, v4, v5
        # Check adj[v0] & (1<<v1) ... etc.
        
        # Heuristic: Random sampling of 6-sets? 
        # If we find one, return it.
        # If we don't find one after many samples, assume none? 
        # But we need exact check for validity.
        # For the repair step, we just need to find A violation.
        # If we don't find a K4, we assume no K4? 
        # No, we must be exact.
        
        # Let's do a more structured search for K6 in comp.
        # Since n is small, we can iterate.
        # But 1.9M iterations per step is too slow if we do it 1000 times.
        # 1.9M * 1000 = 1.9B operations. Too slow in Python.
        
        # Optimization:
        # We can check for K6 by iterating pairs (u,v) in comp.
        # common = comp[u] & comp[v]
        # If common has size >= 4, check if any 4 of them form a K4 in comp?
        # This is getting complex.
        
        # Alternative: Use a fast clique finding algorithm.
        # Or, just check K4 in G and K6 in G_bar.
        # For K6 in G_bar, we can use the same logic as K4 but on comp.
        # Wait, K6 is larger.
        
        # Let's try a different approach for the check.
        # For the purpose of *finding* a violation to repair:
        # We can randomly sample 6 vertices. If they are independent, return them.
        # Probability of hitting an independent set if one exists?
        # If there is only 1 independent set of size 6, and C(36,6)=1.9M, 
        # probability is 1/1.9M per sample.
        # We need ~1.9M samples to find it with 63% probability.
        # Too slow.
        
        # However, if the graph is "almost" valid, there might be few violations.
        # Or many.
        
        # Let's stick to exact check but optimize.
        # For n=36, we can precompute all 6-subsets? No, memory.
        
        # Let's use a recursive backtracking for K6 in comp.
        # It prunes quickly if the graph is sparse (comp is dense if G is sparse).
        # G is K4-free, so G is sparse. Comp is dense.
        # Finding K6 in a dense graph is easy.
        
        def find_k6_in_comp():
            # Backtracking
            def backtrack(candidates, k, path):
                if k == 0:
                    return path
                if not candidates:
                    return None
                # Heuristic: pick vertex with max degree in induced subgraph?
                # Or just iterate.
                for i in range(len(candidates)):
                    v = candidates[i]
                    # New candidates: neighbors of v in comp, that are in candidates and > v?
                    # To avoid permutations, maintain order.
                    new_cands = [c for c in candidates[i+1:] if (comp[v] >> c) & 1]
                    res = backtrack(new_cands, k-1, path + [v])
                    if res:
                        return res
                return None
            
            return backtrack(list(range(n)), 6, [])

        return find_k6_in_comp()

    def local_search(n, time_budget):
        deadline_ls = time.time() + time_budget
        adj = [0] * n
        mask = (1 << n) - 1
        
        # Initialize random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.3: # Rough density estimate
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        while time.time() < deadline_ls:
            # Check K4
            k4 = check_k4(n, adj)
            if k4:
                u, v, w = k4
                # Flip a random edge in the K4
                # K4 vertices are u, v, w, x. We found u, v, w and an edge in common.
                # Let's just pick a random edge from the triangle u,v,w and flip it?
                # Actually, we need the 4th vertex.
                # My check_k4 returns u, v, w where w is in common(u,v) and connected to another in common.
                # Let's refine check_k4 to return all 4.
                
                # For now, just flip edge (u,v) or (u,w) or (v,w)
                if random.random() < 0.5:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] ^= (1 << w)
                    adj[w] ^= (1 << u)
                continue
            
            # Check Ind 6
            ind6 = check_ind6(n, adj)
            if ind6:
                # Flip a random edge within the independent set
                # ind6 is a list of 6 vertices
                u, v = random.sample(ind6, 2)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                continue
            
            # Valid!
            return adj
        
        return None

    best_n = 0
    best_adj = None
    
    # Try 36
    adj36 = local_search(36, 30)
    if adj36:
        best_n = 36
        best_adj = adj36
    else:
        # Try 35
        adj35 = local_search(35, 60)
        if adj35:
            best_n = 35
            best_adj = adj35
        else:
            # Try 34
            adj34 = local_search(34, 20)
            if adj34:
                best_n = 34
                best_adj = adj34
            else:
                # Fallback to 17 Paley
                squares = {x * x % 17 for x in range(1, 17)}
                best_n = 17
                best_adj = [0]*17
                for i in range(17):
                    for j in range(i+1, 17):
                        if (j-i)%17 in squares:
                            best_adj[i] |= (1<<j)
                            best_adj[j] |= (1<<i)

    # Convert bitsets to edges
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': best_n, 'edges': edges}
