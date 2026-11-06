import random
import math
import time
import sys

def construct():
    n = 18
    all_mask = (1 << n) - 1
    
    # Precompute bit positions
    bits = [1 << i for i in range(n)]
    
    def is_k4_free(adj_mask):
        # Check for K4
        # For each edge (u, v), check if common neighbors contain an edge.
        for u in range(n):
            mu = adj_mask[u]
            v = 0
            while v < n:
                if mu & bits[v]:
                    common = mu & adj_mask[v]
                    # Check if common has an edge
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if adj_mask[w] & common:
                            return False
                        c ^= lsb
                v += 1
        return True

    def max_triangle_free_size(adj_mask):
        # B&B for Max Triangle-Free Set
        best_size = 0
        
        # Precompute common neighbors for all pairs
        # common[u][v] is mask of common neighbors
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u][v] = adj_mask[u] & adj_mask[v]
        
        # Greedy lower bound initialization
        # Try to build a TF set greedily
        def greedy_tf_size():
            current = 0
            pool = all_mask
            while pool:
                # Pick vertex with min degree in induced subgraph of pool?
                # Or just first one
                # To be fast, just pick lowest set bit
                lsb = pool & -pool
                v = lsb.bit_length() - 1
                # Check if v can be added
                # v is compatible if no triangle with current
                # Since we add in order, we just need to check against current set
                # But greedy doesn't track "current set" as a mask for fast triangle check easily
                # Let's just return a simple bound: independence number lower bound?
                # Actually, let's just use 0 as initial best and let B&B find it.
                # Or use a quick heuristic: 
                # Pick random vertices, check if TF.
                pass
            return 0

        # Better: Use a simple heuristic to set initial best_size
        # Random sampling
        for _ in range(50):
            sample = 0
            for i in range(n):
                if random.random() < 0.5:
                    sample |= bits[i]
            # Check if sample is TF
            is_tf = True
            c = sample
            while c:
                lsb1 = c & -c
                u = lsb1.bit_length() - 1
                c ^= lsb1
                # Check edges in sample
                m = sample & adj_mask[u]
                while m:
                    lsb2 = m & -m
                    v = lsb2.bit_length() - 1
                    # If u-v edge, check if they have common neighbor in sample
                    if common[u][v] & sample:
                        is_tf = False
                        break
                    m ^= lsb2
                if not is_tf:
                    break
            if is_tf:
                best_size = max(best_size, bin(sample).count('1'))
        
        def dfs(c_mask, p_mask):
            nonlocal best_size
            
            # Pruning
            # Upper bound: size of c_mask + size of p_mask
            if bin(c_mask).count('1') + bin(p_mask).count('1') <= best_size:
                return
            
            if p_mask == 0:
                if bin(c_mask).count('1') > best_size:
                    best_size = bin(c_mask).count('1')
                return
            
            # Heuristic: Pick vertex with fewest "conflicts" or highest degree?
            # Let's pick the vertex in p_mask with the highest degree in the "conflict graph" 
            # where conflict means "forms triangle with c_mask and another vertex"?
            # Simpler: Pick vertex with lowest index in p_mask.
            # Or: Pick vertex that maximizes the reduction of p_mask?
            
            # Let's just pick the first one for now, but we can improve.
            v_lsb = p_mask & -p_mask
            v_idx = v_lsb.bit_length() - 1
            
            # Option 1: Include v
            # Calculate new p_mask
            invalid = 0
            c = c_mask
            while c:
                lsb_u = c & -c
                u_idx = lsb_u.bit_length() - 1
                if adj_mask[u_idx] & v_lsb:
                    invalid |= common[v_idx][u_idx]
                c ^= lsb_u
            
            new_c_mask = c_mask | v_lsb
            new_p_mask = p_mask & ~invalid & ~v_lsb
            
            # Recurse
            dfs(new_c_mask, new_p_mask)
            
            # Option 2: Exclude v
            dfs(c_mask, p_mask & ~v_lsb)

        dfs(0, all_mask)
        return best_size

    # --- Main Search: Iterated Local Search ---
    
    def random_k4_free_graph():
        adj = [0]*n
        # Start with empty, add edges randomly if no K4
        # This might be slow to generate dense graphs.
        # Better: Start with Turan T(18,3) and randomize
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        if random.random() < 0.8: # Keep 80% of Turan edges
                            adj[u] |= bits[v]
                            adj[v] |= bits[u]
        # Remove any K4s by removing random edges from K4s
        # Simple approach: While not K4-free, find K4, remove random edge
        # But finding K4 is O(N^4).
        # Let's just use the Turan graph as a base and flip edges.
        return adj

    def get_best_adj():
        # Start with Turan
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj = [0]*n
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= bits[v]
                        adj[v] |= bits[u]
        return adj

    best_adj = get_best_adj()
    best_score = max_triangle_free_size(best_adj)
    
    time_limit = 110.0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        current_adj = best_adj[:]
        current_score = best_score
        
        # Local Search
        improved = True
        while improved:
            improved = False
            # Try all single edge flips
            for u in range(n):
                for v in range(u+1, n):
                    bit_u = bits[v]
                    bit_v = bits[u]
                    
                    # Flip
                    current_adj[u] ^= bit_u
                    current_adj[v] ^= bit_v
                    
                    if not is_k4_free(current_adj):
                        current_adj[u] ^= bit_u
                        current_adj[v] ^= bit_v
                        continue
                    
                    new_score = max_triangle_free_size(current_adj)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # If we found an improvement, we can break or continue
                        # Breaking and restarting scan is safer for "First-Best"
                        # But "Best-First" is better.
                        # Let's collect best in this pass.
                        # To keep it simple and fast, let's just accept first improvement
                        # and restart the outer loop of local search.
                        # Actually, let's just break out of the inner loops to restart scan
                        # from scratch to avoid missing better moves later.
                        break
                    else:
                        # Revert
                        current_adj[u] ^= bit_u
                        current_adj[v] ^= bit_v
                
                if improved:
                    break
            
            if improved:
                # Update best if better
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj[:]
        
        # If no improvement, Perturb
        # Randomly flip 3-5 edges
        num_flips = random.randint(3, 5)
        for _ in range(num_flips):
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            bit_u = bits[v]
            bit_v = bits[u]
            
            current_adj[u] ^= bit_u
            current_adj[v] ^= bit_v
            
            if not is_k4_free(current_adj):
                current_adj[u] ^= bit_u
                current_adj[v] ^= bit_v
            else:
                # Accept perturbation
                pass
        
        # Evaluate perturbed state
        perturbed_score = max_triangle_free_size(current_adj)
        if perturbed_score < best_score:
            best_score = perturbed_score
            best_adj = current_adj[:]
            
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        m = best_adj[u]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
            
    return edges
