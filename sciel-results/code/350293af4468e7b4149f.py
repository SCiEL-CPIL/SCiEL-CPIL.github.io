import random
import math
import time

def construct():
    n = 18
    all_mask = (1 << n) - 1
    
    def creates_k4(adj_mask, u, v):
        # Check if adding edge (u,v) creates a K4.
        # This happens if there exist x, y such that {u,v,x,y} is a K4.
        # This requires x, y in N(u) & N(v) and edge (x,y) exists.
        common = adj_mask[u] & adj_mask[v]
        # Check if induced subgraph on 'common' has any edge.
        # Iterate through bits in common.
        c = common
        while c:
            lsb = c & -c
            w_idx = lsb.bit_length() - 1
            # If w_idx is connected to any other vertex in common, edge exists.
            # adj_mask[w_idx] & common will contain w_idx's neighbors in common.
            # Since w_idx is in common, we check if there is ANY neighbor in common.
            # Note: adj_mask[w_idx] does not have bit w_idx set.
            if adj_mask[w_idx] & common:
                return True
            c ^= lsb
        return False

    def max_triangle_free_size(adj_mask):
        best_size = 0
        
        # Precompute common neighbors for speed in B&B
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u][v] = adj_mask[u] & adj_mask[v]
        
        def dfs(c_mask, p_mask):
            nonlocal best_size
            # Pruning
            if c_mask.bit_count() + p_mask.bit_count() <= best_size:
                return
            
            if p_mask == 0:
                if c_mask.bit_count() > best_size:
                    best_size = c_mask.bit_count()
                return
            
            # Heuristic: Pick vertex with highest degree in p_mask? 
            # Or just lowest bit for simplicity but maybe slower.
            # Let's try picking the vertex that removes the most from p_mask if included?
            # This is expensive to compute for all.
            # Let's stick to lowest bit for now, but optimize the conflict calculation.
            
            v_lsb = p_mask & -p_mask
            v_idx = v_lsb.bit_length() - 1
            p_mask_without_v = p_mask ^ v_lsb
            
            # Option 1: Include v
            # Calculate invalid mask
            invalid = 0
            c = c_mask
            while c:
                lsb_u = c & -c
                u_idx = lsb_u.bit_length() - 1
                # w is invalid if w in p_mask and w connected to v and u connected to v and u connected to w?
                # Triangle (v, u, w).
                # u in c_mask.
                # We need edge (v,u), edge (v,w), edge (u,w).
                # We know u is in c_mask.
                # If edge (v,u) exists, then w must be in N(v) & N(u).
                # So w in common[v_idx][u_idx].
                if adj_mask[u_idx] & v_lsb:
                    invalid |= common[v_idx][u_idx]
                c ^= lsb_u
            
            new_p_mask = p_mask_without_v & ~invalid
            dfs(c_mask | v_lsb, new_p_mask)
            
            # Option 2: Exclude v
            dfs(c_mask, p_mask_without_v)

        dfs(0, all_mask)
        return best_size

    random.seed(42)
    
    # Initialize with Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj_mask = [0]*n
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
    
    best_adj = adj_mask[:]
    best_score = max_triangle_free_size(best_adj)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    T_start = 10.0
    T_end = 0.01
    iterations = 10000 # Increase iterations since check is faster
    
    start_time = time.time()
    
    for i in range(iterations):
        if time.time() - start_time > 110: # Leave buffer
            break
            
        T = T_start * (T_end / T_start) ** (i / iterations)
        
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        bit_u = 1 << v
        bit_v = 1 << u
        
        is_edge = (adj_mask[u] & bit_u) != 0
        
        # Flip edge
        adj_mask[u] ^= bit_u
        adj_mask[v] ^= bit_v
        
        if is_edge:
            # We removed an edge. K4-free is maintained.
            pass
        else:
            # We added an edge. Check if it creates K4.
            if creates_k4(adj_mask, u, v):
                # Revert
                adj_mask[u] ^= bit_u
                adj_mask[v] ^= bit_v
                continue
        
        new_score = max_triangle_free_size(adj_mask)
        
        if new_score < current_score:
            current_score = new_score
            current_adj = adj_mask[:]
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                    current_adj = adj_mask[:]
                else:
                    # Revert
                    adj_mask[u] ^= bit_u
                    adj_mask[v] ^= bit_v
            else:
                current_score = new_score
                current_adj = adj_mask[:]
                
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
