def construct():
    import random
    import time

    n = 18
    random.seed(42)

    # Precompute triangle masks
    # A triangle is (i, j, k) with i<j<k and all edges present.
    # But edges change. So we can't precompute triangles for the graph.
    # We CAN precompute which subsets of vertices form a triangle in a COMPLETE graph? No.
    
    # Fast Evaluator:
    # Given adj_masks, find max size of triangle-free induced subgraph.
    # Use branch and bound but optimized.
    
    def max_triangle_free(adj_masks):
        best = 0
        
        def has_edge_in_mask(mask):
            m = mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                if adj_masks[v] & (mask ^ v_bit):
                    return True
                m ^= v_bit
            return False

        def backtrack(idx, mask, count):
            nonlocal best
            if count > best:
                best = count
            if idx == n:
                return
            
            # Prune
            if count + (n - idx) <= best:
                return
            
            # Option 1: Skip vertex idx
            backtrack(idx + 1, mask, count)
            
            # Option 2: Include vertex idx
            N = mask & adj_masks[idx]
            if not has_edge_in_mask(N):
                backtrack(idx + 1, mask | (1 << idx), count + 1)

        backtrack(0, 0, 0)
        return best

    def generate_random_k4_free():
        adj = [[False]*n for _ in range(n)]
        adj_masks = [0]*n
        
        # Helper to check if adding edge (u,v) creates a K4
        def creates_k4(u, v):
            # Common neighbors of u and v
            common = adj_masks[u] & adj_masks[v]
            # Check if any two common neighbors are connected
            m = common
            while m:
                w_bit = m & -m
                w = w_bit.bit_length() - 1
                if adj_masks[w] & (common ^ w_bit):
                    return True
                m ^= w_bit
            return False

        # Add edges in random order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if not creates_k4(u, v):
                adj[u][v] = True
                adj[v][u] = True
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
        
        # Convert to edge list
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edge_list.append([i, j])
        
        return edge_list, adj_masks

    best_edges = None
    best_score = 999

    # Try many random instances
    start_time = time.time()
    limit = 100  # Adjust based on time
    
    for i in range(200):
        if time.time() - start_time > 100:
            break
        edges, adj_masks = generate_random_k4_free()
        score = max_triangle_free(adj_masks)
        if score < best_score:
            best_score = score
            best_edges = edges
            # print(f"Found {score} at iter {i}")

    # Fallback to Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    t_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    t_edges.append([a, b])
    
    adj_t = [[False]*n for _ in range(n)]
    for u, v in t_edges:
        adj_t[u][v] = True
        adj_t[v][u] = True
    adj_masks_t = [0]*n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj_t[i][j]:
                mask |= (1 << j)
        adj_masks_t[i] = mask
        
    score_t = max_triangle_free(adj_masks_t)
    if score_t < best_score:
        best_score = score_t
        best_edges = t_edges

    return best_edges
