def construct():
    import random

    n = 18
    random.seed(42)

    def max_triangle_free(adj_masks):
        best = 0
        
        def has_edge_in_mask(mask):
            m = mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                # Check if v has any neighbor in mask excluding v
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
            # Check if adding idx creates a triangle
            # Triangle if there are u, w in mask such that u~idx, w~idx, u~w
            # Let N = mask & adj_masks[idx]
            # If N contains an edge, then triangle exists.
            N = mask & adj_masks[idx]
            if not has_edge_in_mask(N):
                backtrack(idx + 1, mask | (1 << idx), count + 1)

        backtrack(0, 0, 0)
        return best

    def generate_k4_free_graph(p):
        # Start with random graph
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Prune K4s
        while True:
            k4_found = False
            # Find a K4
            for a in range(n):
                if k4_found: break
                for b in range(a+1, n):
                    if k4_found: break
                    if not adj[a][b]: continue
                    for c in range(b+1, n):
                        if k4_found: break
                        if not adj[a][c] or not adj[b][c]: continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4
                                k4_found = True
                                # Remove random edge
                                edges_to_break = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                u, v = random.choice(edges_to_break)
                                adj[u][v] = False
                                adj[v][u] = False
                                break
            if not k4_found:
                break
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        
        # Convert to bitmask adj for evaluation
        adj_masks = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
            
        return edges, adj_masks

    best_edges = None
    best_score = 999

    # Try a few seeds/densities
    for p in [0.4, 0.5, 0.6]:
        for trial in range(20):
            edges, adj_masks = generate_k4_free_graph(p)
            score = max_triangle_free(adj_masks)
            if score < best_score:
                best_score = score
                best_edges = edges

    # Fallback to Turan T(18,3) if random failed or for comparison
    # T(18,3) parts: 0-5, 6-11, 12-17
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    t_edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    t_edges.append([a, b])
    
    # Evaluate Turan
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
