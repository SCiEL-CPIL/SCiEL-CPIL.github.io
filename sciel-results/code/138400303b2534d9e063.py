def construct():
    # Helper to check triangle-free using adjacency matrix
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Helper to calculate independence number using bitset branch and bound
    def calc_alpha(n, edges):
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Precompute non-adjacency masks (including self? No, candidates exclude self)
        # non_adj_mask[v] has bits set for u where u is NOT adjacent to v
        non_adj_mask = [0] * n
        for v in range(n):
            mask = (1 << n) - 1
            mask &= ~(1 << v) # exclude self
            mask &= ~adj_mask[v] # exclude neighbors
            non_adj_mask[v] = mask
            
        best = 0
        
        def dfs(candidates_mask, current_size):
            nonlocal best
            if candidates_mask == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning
            if current_size + bin(candidates_mask).count('1') <= best:
                return
            
            # Pick a vertex
            # Heuristic: pick vertex with highest degree in candidates?
            # For speed, just pick lowest set bit
            lsb = candidates_mask & -candidates_mask
            v = (lsb.bit_length() - 1)
            rest = candidates_mask ^ lsb
            
            # Option 1: Include v
            # New candidates are those in rest not adjacent to v
            new_cands = rest & non_adj_mask[v]
            dfs(new_cands, current_size + 1)
            
            # Option 2: Exclude v
            dfs(rest, current_size)

        dfs((1 << n) - 1, 0)
        return best

    # Helper to construct Andrásfai graph And(3k+1)
    def andrasfai_edges(n):
        # n must be 3k+1
        k = (n - 1) // 3
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                # Edge if diff mod 3 == 1
                if diff % 3 == 1:
                    edges.append([i, j])
        return edges

    # Helper to construct a maximal triangle-free graph greedily
    def greedy_triangle_free(n):
        adj = [[False]*n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                # Check if adding (u,v) creates a triangle
                # Triangle exists if there is w such that adj[u][w] and adj[v][w]
                conflict = False
                for w in range(n):
                    if w != u and w != v:
                        if adj[u][w] and adj[v][w]:
                            conflict = True
                            break
                if not conflict:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append([u, v])
        return edges

    # Try Andrásfai graphs for largest n first
    # n = 3k+1
    for k in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        n = 3 * k + 1
        if n > 41:
            continue
        edges = andrasfai_edges(n)
        if is_triangle_free(n, edges):
            alpha_val = calc_alpha(n, edges)
            if alpha_val < 10:
                return {'n': n, 'edges': edges}

    # Try greedy construction for n from 40 down to 20
    for n in range(40, 19, -1):
        edges = greedy_triangle_free(n)
        if is_triangle_free(n, edges):
            alpha_val = calc_alpha(n, edges)
            if alpha_val < 10:
                return {'n': n, 'edges': edges}

    # Fallback: Return a small valid graph (e.g., n=1)
    return {'n': 1, 'edges': []}
