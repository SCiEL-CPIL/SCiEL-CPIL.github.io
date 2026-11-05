"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Base: Paley(29)
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Generate colors in lexicographic order (i, j) with i < j
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    # Try to extend to n=30
    n = 30
    num_edges = n * (n - 1) // 2
    
    # Build edge index map
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
            
    # Reverse map for quick access
    idx_edge = [None] * num_edges
    for (i, j), idx_val in edge_idx.items():
        idx_edge[idx_val] = (i, j)

    def build_adj(colors_list):
        """Build adjacency bitmasks for each vertex and color."""
        adj = [[0, 0] for _ in range(n)]
        for idx_val, c in enumerate(colors_list):
            i, j = idx_edge[idx_val]
            adj[i][c] |= (1 << j)
            adj[j][c] |= (1 << i)
        return adj

    def has_k222(adj):
        """Fast check for K_{2,2,2} in either color."""
        for c in range(2):
            # Precompute common neighborhoods for all pairs? 
            # Or just on the fly.
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if (u,v) is color c
                    if not (adj[u][c] & (1 << v)):
                        continue
                    S = adj[u][c] & adj[v][c]
                    # Check if subgraph induced by S in color c has K_{2,2}
                    # K_{2,2} exists if there are a,b in S such that |N_c(a,b) & S| >= 2
                    # Iterate pairs in S
                    # Extract vertices in S
                    # To speed up, we can just iterate all pairs (a,b) in 0..n-1 
                    # and check if they are in S.
                    # But iterating all n^2 pairs is 900, which is fine.
                    # However, we only care about pairs in S.
                    # Let's iterate a, b in range(n)
                    # If a in S and b in S and a < b:
                    #   T = adj[a][c] & adj[b][c] & S
                    #   if bit_count(T) >= 2: return True
                    
                    # Optimization:
                    # We can iterate a in S, b in S.
                    # Using bit operations to iterate set bits in S.
                    mask = S
                    while mask:
                        lb = mask & (-mask)
                        a = lb.bit_length() - 1
                        mask ^= lb
                        # For each a in S, check all b in S with b > a?
                        # No, we just need any pair.
                        # Let's iterate b in S, b != a.
                        mask_b = S
                        while mask_b:
                            lb_b = mask_b & (-mask_b)
                            b = lb_b.bit_length() - 1
                            mask_b ^= lb_b
                            if a == b:
                                continue
                            # Check common neighbors in S
                            T = adj[a][c] & adj[b][c] & S
                            if T & (T - 1): # If at least 2 bits set
                                return True
        return False

    best_n = 29
    best_colors = base_colors[:]

    # Local search for n=30
    # We initialize the edges of vertex 29 randomly or heuristically
    # and then flip edges to remove violations.
    
    # Heuristic initialization: 
    # Since Paley(29) is self-complementary, maybe a balanced coloring for the new vertex is good.
    # Let's try a few different initial patterns for the 29 edges incident to vertex 29.
    
    patterns = [
        [0]*29,
        [1]*29,
        [i % 2 for i in range(29)],
        [random.randint(0,1) for _ in range(29)]
    ]
    
    # Add more random patterns
    for _ in range(10):
        patterns.append([random.randint(0,1) for _ in range(29)])

    for pattern in patterns:
        if time.time() > deadline:
            break
            
        colors_list = base_colors[:] + pattern
        
        adj = build_adj(colors_list)
        
        # Local search
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            if not has_k222(adj):
                best_n = 30
                best_colors = colors_list[:]
                break
                
            # Find a violation and flip an edge in it
            # We need a version of has_k222 that returns the violating edge
            # For now, let's just flip a random edge if we don't have a specific one.
            # But targeted is better.
            
            # Let's implement a function to find a violating K222
            # This will be slow if done naively, but we can reuse the logic.
            
            found_violation = False
            for c in range(2):
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (adj[u][c] & (1 << v)):
                            continue
                        S = adj[u][c] & adj[v][c]
                        mask = S
                        while mask and not found_violation:
                            lb = mask & (-mask)
                            a = lb.bit_length() - 1
                            mask ^= lb
                            mask_b = S
                            while mask_b and not found_violation:
                                lb_b = mask_b & (-mask_b)
                                b = lb_b.bit_length() - 1
                                mask_b ^= lb_b
                                if a == b:
                                    continue
                                T = adj[a][c] & adj[b][c] & S
                                if T & (T - 1):
                                    # Found K222 in S induced by a,b
                                    # The K222 vertices are u,v,a,b and two from T
                                    # Let's pick two from T
                                    t_mask = T
                                    lb_t1 = t_mask & (-t_mask)
                                    t1 = lb_t1.bit_length() - 1
                                    t_mask ^= lb_t1
                                    lb_t2 = t_mask & (-t_mask)
                                    t2 = lb_t2.bit_length() - 1
                                    
                                    # Vertices: u, v, a, b, t1, t2
                                    # Edges in the K222:
                                    # (u,v), (u,a), (u,b), (u,t1), (u,t2)
                                    # (v,a), (v,b), (v,t1), (v,t2)
                                    # (a,b), (a,t1), (a,t2)
                                    # (b,t1), (b,t2)
                                    # (t1,t2)
                                    
                                    # Flip a random edge among these
                                    # We need the edge index
                                    # Let's just flip (u,v) or (a,b)
                                    # Flipping (u,v) changes color of (u,v)
                                    # This might break the condition that (u,v) is color c,
                                    # which might remove the violation or move it.
                                    
                                    # Let's flip (a,b)
                                    i, j = (a, b) if a < b else (b, a)
                                    e_idx = edge_idx[(i, j)]
                                    colors_list[e_idx] ^= 1
                                    adj = build_adj(colors_list)
                                    found_violation = True
                                    break
            if not found_violation:
                # Should not happen if has_k222 returned True
                # But if it does, flip random
                e = random.randint(0, num_edges - 1)
                colors_list[e] ^= 1
                adj = build_adj(colors_list)

    if best_n == 30:
        return {'n': 30, 'colors': best_colors}
    else:
        return {'n': 29, 'colors': base_colors}
# EVOLVE-BLOCK-END
