import numpy as np
import time
import random

def _has_c6_bitmask(adj, n):
    """
    Check if adjacency (list of int bitmasks) contains a C6.
    adj[i] is a bitmask of neighbors of i.
    """
    # To speed up, we can iterate over edges (v0, v1) and check for paths of length 4 from v1 to v0
    # that don't touch v0 or v1 internally.
    # Path: v0-v1-v2-v3-v4-v5-v0
    # Fix v0, v1.
    # v5 must be in adj[v0] \ {v1}.
    # We need a path v1-v2-v3-v4-v5 of length 4.
    # v2 in adj[v1] \ {v0, v5}
    # v3 in adj[v2] \ {v0, v1, v5}
    # v4 in adj[v3] \ {v0, v1, v2, v5}
    # Check if v5 in adj[v4]
    
    # Optimization: 
    # For each v0:
    #   For each v1 in adj[v0]:
    #     If v1 <= v0: continue (to avoid symmetry? No, bitmask doesn't care, but we can restrict v1 > v0 to halve work? 
    #     Actually, the cycle is undirected. We can fix v0 and v1 < v0? No, just iterate all.
    
    for v0 in range(n):
        # Candidates for v5
        # We need to find a path of length 4 from v1 to v5 in the graph induced by V \ {v0}
        # Let's iterate v1 in adj[v0]
        v1_mask = adj[v0]
        # We will iterate over v1
        # To be efficient, we can precompute something?
        
        # Let's try:
        # For each v1 in adj[v0]:
        #   For each v5 in adj[v0] \ {v1}:
        #     Check if there is a path of length 3 from v1 to v5 in G \ {v0, v1, v5}?
        #     Path v1-v2-v3-v4-v5.
        #     v2 in adj[v1] \ {v0, v5}
        #     v3 in adj[v2] \ {v0, v1, v5}
        #     v4 in adj[v3] \ {v0, v1, v2, v5}
        #     Check if v5 in adj[v4]
        
        # This is O(n^4) per edge? No, O(n^3) per (v0, v1) pair?
        # Total O(n^5) which is ~10^6 for n=17. Acceptable.
        
        for v1 in range(n):
            if not (v1_mask >> v1) & 1: continue
            if v1 == v0: continue
            
            # We need v5 such that v5 in adj[v0] and v5 != v1
            # And path v1->...->v5 length 4.
            # Let's iterate v5
            v5_mask = adj[v0] & ~(1 << v1)
            # Iterate v5
            temp_mask = v5_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v5 = lsb.bit_length() - 1
                temp_mask ^= lsb
                
                # Check path v1-v2-v3-v4-v5
                # v2 in adj[v1] \ {v0, v5}
                v2_mask = adj[v1] & ~(1 << v0) & ~(1 << v5)
                temp_v2 = v2_mask
                while temp_v2:
                    lsb_v2 = temp_v2 & -temp_v2
                    v2 = lsb_v2.bit_length() - 1
                    temp_v2 ^= lsb_v2
                    
                    # v3 in adj[v2] \ {v0, v1, v5}
                    v3_mask = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    temp_v3 = v3_mask
                    while temp_v3:
                        lsb_v3 = temp_v3 & -temp_v3
                        v3 = lsb_v3.bit_length() - 1
                        temp_v3 ^= lsb_v3
                        
                        # v4 in adj[v3] \ {v0, v1, v2, v5}
                        v4_mask = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        temp_v4 = v4_mask
                        while temp_v4:
                            lsb_v4 = temp_v4 & -temp_v4
                            v4 = lsb_v4.bit_length() - 1
                            temp_v4 ^= lsb_v4
                            
                            # Check if v5 in adj[v4]
                            if (adj[v4] >> v5) & 1:
                                return True
    return False

def _find_c6_bitmask(adj, n):
    """
    Find a C6 in the graph represented by adj (bitmasks).
    Returns list of vertices [v0, v1, v2, v3, v4, v5] or None.
    """
    for v0 in range(n):
        v1_mask = adj[v0]
        for v1 in range(n):
            if not (v1_mask >> v1) & 1: continue
            if v1 == v0: continue
            
            v5_mask = adj[v0] & ~(1 << v1)
            temp_v5 = v5_mask
            while temp_v5:
                lsb_v5 = temp_v5 & -temp_v5
                v5 = lsb_v5.bit_length() - 1
                temp_v5 ^= lsb_v5
                
                v2_mask = adj[v1] & ~(1 << v0) & ~(1 << v5)
                temp_v2 = v2_mask
                while temp_v2:
                    lsb_v2 = temp_v2 & -temp_v2
                    v2 = lsb_v2.bit_length() - 1
                    temp_v2 ^= lsb_v2
                    
                    v3_mask = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    temp_v3 = v3_mask
                    while temp_v3:
                        lsb_v3 = temp_v3 & -temp_v3
                        v3 = lsb_v3.bit_length() - 1
                        temp_v3 ^= lsb_v3
                        
                        v4_mask = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        temp_v4 = v4_mask
                        while temp_v4:
                            lsb_v4 = temp_v4 & -temp_v4
                            v4 = lsb_v4.bit_length() - 1
                            temp_v4 ^= lsb_v4
                            
                            if (adj[v4] >> v5) & 1:
                                return [v0, v1, v2, v3, v4, v5]
    return None

def _build_adj_bitmask(edge_colors, n):
    """Build adjacency bitmasks for each color class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _is_valid_coloring(edge_colors, n):
    adj = _build_adj_bitmask(edge_colors, n)
    for c in range(4):
        if _has_c6_bitmask(adj[c], n):
            return False
    return True

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random regular graph
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    # Simple random regular graph generation
    adj = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
    
    np.fill_diagonal(adj, 0)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    # Take top 4 non-zero eigenvectors
    vecs = eigvecs[:, 1:5]
    
    # Project vertices to R^4
    points = vecs.T # Shape (4, n)
    
    # Normalize
    norms = np.linalg.norm(points, axis=0, keepdims=True)
    norms[norms == 0] = 1
    points = points / norms
    
    # Assign vertices to 4 clusters based on points
    # Use a simple heuristic: project to 1D using random vector and split into quartiles
    # Or use the 4 coordinates to define a "type"
    # Let's try: For each vertex, look at the 4 coordinates.
    # Sort them. The rank of the largest coordinate determines the cluster?
    # Or just use the sign of the first 2 coordinates to make 4 quadrants?
    # Quadrants: (x>0, y>0) -> 0, (x<0, y>0) -> 1, etc.
    # This gives 4 clusters.
    
    clusters = []
    for i in range(n):
        x = points[0, i]
        y = points[1, i]
        if x > 0 and y > 0:
            clusters.append(0)
        elif x < 0 and y > 0:
            clusters.append(1)
        elif x < 0 and y < 0:
            clusters.append(2)
        else:
            clusters.append(3)
            
    # Now map cluster pairs to colors
    # We have 4 clusters. Pairs: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    # 10 pairs. Map to 4 colors.
    # Random mapping
    mapping = {}
    pair_idx = 0
    for u in range(4):
        for v in range(u, 4):
            mapping[(u, v)] = pair_idx % 4
            pair_idx += 1
            
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            cu = clusters[i]
            cv = clusters[j]
            if cu > cv: cu, cv = cv, cu
            color = mapping[(cu, cv)]
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=10000, time_limit=10.0):
    """
    Local search to remove C6s.
    """
    colors = list(edge_colors)
    start_time = time.time()
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    adjs = _build_adj_bitmask(colors, n)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Find a C6 in any color class
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = _find_c6_bitmask(adjs[c], n)
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors # Valid!
            
        # Recolor an edge in the cycle
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        u, v = random.choice(edges_in_cycle)
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try a new color
        # Prefer colors that are less likely to create C6s?
        # Just pick a random different color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        # Tentatively change
        colors[idx] = new_color
        
        # Update adjs
        adjs[old_color][u] &= ~(1 << v)
        adjs[old_color][v] &= ~(1 << u)
        adjs[new_color][u] |= (1 << v)
        adjs[new_color][v] |= (1 << u)
        
        # Check if new_color has a C6 involving u or v?
        # We can just check if the new edge is part of a C6 in new_color
        # A C6 containing (u,v) in new_color:
        # u-v-w1-w2-w3-w4-u
        # Path v -> w1 -> w2 -> w3 -> w4 -> u of length 5? No, length 4 internal.
        # v-w1-w2-w3-w4-u.
        # w1 in adj_new[v] \ {u}
        # w2 in adj_new[w1] \ {u, v}
        # w3 in adj_new[w2] \ {u, v, w1}
        # w4 in adj_new[w3] \ {u, v, w1, w2}
        # Check if u in adj_new[w4]
        
        def has_c6_through_edge(adj, u, v):
            v_mask = adj[v] & ~(1 << u)
            temp_v1 = v_mask
            while temp_v1:
                lsb_v1 = temp_v1 & -temp_v1
                w1 = lsb_v1.bit_length() - 1
                temp_v1 ^= lsb_v1
                
                w2_mask = adj[w1] & ~(1 << u) & ~(1 << v)
                temp_w2 = w2_mask
                while temp_w2:
                    lsb_w2 = temp_w2 & -temp_w2
                    w2 = lsb_w2.bit_length() - 1
                    temp_w2 ^= lsb_w2
                    
                    w3_mask = adj[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
                    temp_w3 = w3_mask
                    while temp_w3:
                        lsb_w3 = temp_w3 & -temp_w3
                        w3 = lsb_w3.bit_length() - 1
                        temp_w3 ^= lsb_w3
                        
                        w4_mask = adj[w3] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2)
                        temp_w4 = w4_mask
                        while temp_w4:
                            lsb_w4 = temp_w4 & -temp_w4
                            w4 = lsb_w4.bit_length() - 1
                            temp_w4 ^= lsb_w4
                            
                            if (adj[w4] >> u) & 1:
                                return True
            return False

        if has_c6_through_edge(adjs[new_color], u, v):
            # Revert
            colors[idx] = old_color
            adjs[new_color][u] &= ~(1 << v)
            adjs[new_color][v] &= ~(1 << u)
            adjs[old_color][u] |= (1 << v)
            adjs[old_color][v] |= (1 << u)
        else:
            # Accept
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n=17 first
    for n in range(17, 9, -1):
        found = False
        # Try multiple seeds
        for seed in range(100):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=5000, time_limit=5.0)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback
    for n in range(9, 1, -1):
        for seed in range(100):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=5000, time_limit=5.0)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
