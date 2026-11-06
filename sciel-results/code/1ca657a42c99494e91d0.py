import numpy as np
import time
import random

def _has_c6(adj, n):
    """
    Check if adjacency (list of sets) contains a C6.
    Optimized for small n.
    """
    # For speed, we can use a bitset representation if n <= 64
    # But for n up to 20, sets are okay if we prune well.
    # A C6 is a cycle of length 6.
    # We look for paths of length 5 that close the loop.
    
    # To speed up, we can iterate over edges (v0, v1) and look for paths of length 4 from v1 to v0 
    # that don't touch v0 or v1 internally, and check if the last vertex connects to v0? No, that's C5 or C6.
    # Path v0-v1-v2-v3-v4-v5-v0.
    # Fix v0, v1.
    # Find v2 in adj[v1] \ {v0}
    # Find v3 in adj[v2] \ {v0, v1}
    # Find v4 in adj[v3] \ {v0, v1, v2}
    # Find v5 in adj[v4] \ {v0, v1, v2, v3}
    # Check if v0 in adj[v5] and v5 != v1 (already ensured) and v5 != v0 (ensured).
    # Also need to ensure v5 is distinct from v2, v3.
    
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0: continue
            for v5 in adj[v0]:
                if v5 == v1 or v5 <= v0: continue
                # We have edges (v0,v1) and (v0,v5).
                # We need a path of length 3 from v1 to v5: v1-v2-v3-v4-v5.
                # v2, v3, v4 must be distinct from v0, v1, v5.
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v5: continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v5: continue
                        for v4 in adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5: continue
                            if v5 in adj[v4]:
                                return True
    return False

def _build_adj_from_edge_colors(edge_colors, n):
    """Build adjacency lists for each color class from edge color list."""
    adj = [set() for _ in range(4)]
    for c in range(4):
        adj[c] = [set() for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj

def _is_valid_coloring(edge_colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    adj = _build_adj_from_edge_colors(edge_colors, n)
    for c in range(4):
        if _has_c6(adj[c], n):
            return False
    return True

def _generate_random_regular_graph(n, d, rng):
    """
    Generate a random d-regular graph on n vertices.
    Returns adjacency matrix.
    """
    if d >= n:
        d = n - 1
    if d % 2 != 0:
        d -= 1 # Must be even for regular graph existence (sum of degrees)
        
    adj = np.zeros((n, n), dtype=int)
    # Simple method: random permutation matching for each degree layer
    # This is not the most efficient but works for n<=20
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
            else:
                # Retry if self-loop or duplicate (should be rare for random perm)
                # For small n, we might fail to achieve exact regularity easily.
                # We will just use the resulting graph, even if slightly irregular, 
                # or force it. For spectral purposes, approximate regularity is fine.
                pass
    
    # Ensure no self loops
    np.fill_diagonal(adj, 0)
    return adj

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random graph, preferably regular
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = _generate_random_regular_graph(n, d, rng)
    
    # Compute Laplacian
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take the 4 smallest non-zero eigenvalues
    # eigvals[0] is 0.
    # We want indices 1, 2, 3, 4
    if n <= 4:
        # Fallback
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    vecs = eigvecs[:, 1:5]
    
    # Normalize vectors to have zero mean (they already do for Laplacian)
    # Project vertices to R^4
    points = vecs.T # Shape (4, n)
    
    # We need to assign colors to edges (i, j).
    # Strategy: For each edge (i, j), look at the vector difference p_j - p_i.
    # Or better: Use the signs of the coordinates of p_i and p_j to determine a "region".
    # A common technique for k-coloring is to partition the space into k regions.
    # Here we have 4 dimensions. We can map each vertex to one of 4 "clusters" based on its position.
    # Then color edges based on cluster pairs? No, that would only give 4 colors if we map pairs to 4 colors.
    # Actually, we want to color edges directly.
    
    # Alternative: Use the 4 eigenvectors to define 4 hyperplanes.
    # Let's try a simple heuristic:
    # For each edge (i, j), compute the dot product of (p_j - p_i) with a fixed set of vectors?
    
    # Let's use a "Sign-based" coloring with random rotation to break symmetry.
    # Rotate the 4D space randomly.
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    
    rotated_points = R @ points
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            # Assign color based on the quadrant in the first 2 dimensions?
            # Or use all 4 dimensions.
            # Let's use a hash of the signs of the 4 components of the difference.
            # 2^4 = 16 signs. Map to 4 colors.
            signs = np.sign(diff)
            # Treat 0 as 1
            signs[signs == 0] = 1
            # Convert to binary
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            # Map 4 bits to 4 colors (0-3)
            color = (b0 * 3 + b1 * 2 + b2 * 1 + b3 * 0) % 4 # Just a mapping
            # Better: color = (b0 + b1 + b2 + b3) % 4 ? No, that's biased.
            # color = (b0 << 1 | b1) % 4 ? Uses only 2 bits.
            
            # Let's use a more balanced mapping.
            # Index = b0*8 + b1*4 + b2*2 + b3
            idx = b0*8 + b1*4 + b2*2 + b3
            # Map 16 indices to 4 colors
            color = idx % 4
            
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=5000):
    """
    Local search to remove C6s.
    """
    # Convert to list for mutability
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    # Precompute adjacency for each color
    def build_adjs(cols):
        adjs = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = cols[get_edge_idx(i, j)]
                adjs[c][i].add(j)
                adjs[c][j].add(i)
        return adjs

    def find_c6(adj_c):
        # Returns a list of vertices in the C6 if found, else None
        for v0 in range(n):
            for v1 in adj_c[v0]:
                if v1 <= v0: continue
                for v5 in adj_c[v0]:
                    if v5 == v1 or v5 <= v0: continue
                    for v2 in adj_c[v1]:
                        if v2 == v0 or v2 == v5: continue
                        for v3 in adj_c[v2]:
                            if v3 == v0 or v3 == v1 or v3 == v5: continue
                            for v4 in adj_c[v3]:
                                if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5: continue
                                if v5 in adj_c[v4]:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    adjs = build_adjs(colors)
    
    for step in range(max_steps):
        # Find a C6 in any color class
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6(adjs[c])
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors # Valid!
            
        # We have a C6 in color c_found: v0-v1-v2-v3-v4-v5-v0
        # Recolor one of the edges to a different color.
        # Pick a random edge from the cycle
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        # Try all edges in the cycle, pick a random one
        u, v = random.choice(edges_in_cycle)
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try a new color different from old_color
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        # Tentatively change color
        colors[idx] = new_color
        
        # Check if this creates a new C6 in new_color
        # We only need to check if the new edge (u,v) is part of a C6 in new_color
        # A C6 containing (u,v) in color new_color must use 4 other edges in new_color.
        # This is a local check.
        
        # Update adjs for old_color and new_color
        adjs[old_color][u].remove(v)
        adjs[old_color][v].remove(u)
        adjs[new_color][u].add(v)
        adjs[new_color][v].add(u)
        
        # Check if new_color has a C6 involving u or v?
        # Actually, we should check if the *entire* new_color graph has a C6.
        # For speed, we can just check if there's a C6 in new_color.
        # If there is, we might have created one.
        # To be safe, we check if the specific edge (u,v) is in a C6.
        # A C6 containing (u,v) looks like u-v-a-b-c-d-u or u-v-d-c-b-a-u.
        # So we need a path of length 4 from v to u in new_color \ {edge(v,u)}?
        # No, path from v to u of length 5? No, cycle length 6.
        # u-v is one edge. Need path v->...->u of length 5.
        
        def has_c6_through_edge(adj, u, v):
            # Check if there is a path of length 5 from v to u in adj, 
            # such that the internal vertices are distinct from u, v.
            # v - w1 - w2 - w3 - w4 - u
            for w1 in adj[v]:
                if w1 == u: continue
                for w2 in adj[w1]:
                    if w2 == u or w2 == v: continue
                    for w3 in adj[w2]:
                        if w3 == u or w3 == v or w3 == w1: continue
                        for w4 in adj[w3]:
                            if w4 == u or w4 == v or w4 == w1 or w4 == w2: continue
                            if u in adj[w4]:
                                return True
            return False

        if has_c6_through_edge(adjs[new_color], u, v):
            # Revert
            colors[idx] = old_color
            adjs[new_color][u].remove(v)
            adjs[new_color][v].remove(u)
            adjs[old_color][u].add(v)
            adjs[old_color][v].add(u)
        else:
            # Accept change
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # We know n=17 is possible. Let's try to find it.
    # Start from high n and go down? Or low to high?
    # Low to high is safer for early return, but we want max n.
    # Let's try n=17 first, then 16, etc.
    # But generating valid colorings is hard.
    # Let's try n from 10 to 17.
    
    for n in range(17, 9, -1):
        found = False
        # Try multiple seeds
        for seed in range(50):
            # Generate spectral coloring
            colors = _spectral_coloring(n, seed)
            
            # Repair
            colors = _repair_coloring(colors, n, max_steps=2000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback for smaller n if 17-10 failed
    for n in range(9, 1, -1):
        for seed in range(50):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=2000)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
