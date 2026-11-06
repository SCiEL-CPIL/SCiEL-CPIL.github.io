import numpy as np
import time
import random

def _has_c6(adj, n):
    """
    Check if adjacency (list of sets) contains a C6.
    Optimized for small n.
    """
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0: continue
            for v5 in adj[v0]:
                if v5 == v1 or v5 <= v0: continue
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
        d -= 1
        
    adj = np.zeros((n, n), dtype=int)
    for _ in range(d // 2):
        perm = rng.permutation(n)
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            if u != v and adj[u, v] == 0:
                adj[u, v] = 1
                adj[v, u] = 1
            else:
                pass
    
    np.fill_diagonal(adj, 0)
    return adj

def _spectral_coloring_voronoi(n, seed):
    """
    Generate edge colors using spectral embedding with Voronoi partitioning.
    """
    rng = np.random.RandomState(seed)
    
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = _generate_random_regular_graph(n, d, rng)
    
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    vecs = eigvecs[:, 1:5]
    points = vecs.T  # Shape (4, n)
    
    # Random rotation to break symmetry
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    rotated_points = R @ points
    
    # Assign each vertex to one of 4 clusters using Voronoi (nearest center)
    # Initialize centers as random points in the data space
    centers = rng.randn(4, 4)
    # Scale centers to match data range
    data_min = rotated_points.min(axis=1)
    data_max = rotated_points.max(axis=1)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    centers = data_min + rng.rand(4, 4) * data_range
    
    # K-means style iteration
    labels = np.zeros(n, dtype=int)
    for _ in range(20):
        # Assign each point to nearest center
        dists = np.sum((rotated_points - centers[:, np.newaxis, :])**2, axis=0)  # (4, n)
        new_labels = np.argmin(dists, axis=0)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        
        # Update centers
        for c in range(4):
            mask = labels == c
            if np.sum(mask) > 0:
                centers[:, c] = rotated_points[:, mask].mean(axis=1)
    
    # Color edges: if both endpoints in same cluster, use color = cluster
    # If in different clusters, use a deterministic mapping
    # Actually, let's use a different strategy:
    # Color (i,j) = (label[i] + label[j]) % 4? No, that's biased.
    # Better: For each pair of clusters (a,b), assign a fixed color.
    # There are 10 pairs of clusters. Map to 4 colors.
    
    # Alternative: Use the actual positions more directly.
    # For edge (i,j), compute the direction from i to j in 4D.
    # Use the signs of the 4 components to determine a "region".
    # But we want 4 colors, not 16.
    
    # Let's use: color = argmax of dot product with 4 fixed vectors
    # This partitions the 4D space into 4 cones.
    
    # Generate 4 random vectors as "color directions"
    color_dirs = rng.randn(4, 4)
    # Normalize
    norms = np.linalg.norm(color_dirs, axis=0, keepdims=True)
    norms[norms == 0] = 1
    color_dirs = color_dirs / norms
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            # Dot product with each color direction
            dots = color_dirs.T @ diff  # Shape (4,)
            color = np.argmax(dots)
            edge_colors.append(int(color))
            
    return edge_colors

def _spectral_coloring_sign(n, seed):
    """
    Generate edge colors using spectral embedding with sign-based coloring.
    """
    rng = np.random.RandomState(seed)
    
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = _generate_random_regular_graph(n, d, rng)
    
    degree = adj.sum(axis=1)
    D = np.diag(degree)
    L = D - adj
    
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
        
    vecs = eigvecs[:, 1:5]
    points = vecs.T  # Shape (4, n)
    
    # Random rotation
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    rotated_points = R @ points
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_points[:, j] - rotated_points[:, i]
            signs = np.sign(diff)
            signs[signs == 0] = 1
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            idx = b0*8 + b1*4 + b2*2 + b3
            color = idx % 4
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=5000):
    """
    Local search to remove C6s.
    """
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def build_adjs(cols):
        adjs = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = cols[get_edge_idx(i, j)]
                adjs[c][i].add(j)
                adjs[c][j].add(i)
        return adjs

    def find_c6(adj_c):
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

    def has_c6_through_edge(adj, u, v):
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

    adjs = build_adjs(colors)
    
    for step in range(max_steps):
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6(adjs[c])
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors
            
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
        
        new_color = random.randint(0, 3)
        if new_color == old_color:
            new_color = (new_color + 1) % 4
            
        colors[idx] = new_color
        
        adjs[old_color][u].remove(v)
        adjs[old_color][v].remove(u)
        adjs[new_color][u].add(v)
        adjs[new_color][v].add(u)
        
        if has_c6_through_edge(adjs[new_color], u, v):
            colors[idx] = old_color
            adjs[new_color][u].remove(v)
            adjs[new_color][v].remove(u)
            adjs[old_color][u].add(v)
            adjs[old_color][v].add(u)
        else:
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    for n in range(17, 9, -1):
        found = False
        for seed in range(30):
            # Try both Voronoi and sign-based spectral coloring
            if seed % 2 == 0:
                colors = _spectral_coloring_voronoi(n, seed)
            else:
                colors = _spectral_coloring_sign(n, seed)
            
            colors = _repair_coloring(colors, n, max_steps=3000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    for n in range(9, 1, -1):
        for seed in range(30):
            if seed % 2 == 0:
                colors = _spectral_coloring_voronoi(n, seed)
            else:
                colors = _spectral_coloring_sign(n, seed)
            
            colors = _repair_coloring(colors, n, max_steps=3000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}

