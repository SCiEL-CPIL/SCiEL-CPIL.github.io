import numpy as np
import time
import random

def _has_c6_fast(adj, n):
    """
    Check if adjacency (list of sets) contains a C6.
    Optimized for small n using bitsets.
    """
    if n < 6:
        return False
    
    # Convert to bitsets for speed
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    
    for v0 in range(n):
        neighbors0 = adj_bits[v0]
        # Get neighbors of v0
        for v1 in range(n):
            if v1 <= v0:
                continue
            if not (neighbors0 >> v1) & 1:
                continue
            # v0-v1 edge exists
            # Find v5: neighbor of v0, v5 != v1, v5 > v0 (to avoid duplicates)
            for v5 in range(v0 + 1, n):
                if v5 == v1:
                    continue
                if not (neighbors0 >> v5) & 1:
                    continue
                # Now need path v1-v2-v3-v4-v5 where v2,v3,v4 distinct from v0,v1,v5
                # v2 in adj[v1] \ {v0, v5}
                neighbors1 = adj_bits[v1]
                for v2 in range(n):
                    if v2 == v0 or v2 == v5:
                        continue
                    if not (neighbors1 >> v2) & 1:
                        continue
                    # v3 in adj[v2] \ {v0, v1, v5}
                    neighbors2 = adj_bits[v2]
                    for v3 in range(n):
                        if v3 == v0 or v3 == v1 or v3 == v5:
                            continue
                        if not (neighbors2 >> v3) & 1:
                            continue
                        # v4 in adj[v3] \ {v0, v1, v2, v5}
                        neighbors3 = adj_bits[v3]
                        for v4 in range(n):
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5:
                                continue
                            if not (neighbors3 >> v4) & 1:
                                continue
                            # Check if v5 in adj[v4]
                            if (adj_bits[v4] >> v5) & 1:
                                return True
    return False

def _validate_coloring(colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    if n < 6:
        return True
    
    # Build adjacency for each color
    adjs = [[set() for _ in range(n)] for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adjs[c][i].add(j)
            adjs[c][j].add(i)
            idx += 1
    
    for c in range(4):
        if _has_c6_fast(adjs[c], n):
            return False
    return True

def _generate_random_regular_graph(n, d, rng):
    """
    Generate a random d-regular graph on n vertices using configuration model.
    Returns adjacency matrix.
    """
    if d >= n:
        d = n - 1
    if d % 2 != 0:
        d -= 1
    if d < 2:
        d = 2
    
    # Configuration model
    # Each vertex has d stubs
    stubs = []
    for i in range(n):
        for _ in range(d):
            stubs.append(i)
    
    adj = np.zeros((n, n), dtype=int)
    np.fill_diagonal(adj, 0)
    
    # Randomly pair up stubs
    rng.shuffle(stubs)
    for i in range(len(stubs) // 2):
        u = stubs[2 * i]
        v = stubs[2 * i + 1]
        if u != v and adj[u, v] == 0:
            adj[u, v] = 1
            adj[v, u] = 1
    
    # Verify regularity, if not, retry
    degrees = adj.sum(axis=1)
    if not np.all(degrees == d):
        # Try again with different seed offset
        return _generate_random_regular_graph(n, d, rng)
    
    return adj

def _spectral_color_v2(n, seed, degree=None):
    """
    Generate edge colors using spectral embedding of a random regular graph.
    Uses the 4 smallest non-trivial eigenvectors of the Laplacian.
    Colors edges based on the sign pattern of the difference vector.
    """
    rng = np.random.RandomState(seed)
    
    if n < 6:
        # Simple fallback
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
    
    if degree is None:
        # Choose degree to be roughly n/2 for good spectral properties
        degree = max(4, min(n - 1, n // 2))
        if degree % 2 != 0:
            degree -= 1
        if degree < 2:
            degree = 2
    
    # Generate random regular graph
    adj = _generate_random_regular_graph(n, degree, rng)
    
    # Compute Laplacian
    degree_vec = adj.sum(axis=1).astype(float)
    L = np.diag(degree_vec) - adj.astype(float)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    
    # Take the 4 smallest non-zero eigenvalues
    # eigvals[0] is 0 (for connected graph)
    # We want indices 1, 2, 3, 4
    if n <= 4:
        return [(i + j) % 4 for i in range(n) for j in range(i+1, n)]
    
    vecs = eigvecs[:, 1:5]  # Shape (n, 4)
    
    # Normalize each column to have unit norm
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    norms[norms == 0] = 1
    vecs = vecs / norms
    
    # Apply random rotation to break symmetry
    R = rng.randn(4, 4)
    Q, _ = np.linalg.qr(R)
    rotated_vecs = vecs @ Q.T  # Shape (n, 4)
    
    # For each edge (i, j), compute the difference vector
    # and assign color based on which octant/quadrant it falls into
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = rotated_vecs[j] - rotated_vecs[i]
            # Get sign pattern (4 bits)
            signs = np.sign(diff)
            signs[signs == 0] = 1  # Treat 0 as positive
            b0 = 1 if signs[0] > 0 else 0
            b1 = 1 if signs[1] > 0 else 0
            b2 = 1 if signs[2] > 0 else 0
            b3 = 1 if signs[3] > 0 else 0
            # Map 16 possible sign patterns to 4 colors
            # Use a balanced mapping
            idx = b0 * 8 + b1 * 4 + b2 * 2 + b3
            # Ensure balanced distribution
            color = (idx * 7) % 4  # Multiply by 7 to spread out
            edge_colors.append(color)
    
    return edge_colors

def _local_repair(colors, n, max_iter=500):
    """
    Local search to fix C6 violations.
    Only recolors edges that are part of a C6.
    """
    colors = list(colors)
    
    def get_edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
    
    def build_adjs(cols):
        adjs = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = cols[get_edge_idx(i, j)]
                adjs[c][i].add(j)
                adjs[c][j].add(i)
        return adjs
    
    def find_c6_in_color(adj_c):
        """Find a C6 in the given color class, return vertices or None."""
        for v0 in range(n):
            for v1 in adj_c[v0]:
                if v1 <= v0:
                    continue
                for v5 in adj_c[v0]:
                    if v5 == v1 or v5 <= v0:
                        continue
                    for v2 in adj_c[v1]:
                        if v2 == v0 or v2 == v5:
                            continue
                        for v3 in adj_c[v2]:
                            if v3 == v0 or v3 == v1 or v3 == v5:
                                continue
                            for v4 in adj_c[v3]:
                                if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5:
                                    continue
                                if v5 in adj_c[v4]:
                                    return [v0, v1, v2, v3, v4, v5]
        return None
    
    adjs = build_adjs(colors)
    
    for _ in range(max_iter):
        # Find any C6
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6_in_color(adjs[c])
            if c6:
                c6_found = c6
                c_found = c
                break
        
        if not c6_found:
            return colors  # Valid!
        
        # Pick a random edge from the C6 to recolor
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
        
        # Try each other color
        for new_color in range(4):
            if new_color == old_color:
                continue
            
            # Tentatively change
            colors[idx] = new_color
            adjs[old_color][u].remove(v)
            adjs[old_color][v].remove(u)
            adjs[new_color][u].add(v)
            adjs[new_color][v].add(u)
            
            # Check if this creates a C6 in new_color through edge (u,v)
            # A C6 through (u,v) needs a path of length 5 from v to u in new_color
            has_new_c6 = False
            for w1 in adjs[new_color][v]:
                if w1 == u:
                    continue
                for w2 in adjs[new_color][w1]:
                    if w2 == u or w2 == v:
                        continue
                    for w3 in adjs[new_color][w2]:
                        if w3 == u or w3 == v or w3 == w1:
                            continue
                        for w4 in adjs[new_color][w3]:
                            if w4 == u or w4 == v or w4 == w1 or w4 == w2:
                                continue
                            if u in adjs[new_color][w4]:
                                has_new_c6 = True
                                break
                        if has_new_c6:
                            break
                    if has_new_c6:
                        break
                if has_new_c6:
                    break
            
            if not has_new_c6:
                # Accept this change
                break
            else:
                # Revert
                colors[idx] = old_color
                adjs[new_color][u].remove(v)
                adjs[new_color][v].remove(u)
                adjs[old_color][u].add(v)
                adjs[old_color][v].add(u)
    
    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        found = False
        start_time = time.time()
        
        # Try many seeds with different degrees
        for seed in range(200):
            if time.time() - start_time > 100:
                break
            
            # Try different degrees for variety
            degree = None
            if seed % 4 == 0:
                degree = max(4, min(n-1, 4))
            elif seed % 4 == 1:
                degree = max(6, min(n-1, 6))
            elif seed % 4 == 2:
                degree = max(8, min(n-1, 8))
            elif seed % 4 == 3:
                degree = max(10, min(n-1, 10))
            
            colors = _spectral_color_v2(n, seed, degree)
            
            # Quick validation first
            if _validate_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
            
            # If invalid, try local repair
            if time.time() - start_time > 105:
                break
            
            colors = _local_repair(colors, n, max_iter=200)
            
            if _validate_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
    
    # Fallback for smaller n
    for n in range(9, 1, -1):
        for seed in range(50):
            colors = _spectral_color_v2(n, seed, None)
            if _validate_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
    
    return {'n': best_n, 'colors': best_colors}
