import numpy as np
import time
import random

def _find_c6_bitset(adj, n):
    """
    Find a C6 in a graph represented by bitset adjacency lists.
    adj[i] is an integer bitmask of neighbors of i.
    Returns a list of 6 vertices forming a C6, or None.
    """
    # To avoid finding the same C6 multiple times or symmetric ones, 
    # we can enforce v0 < v1 < v5 ? No, just v0 < v1 is enough for edge (v0,v1).
    # But we iterate v5 from adj[v0].
    # Let's just find any C6.
    
    for v0 in range(n):
        # Get neighbors of v0
        n0 = adj[v0]
        # We want v1 > v0 to avoid duplicate edges (v0,v1) and (v1,v0)
        # But adj is symmetric. 
        # Let's just iterate all v1 in adj[v0] with v1 > v0.
        while n0:
            v1_bit = n0 & -n0
            v1 = v1_bit.bit_length() - 1
            n0 ^= v1_bit
            if v1 <= v0:
                continue
            
            # v5 is another neighbor of v0, distinct from v1
            # We can restrict v5 > v0 to reduce search space? 
            # If we restrict v5 > v0, we might miss C6s where v5 < v0?
            # A C6 is v0-v1-...-v5-v0. 
            # If we fix v0 and v1 (v1>v0), v5 can be any other neighbor.
            # To avoid duplicates, we can say v5 is the "last" vertex.
            # It's easier to just check all.
            
            n5 = adj[v0] & ~v1_bit
            while n5:
                v5_bit = n5 & -n5
                v5 = v5_bit.bit_length() - 1
                n5 ^= v5_bit
                if v5 == v1:
                    continue
                # We don't strictly need v5 > v0 for correctness, just distinctness.
                
                # Path v1-v2-v3-v4-v5
                # v2 in adj[v1] \ {v0, v5, v1}
                n2 = adj[v1] & ~(1 << v0) & ~(1 << v5) & ~(1 << v1)
                while n2:
                    v2_bit = n2 & -n2
                    v2 = v2_bit.bit_length() - 1
                    n2 ^= v2_bit
                    
                    # v3 in adj[v2] \ {v0, v1, v5, v2}
                    n3 = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5) & ~(1 << v2)
                    while n3:
                        v3_bit = n3 & -n3
                        v3 = v3_bit.bit_length() - 1
                        n3 ^= v3_bit
                        
                        # v4 in adj[v3] \ {v0, v1, v2, v5, v3}
                        n4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5) & ~(1 << v3)
                        while n4:
                            v4_bit = n4 & -n4
                            v4 = v4_bit.bit_length() - 1
                            n4 ^= v4_bit
                            
                            if adj[v4] & (1 << v5):
                                return [v0, v1, v2, v3, v4, v5]
    return None

def _count_c6s_bitset(adj, n, limit=100):
    """
    Count the number of C6s, up to a limit.
    """
    count = 0
    # This is expensive. We might just use _find_c6 in a loop for repair,
    # but for "min-conflicts" we need a score.
    # For now, let's just find one and assume it's bad.
    # A better heuristic: try the 3 new colors, and for each, check if 
    # the specific edge creates a C6.
    pass

def _has_c6_through_edge_bitset(adj, n, u, v):
    """
    Check if edge (u,v) is part of a C6 in the graph defined by adj.
    This is much faster than finding any C6.
    """
    # We need a path v-w1-w2-w3-w4-u of length 5 in adj,
    # with internal vertices distinct from u, v.
    # Note: adj[u] contains v, and adj[v] contains u.
    
    # Remove edge (u,v) temporarily? No, just exclude u and v from internal nodes.
    # w1 in adj[v] \ {u}
    n1 = adj[v] & ~(1 << u)
    while n1:
        w1_bit = n1 & -n1
        w1 = w1_bit.bit_length() - 1
        n1 ^= w1_bit
        
        # w2 in adj[w1] \ {u, v, w1}
        n2 = adj[w1] & ~(1 << u) & ~(1 << v) & ~(1 << w1)
        while n2:
            w2_bit = n2 & -n2
            w2 = w2_bit.bit_length() - 1
            n2 ^= w2_bit
            
            # w3 in adj[w2] \ {u, v, w1, w2}
            n3 = adj[w2] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2)
            while n3:
                w3_bit = n3 & -n3
                w3 = w3_bit.bit_length() - 1
                n3 ^= w3_bit
                
                # w4 in adj[w3] \ {u, v, w1, w2, w3}
                n4 = adj[w3] & ~(1 << u) & ~(1 << v) & ~(1 << w1) & ~(1 << w2) & ~(1 << w3)
                while n4:
                    w4_bit = n4 & -n4
                    w4 = w4_bit.bit_length() - 1
                    n4 ^= w4_bit
                    
                    if adj[w4] & (1 << u):
                        return True
    return False

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    # Generate a random graph, preferably regular
    d = min(10, n - 1)
    if d % 2 == 1: d -= 1
    if d < 2: d = 2
    
    adj = np.zeros((n, n), dtype=int)
    # Simple method: random permutation matching for each degree layer
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
        
    vecs = eigvecs[:, 1:5]
    points = vecs.T # Shape (4, n)
    
    # Random rotation to break symmetry
    R = rng.randn(4, 4)
    R, _ = np.linalg.qr(R)
    
    rotated_points = R @ points
    
    # Map sign patterns to colors
    # There are 2^4 = 16 sign patterns.
    # We map them to 4 colors.
    # To ensure balance, we can shuffle the mapping.
    sign_to_color = {}
    colors_pool = [0, 1, 2, 3]
    rng.shuffle(colors_pool)
    
    # Generate all 16 patterns
    patterns = []
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            for b2 in [0, 1]:
                for b3 in [0, 1]:
                    patterns.append((b0, b1, b2, b3))
    
    # Shuffle the assignment of colors to patterns
    # We want each color to be used 4 times (16 patterns / 4 colors)
    # Just assign randomly but ensure balance? 
    # Simple: cycle through colors_pool for each pattern? 
    # That might create structure. 
    # Let's just map pattern index % 4 to a shuffled color list?
    # No, let's map pattern index to a random color, but ensure 
    # the mapping is a bijection from 4 groups of 4 patterns to 4 colors.
    
    # Group patterns into 4 groups of 4?
    # Actually, just random assignment is fine for initialization.
    # We'll use a random permutation of 4 colors repeated 4 times?
    # No, let's just use (b0*8 + b1*4 + b2*2 + b3) % 4, but with a random offset?
    # Let's stick to the previous method but with a random permutation of the final 4 colors.
    
    perm = rng.permutation(4)
    
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
            # Map 16 indices to 4 colors
            # Use a balanced mapping: idx // 4? No, that groups similar signs.
            # idx % 4 is better for mixing.
            color = perm[idx % 4]
            edge_colors.append(color)
            
    return edge_colors

def _repair_coloring(edge_colors, n, max_steps=10000):
    """
    Local search to remove C6s using bitsets and min-conflicts heuristic.
    """
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    # Build bitset adjacency
    def build_adjs(cols):
        adjs = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = cols[get_edge_idx(i, j)]
                adjs[c][i] |= (1 << j)
                adjs[c][j] |= (1 << i)
        return adjs

    def find_c6(adj_c):
        return _find_c6_bitset(adj_c, n)

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
            
        # We have a C6 in color c_found
        edges_in_cycle = [
            (c6_found[0], c6_found[1]),
            (c6_found[1], c6_found[2]),
            (c6_found[2], c6_found[3]),
            (c6_found[3], c6_found[4]),
            (c6_found[4], c6_found[5]),
            (c6_found[5], c6_found[0])
        ]
        
        # Pick a random edge
        u, v = random.choice(edges_in_cycle)
        idx = get_edge_idx(u, v)
        old_color = colors[idx]
        
        # Try all 3 other colors
        best_new_color = -1
        min_conflicts = float('inf')
        
        for new_color in range(4):
            if new_color == old_color:
                continue
                
            # Tentatively change color
            # Remove from old, add to new
            adjs[old_color][u] &= ~(1 << v)
            adjs[old_color][v] &= ~(1 << u)
            adjs[new_color][u] |= (1 << v)
            adjs[new_color][v] |= (1 << u)
            
            # Check if this creates a C6 through (u,v) in new_color
            # We only check the specific edge to be fast.
            # If it creates a C6, it's a conflict.
            # We don't count ALL C6s, just if this edge is in one.
            # This is a local heuristic.
            if _has_c6_through_edge_bitset(adjs[new_color], n, u, v):
                conflicts = 1
            else:
                conflicts = 0
            
            # Revert
            adjs[new_color][u] &= ~(1 << v)
            adjs[new_color][v] &= ~(1 << u)
            adjs[old_color][u] |= (1 << v)
            adjs[old_color][v] |= (1 << u)
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_new_color = new_color
                
            if min_conflicts == 0:
                break # Found a perfect move
        
        if best_new_color == -1:
            # Should not happen
            break
            
        # Apply the best move
        colors[idx] = best_new_color
        adjs[old_color][u] &= ~(1 << v)
        adjs[old_color][v] &= ~(1 << u)
        adjs[best_new_color][u] |= (1 << v)
        adjs[best_new_color][v] |= (1 << u)

    return colors

def _is_valid_coloring(edge_colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    adjs = [[0] * n for _ in range(4)]
    for i in range(n):
        for j in range(i+1, n):
            c = edge_colors[get_edge_idx(i, j)]
            adjs[c][i] |= (1 << j)
            adjs[c][j] |= (1 << i)
            
    for c in range(4):
        if _find_c6_bitset(adjs[c], n):
            return False
    return True

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        found = False
        # Try multiple seeds
        for seed in range(20):
            # Generate spectral coloring
            colors = _spectral_coloring(n, seed)
            
            # Repair
            colors = _repair_coloring(colors, n, max_steps=5000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback for smaller n if 17-10 failed
    for n in range(9, 1, -1):
        for seed in range(20):
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=5000)
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                break
        if best_n == n:
            break
            
    return {'n': best_n, 'colors': best_colors}
