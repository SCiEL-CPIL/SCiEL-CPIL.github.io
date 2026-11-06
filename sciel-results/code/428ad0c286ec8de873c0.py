import numpy as np
import time
import random

def _has_c6_bitset(adj, n):
    """
    Check if adjacency (list of int bitmasks) contains a C6.
    adj[i] is an integer where bit j is set if (i,j) is an edge.
    """
    for v0 in range(n):
        # Get neighbors of v0
        n0_mask = adj[v0]
        # Iterate over v1
        v1_list = [i for i in range(n) if (n0_mask >> i) & 1 and i > v0]
        for v1 in v1_list:
            # Get neighbors of v1, excluding v0
            n1_mask = adj[v1] & ~(1 << v0)
            # Iterate over v5 (neighbor of v0, distinct from v1)
            v5_list = [i for i in range(n) if (n0_mask >> i) & 1 and i != v1 and i > v0]
            for v5 in v5_list:
                # We need path v1 - v2 - v3 - v4 - v5
                # v2 in N(v1) \ {v0, v5}
                v2_mask = n1_mask & ~(1 << v5)
                # Iterate v2
                v2_list = [i for i in range(n) if (v2_mask >> i) & 1]
                for v2 in v2_list:
                    # v3 in N(v2) \ {v0, v1, v5}
                    v3_mask = adj[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                    v3_list = [i for i in range(n) if (v3_mask >> i) & 1]
                    for v3 in v3_list:
                        # v4 in N(v3) \ {v0, v1, v2, v5}
                        v4_mask = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                        v4_list = [i for i in range(n) if (v4_mask >> i) & 1]
                        for v4 in v4_list:
                            # Check if v5 in N(v4)
                            if (adj[v4] >> v5) & 1:
                                return True
    return False

def _build_adj_bitsets(edge_colors, n):
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
    """Check if the edge coloring has no C6 in any color class."""
    adj = _build_adj_bitsets(edge_colors, n)
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
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
    
    np.fill_diagonal(adj, 0)
    return adj

def _spectral_coloring(n, seed):
    """
    Generate edge colors using spectral embedding.
    """
    rng = np.random.RandomState(seed)
    
    d = min(8, n - 1)
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
    points = vecs.T # Shape (4, n)
    
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

def _repair_coloring(edge_colors, n, max_steps=10000):
    """
    Local search to remove C6s using bitsets.
    """
    colors = list(edge_colors)
    
    def get_edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (2 * n - i - 1) // 2 + (j - i - 1)
        
    def build_adjs(cols):
        return _build_adj_bitsets(cols, n)

    def find_c6_bitset(adj_c):
        # Returns a list of vertices in the C6 if found, else None
        # Reuses logic from _has_c6_bitset but returns the cycle
        for v0 in range(n):
            n0_mask = adj_c[v0]
            v1_list = [i for i in range(n) if (n0_mask >> i) & 1 and i > v0]
            for v1 in v1_list:
                n1_mask = adj_c[v1] & ~(1 << v0)
                v5_list = [i for i in range(n) if (n0_mask >> i) & 1 and i != v1 and i > v0]
                for v5 in v5_list:
                    v2_mask = n1_mask & ~(1 << v5)
                    v2_list = [i for i in range(n) if (v2_mask >> i) & 1]
                    for v2 in v2_list:
                        v3_mask = adj_c[v2] & ~(1 << v0) & ~(1 << v1) & ~(1 << v5)
                        v3_list = [i for i in range(n) if (v3_mask >> i) & 1]
                        for v3 in v3_list:
                            v4_mask = adj_c[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v5)
                            v4_list = [i for i in range(n) if (v4_mask >> i) & 1]
                            for v4 in v4_list:
                                if (adj_c[v4] >> v5) & 1:
                                    return [v0, v1, v2, v3, v4, v5]
        return None

    def has_c6_through_edge_bitset(adj, u, v):
        # Check if edge (u,v) is part of a C6 in adj
        # Path u-v-w1-w2-w3-w4-u? No, u-v is edge.
        # Cycle: u-v-a-b-c-d-u.
        # Need path v->a->b->c->d->u of length 4? No, length 4 edges: v-a, a-b, b-c, c-d, d-u.
        # So path of length 4 from v to u in G \ {u,v}?
        # v - a - b - c - d - u
        for a in range(n):
            if a == u or a == v: continue
            if not ((adj[v] >> a) & 1): continue
            for b in range(n):
                if b == u or b == v or b == a: continue
                if not ((adj[a] >> b) & 1): continue
                for c in range(n):
                    if c == u or c == v or c == a or c == b: continue
                    if not ((adj[b] >> c) & 1): continue
                    for d in range(n):
                        if d == u or d == v or d == a or d == b or d == c: continue
                        if not ((adj[c] >> d) & 1): continue
                        if (adj[d] >> u) & 1:
                            return True
        return False

    adjs = build_adjs(colors)
    
    for step in range(max_steps):
        c6_found = None
        c_found = -1
        for c in range(4):
            c6 = find_c6_bitset(adjs[c])
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
        
        # Try other colors
        other_colors = [c for c in range(4) if c != old_color]
        random.shuffle(other_colors)
        
        success = False
        for new_color in other_colors:
            # Tentatively change
            colors[idx] = new_color
            
            # Update adjs
            adjs[old_color][u] &= ~(1 << v)
            adjs[old_color][v] &= ~(1 << u)
            adjs[new_color][u] |= (1 << v)
            adjs[new_color][v] |= (1 << u)
            
            # Check if new_color has C6 through (u,v)
            # Note: has_c6_through_edge_bitset checks if (u,v) is in a C6.
            # It's sufficient to check this because we only added an edge.
            # If (u,v) is not in a C6, we haven't created a new C6 involving this edge.
            # Could we have created a C6 NOT involving this edge? No, we only added this edge.
            # So checking if (u,v) is in a C6 is sufficient.
            
            if not has_c6_through_edge_bitset(adjs[new_color], u, v):
                success = True
                break
            else:
                # Revert
                colors[idx] = old_color
                adjs[new_color][u] &= ~(1 << v)
                adjs[new_color][v] &= ~(1 << u)
                adjs[old_color][u] |= (1 << v)
                adjs[old_color][v] |= (1 << u)
        
        if not success:
            # If no color works, try a different edge or give up on this step
            # For now, just try next step (which will find same C6 or another)
            # To avoid infinite loop, we can force a random change if stuck?
            # Or just continue.
            pass

    return colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    best_n = 1
    best_colors = [0]
    
    start_time = time.time()
    
    # Try n=17 first
    for n in [17, 16, 15, 14, 13]:
        if time.time() - start_time > 100:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(200):
            if time.time() - start_time > 110:
                break
                
            colors = _spectral_coloring(n, seed)
            colors = _repair_coloring(colors, n, max_steps=5000)
            
            if _is_valid_coloring(colors, n):
                best_n = n
                best_colors = colors
                found = True
                break
        
        if found:
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback
    for n in range(12, 1, -1):
        if time.time() - start_time > 115:
            break
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
