"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _contains_k222_bitmask(neigh):
    """Check if graph (given as list of neighbor bitmasks) contains K2,2,2.
    
    K2,2,2 = complete tripartite K_{2,2,2}: three disjoint pairs with all 12 cross-edges.
    For each pair (a,b), S = N(a)∩N(b). If |S|>=4, check if subgraph on S contains K2,2.
    """
    n = len(neigh)
    for a in range(n):
        for b in range(a + 1, n):
            S = neigh[a] & neigh[b]
            if S.bit_count() < 4:
                continue
            # Extract vertices in S
            verts = []
            mask = S
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                verts.append(v)
                mask &= mask - 1
            L = len(verts)
            # Check if induced subgraph on verts contains K2,2
            # For each pair (c,d) in verts, if N(c)∩N(d)∩S has >=2 vertices, we have K2,2
            for i in range(L):
                c = verts[i]
                for j in range(i + 1, L):
                    d = verts[j]
                    T = neigh[c] & neigh[d] & S
                    if T.bit_count() >= 2:
                        return True
    return False

def _check_valid(adj):
    """Check if neither color class contains K2,2,2."""
    n = adj.shape[0]
    
    # Build neighbor bitmasks for color-1 graph
    neigh1 = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
    
    if _contains_k222_bitmask(neigh1):
        return False
    
    # Build neighbor bitmasks for color-0 graph (complement)
    neigh0 = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
    
    if _contains_k222_bitmask(neigh0):
        return False
    
    return True

def _paley_coloring(p):
    """Generate Paley coloring for prime p ≡ 1 (mod 4)."""
    squares = {x * x % p for x in range(1, p)}
    colors = []
    adj = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(i + 1, p):
            c = 1 if (j - i) % p in squares else 0
            colors.append(c)
            if c == 1:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj, colors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    # Try Paley 29 (largest prime ≡ 1 mod 4 that is ≤ 31)
    adj29, colors29 = _paley_coloring(29)
    if _check_valid(adj29):
        return {'n': 29, 'colors': colors29}
    
    # Fallback: try smaller Paley graphs
    for p in [17, 13, 5]:
        if time.time() > DEADLINE:
            break
        adj, colors = _paley_coloring(p)
        if _check_valid(adj):
            return {'n': p, 'colors': colors}
    
    return {'n': 1, 'colors': [0]}

# EVOLVE-BLOCK-END
