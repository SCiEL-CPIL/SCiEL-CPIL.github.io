"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

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

def _try_extend_paley(adj29, colors29, max_n=31, deadline=None):
    """Try to extend the Paley 29 coloring by adding vertices one at a time.
    
    For each new vertex, try all 2^29 colorings is infeasible. Instead, use
    a greedy/heuristic approach: for each new vertex, find a valid coloring
    that doesn't create a monochromatic K2,2,2 in either color class.
    """
    if deadline is None:
        deadline = time.time() + 10
    
    n_current = 29
    adj = adj29.copy()
    colors = list(colors29)
    
    for target_n in range(n_current + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Try to add vertex target_n-1
        # For each possible coloring of edges from new vertex to existing vertices,
        # check if valid. Use a smart heuristic: try all 2^29 is too many.
        # Instead, use the fact that we need to avoid creating K2,2,2.
        
        # Heuristic: try a few random assignments and also try the "balanced" one
        import random
        random.seed(42)
        
        found = False
        # Try random assignments
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            # Random coloring for new vertex
            new_colors = [random.randint(0, 1) for _ in range(n_current)]
            
            # Check validity
            new_adj = np.zeros((target_n, target_n), dtype=np.int8)
            new_adj[:n_current, :n_current] = adj
            for i in range(n_current):
                if new_colors[i] == 1:
                    new_adj[i, n_current] = 1
                    new_adj[n_current, i] = 1
                else:
                    new_adj[i, n_current] = 0
                    new_adj[n_current, i] = 0
            
            if _check_valid(new_adj):
                adj = new_adj
                colors = colors + new_colors
                n_current = target_n
                found = True
                break
        
        if not found:
            break
    
    return n_current, adj, colors

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0
    
    # Try Paley 29 (largest prime ≡ 1 mod 4 that is ≤ 31)
    adj29, colors29 = _paley_coloring(29)
    if _check_valid(adj29):
        # Try to extend beyond 29
        n_extended, adj_ext, colors_ext = _try_extend_paley(adj29, colors29, max_n=31, deadline=DEADLINE)
        if n_extended > 29:
            return {'n': n_extended, 'colors': colors_ext}
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
