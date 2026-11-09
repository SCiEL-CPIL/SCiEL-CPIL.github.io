"""Program for finding colourings of K_n without monochromatic K2,2,2."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

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

def _has_k222_in_set(neigh, S_mask):
    """Check if induced subgraph on S_mask contains K2,2.
    K2,2 = two non-adjacent vertices c,d with two common non-adjacent neighbors e,f.
    """
    verts = []
    mask = S_mask
    while mask:
        v = (mask & (-mask)).bit_length() - 1
        verts.append(v)
        mask &= mask - 1
    L = len(verts)
    if L < 4:
        return False
    
    for i in range(L):
        c = verts[i]
        for j in range(i + 1, L):
            d = verts[j]
            # c and d must NOT be adjacent
            if (neigh[c] >> d) & 1:
                continue
            # Find common neighbors in S_mask
            T = neigh[c] & neigh[d] & S_mask
            if T.bit_count() < 2:
                continue
            # Check if any pair in T is non-adjacent
            t_verts = []
            t_mask = T
            while t_mask:
                v = (t_mask & (-t_mask)).bit_length() - 1
                t_verts.append(v)
                t_mask &= t_mask - 1
            for k in range(len(t_verts)):
                e = t_verts[k]
                for l in range(k + 1, len(t_verts)):
                    f = t_verts[l]
                    if not ((neigh[e] >> f) & 1):
                        return True
    return False

def _count_k222_violations(neigh, S_mask):
    """Count K2,2 subgraphs in the subgraph induced by vertices in S_mask."""
    verts = []
    mask = S_mask
    while mask:
        v = (mask & (-mask)).bit_length() - 1
        verts.append(v)
        mask &= mask - 1
    L = len(verts)
    if L < 4:
        return 0
    
    count = 0
    for i in range(L):
        c = verts[i]
        for j in range(i + 1, L):
            d = verts[j]
            if (neigh[c] >> d) & 1:
                continue
            T = neigh[c] & neigh[d] & S_mask
            if T.bit_count() < 2:
                continue
            t_verts = []
            t_mask = T
            while t_mask:
                v = (t_mask & (-t_mask)).bit_length() - 1
                t_verts.append(v)
                t_mask &= t_mask - 1
            for k in range(len(t_verts)):
                e = t_verts[k]
                for l in range(k + 1, len(t_verts)):
                    f = t_verts[l]
                    if not ((neigh[e] >> f) & 1):
                        count += 1
    return count

def _calc_violations(S1_mask, n, neigh1, neigh0):
    """Calculate total K2,2,2 violations for a given S1 assignment.
    
    A violation occurs when there exists a vertex u such that:
    - For color 1: u is in S1, and G1[S1 ∩ N1(u)] contains K2,2
    - For color 0: u is in S0, and G0[S0 ∩ N0(u)] contains K2,2
    """
    S0_mask = ((1 << n) - 1) ^ S1_mask
    v = 0
    # Check Color 1 violations
    mask_u = S1_mask
    while mask_u:
        u = (mask_u & (-mask_u)).bit_length() - 1
        S_u = S1_mask & neigh1[u]
        if _has_k222_in_set(neigh1, S_u):
            v += 1
        mask_u &= mask_u - 1
    
    # Check Color 0 violations
    mask_u = S0_mask
    while mask_u:
        u = (mask_u & (-mask_u)).bit_length() - 1
        S_u = S0_mask & neigh0[u]
        if _has_k222_in_set(neigh0, S_u):
            v += 1
        mask_u &= mask_u - 1
    return v

def _try_extend(adj, n, deadline):
    """Try to extend the coloring to n+1 vertices using hill climbing."""
    # Build neighbor masks for current graph (n vertices)
    neigh1 = [0] * n
    neigh0 = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                neigh1[i] |= (1 << j)
                neigh1[j] |= (1 << i)
            else:
                neigh0[i] |= (1 << j)
                neigh0[j] |= (1 << i)
    
    best_violations = float('inf')
    best_S1 = None
    
    # Multiple restarts with different initialization strategies
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Start with all 0s
            S1 = 0
        elif restart == 1:
            # Start with all 1s
            S1 = (1 << n) - 1
        elif restart == 2:
            # Start with alternating pattern
            S1 = 0
            for i in range(n):
                if i % 2 == 0:
                    S1 |= (1 << i)
        elif restart == 3:
            # Start with first half 1s
            S1 = 0
            for i in range(n // 2):
                S1 |= (1 << i)
        else:
            # Random initialization with varying density
            density = 0.3 + 0.4 * (restart % 10) / 10.0
            S1 = 0
            for i in range(n):
                if random.random() < density:
                    S1 |= (1 << i)
        
        cur_v = _calc_violations(S1, n, neigh1, neigh0)
        
        if cur_v == 0:
            return S1
        
        # Hill climbing with multiple passes
        for pass_num in range(3):
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                # Try single flips
                for i in range(n):
                    new_S1 = S1 ^ (1 << i)
                    new_v = _calc_violations(new_S1, n, neigh1, neigh0)
                    if new_v < cur_v:
                        S1 = new_S1
                        cur_v = new_v
                        improved = True
                        if cur_v == 0:
                            return S1
                if time.time() > deadline:
                    break
                # Try double flips (more expensive but can escape local minima)
                if pass_num == 2:  # Only on final pass
                    for i in range(n):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, n):
                            new_S1 = S1 ^ (1 << i) ^ (1 << j)
                            new_v = _calc_violations(new_S1, n, neigh1, neigh0)
                            if new_v < cur_v:
                                S1 = new_S1
                                cur_v = new_v
                                improved = True
                                if cur_v == 0:
                                    return S1
        
        if cur_v < best_violations:
            best_violations = cur_v
            best_S1 = S1
            
    return best_S1

def _polish(adj, n, deadline):
    """Final polish pass: try to fix any remaining violations by flipping individual edges."""
    # This is a placeholder for a more sophisticated local search
    # For now, just return the current coloring
    return adj

def construct():
    start_time = time.time()
    DEADLINE = start_time + 145.0
    
    # Try Paley 29 (largest prime ≡ 1 mod 4 that is ≤ 31)
    adj29, colors29 = _paley_coloring(29)
    if _check_valid(adj29):
        # Try to extend to 30
        S1 = _try_extend(adj29, 29, DEADLINE)
        if S1 is not None:
            # Construct new colors
            new_colors = colors29[:]
            for i in range(29):
                c = 1 if (S1 >> i) & 1 else 0
                new_colors.append(c)
            
            # Verify
            adj30 = np.zeros((30, 30), dtype=np.int8)
            for i in range(29):
                for j in range(i + 1, 29):
                    idx = _edge_index(i, j, 29)
                    adj30[i, j] = colors29[idx]
                    adj30[j, i] = colors29[idx]
            for i in range(29):
                c = 1 if (S1 >> i) & 1 else 0
                adj30[29, i] = c
                adj30[i, 29] = c
                
            if _check_valid(adj30):
                return {'n': 30, 'colors': new_colors}
        
        return {'n': 29, 'colors': colors29}
    
    # Fallback: try smaller Paley graphs
    for p in [17, 13, 5]:
        if time.time() > DEADLINE:
            break
        adj, colors = _paley_coloring(p)
        if _check_valid(adj):
            return {'n': p, 'colors': colors}
    
    return {'n': 1, 'colors': [0]}

def _edge_index(i, j, n):
    """Calculate index in lexicographic list for edge (i, j) with i < j."""
    return i * n - i * (i + 1) // 2 + j - i - 1

# EVOLVE-BLOCK-END
