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

def _count_k222_violations(neigh, S_mask):
    """Count K2,2 subgraphs in the subgraph induced by vertices in S_mask.
    A K2,2 is defined by two non-adjacent pairs with all cross-edges.
    Here we check if the induced subgraph on S_mask contains K2,2.
    """
    # Extract vertices in S_mask
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
    # For each pair (c, d) in verts
    for i in range(L):
        c = verts[i]
        # d must be non-adjacent to c in the graph defined by neigh
        if neigh[c] & (1 << c): # self loop check, not needed for simple graph
            continue
        for j in range(i + 1, L):
            d = verts[j]
            # Check if c and d are NOT connected
            if (neigh[c] >> d) & 1:
                continue
            
            # Find common neighbors in S_mask
            T = neigh[c] & neigh[d] & S_mask
            # T contains vertices e such that e is connected to c and d.
            # We need at least two such vertices e, f that are NOT connected to each other.
            # Iterate through pairs in T
            t_verts = []
            t_mask = T
            while t_mask:
                v = (t_mask & (-t_mask)).bit_length() - 1
                t_verts.append(v)
                t_mask &= t_mask - 1
            
            # If we have at least 2 vertices in T, check if any pair is non-adjacent
            if len(t_verts) >= 2:
                # Optimization: if we just want to know if ANY exists, we can stop early.
                # But for hill climbing, count is better.
                # However, counting all K2,2 might be slow.
                # Let's just return 1 if found, to keep it simple?
                # No, gradient is better.
                # But for a first pass, let's just check existence for speed?
                # Actually, let's count.
                for k in range(len(t_verts)):
                    e = t_verts[k]
                    for l in range(k + 1, len(t_verts)):
                        f = t_verts[l]
                        if not ((neigh[e] >> f) & 1):
                            count += 1
    return count

def _try_extend(adj, n):
    """Try to extend the coloring to n+1 vertices."""
    start_time = time.time()
    DEADLINE = start_time + 100.0 # Leave margin for other checks
    
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
    
    # Local search for S1 (subset of 0..n-1 connected to new vertex with color 1)
    # S0 is the complement.
    
    import random
    
    for restart in range(50):
        if time.time() > DEADLINE:
            break
            
        # Initialize S1 randomly
        # Heuristic: maybe start with half 1s?
        S1 = 0
        for i in range(n):
            if random.random() < 0.5:
                S1 |= (1 << i)
        
        S0 = ((1 << n) - 1) ^ S1
        
        # Evaluate
        def calc_viol(S1_mask):
            S0_mask = ((1 << n) - 1) ^ S1_mask
            v = 0
            # Check Color 1 violations
            # For each u in S1, check K2,2 in G1[S1 & N1(u)]
            mask_u = S1_mask
            while mask_u:
                u = (mask_u & (-mask_u)).bit_length() - 1
                # S_u = S1 & N1(u)
                S_u = S1_mask & neigh1[u]
                v += _count_k222_violations(neigh1, S_u)
                mask_u &= mask_u - 1
            
            # Check Color 0 violations
            # For each u in S0, check K2,2 in G0[S0 & N0(u)]
            mask_u = S0_mask
            while mask_u:
                u = (mask_u & (-mask_u)).bit_length() - 1
                S_u = S0_mask & neigh0[u]
                v += _count_k222_violations(neigh0, S_u)
                mask_u &= mask_u - 1
            return v
        
        cur_v = calc_viol(S1)
        
        if cur_v == 0:
            return S1
        
        # Hill climbing
        improved = True
        while improved:
            if time.time() > DEADLINE:
                return None
            improved = False
            for i in range(n):
                # Flip bit i
                new_S1 = S1 ^ (1 << i)
                new_v = calc_viol(new_S1)
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

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    # Try Paley 29 (largest prime ≡ 1 mod 4 that is ≤ 31)
    adj29, colors29 = _paley_coloring(29)
    if _check_valid(adj29):
        # Try to extend to 30
        S1 = _try_extend(adj29, 29)
        if S1 is not None:
            # Construct new colors
            # New vertex is 29.
            # Edges (29, i) for i=0..28.
            # If i is in S1, color is 1. Else 0.
            new_colors = colors29[:]
            for i in range(29):
                c = 1 if (S1 >> i) & 1 else 0
                new_colors.append(c)
            
            # Verify
            adj30 = np.zeros((30, 30), dtype=np.int8)
            for i in range(29):
                for j in range(i + 1, 29):
                    adj30[i, j] = colors29[_edge_index(i, j, 29)]
                    adj30[j, i] = colors29[_edge_index(i, j, 29)]
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
    # Sum of (n - k) for k in i..j-1?
    # Index = sum_{k=0}^{i-1} (n - 1 - k) + (j - i - 1)
    # = i * n - i*(i+1)/2 + j - i - 1
    # Let's verify for n=3, i=0, j=1.
    # Edges: (0,1)->0, (0,2)->1, (1,2)->2
    # Formula: 0*3 - 0 + 1 - 0 - 1 = 0. Correct.
    # i=0, j=2: 0 + 2 - 0 - 1 = 1. Correct.
    # i=1, j=2: 1*3 - 1 + 2 - 1 - 1 = 3 - 1 + 0 = 2. Correct.
    return i * n - i * (i + 1) // 2 + j - i - 1

# EVOLVE-BLOCK-END
