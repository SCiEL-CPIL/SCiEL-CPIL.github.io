"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

# Pre-compute the 15 pairings of 6 vertices into 3 pairs
_PAIRINGS = [
    ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
    ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
    ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
    ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
    ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
]

def _has_k222(adj, n):
    """
    Check if graph with adjacency matrix adj (numpy int array) on n vertices
    contains K2,2,2 as a non-induced subgraph.
    Uses bitmask approach for speed.
    """
    # adj is an n x n numpy array of 0s and 1s
    # For each pair of vertices (a,b), compute common neighbors
    # If there exist two disjoint pairs (c,d) and (e,f) such that
    # c,d are common neighbors of a,b, and e,f are common neighbors of c,d
    # then we have K2,2,2.
    
    # More efficiently: for each 4-tuple (a,b,c,d) where c,d are common neighbors of a,b,
    # check if c and d have at least 2 common neighbors (excluding a,b).
    # Actually, we need: a,b share neighbors c,d (so edge a-c,a-d,b-c,b-d exist),
    # and c,d share neighbors e,f (so edge c-e,c-f,d-e,d-f exist),
    # and a,b share neighbors e,f (so edge a-e,a-f,b-e,b-f exist).
    # So we need 6 vertices a,b,c,d,e,f with all cross-edges present.
    
    # Strategy: For each pair (a,b), let N_ab = common neighbors of a and b.
    # For each pair (c,d) in N_ab, let N_cd = common neighbors of c and d.
    # We need |N_cd \ {a,b}| >= 2, and also e,f must be in N_ab.
    # So we need |N_cd ∩ N_ab \ {a,b}| >= 2.
    
    # Even better: for each pair (c,d), let N_cd = common neighbors of c and d.
    # For each pair (a,b) in N_cd, let N_ab = common neighbors of a and b.
    # We need c,d ∈ N_ab and |N_cd ∩ N_ab \ {a,b}| >= 2.
    
    # Let's use the bitmask approach:
    # For each vertex v, compute bitmask of neighbors.
    # common(a,b) = bitmask_a & bitmask_b
    
    # Build neighbor bitmasks
    # n <= 31, so we can use Python integers as bitmasks
    neigh_mask = [0] * n
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if i != j and row[j]:
                mask |= (1 << j)
        neigh_mask[i] = mask
    
    # For each pair (a,b), compute common neighbors
    # common_ab = neigh_mask[a] & neigh_mask[b]
    # For each pair (c,d) with c,d in common_ab, compute common_cd = neigh_mask[c] & neigh_mask[d]
    # Check if (common_cd & common_ab) has at least 2 bits set (excluding a,b)
    
    # To be efficient, iterate over pairs (c,d) first since there are C(n,2) of them
    # For each (c,d), common_cd = neigh_mask[c] & neigh_mask[d]
    # The candidates for (a,b) are pairs from common_cd
    # For each such pair (a,b), check if common_ab contains c,d and has >=2 other common neighbors with c,d
    
    # Actually, let me think differently.
    # K2,2,2 has 6 vertices. The 12 edges are all cross-edges between the 3 pairs.
    # Equivalently: there exist 6 distinct vertices v1..v6 such that
    # v1,v2 are connected to v3,v4,v5,v6 (8 edges)
    # v3,v4 are connected to v5,v6 (4 edges)
    # Total 12 edges.
    
    # So: pick pair (a,b). They must be connected to 4 vertices c,d,e,f.
    # Among c,d,e,f, pick pair (c,d). c,d must be connected to e,f.
    # So: for pair (a,b), let S = common neighbors of a,b.
    # For each pair (c,d) in S, check if c and d have at least 2 common neighbors in S \ {c,d}.
    
    # Let's implement this:
    for a in range(n):
        for b in range(a+1, n):
            # common neighbors of a and b
            common_ab = neigh_mask[a] & neigh_mask[b]
            # Remove a and b from common_ab (they shouldn't be there anyway since no self-loops)
            common_ab &= ~(1 << a) & ~(1 << b)
            
            # Get list of common neighbors
            cn = []
            for v in range(n):
                if (common_ab >> v) & 1:
                    cn.append(v)
            
            if len(cn) < 4:
                continue
            
            # For each pair (c,d) in cn, check if c and d have >= 2 common neighbors in cn \ {c,d}
            for ci in range(len(cn)):
                c = cn[ci]
                for di in range(ci+1, len(cn)):
                    d = cn[di]
                    # Common neighbors of c and d
                    common_cd = neigh_mask[c] & neigh_mask[d]
                    # Intersect with cn, excluding c and d
                    count = 0
                    for v in cn:
                        if v != c and v != d and ((common_cd >> v) & 1):
                            count += 1
                            if count >= 2:
                                return True
    return False

def _build_adj(colors, n):
    """Build adjacency matrix (numpy int array) from color list."""
    adj = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 1:
                adj[i, j] = 1
                adj[j, i] = 1
            idx += 1
    return adj

def _is_valid(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class)."""
    if len(colors) != n*(n-1)//2:
        return False
    
    adj1 = _build_adj(colors, n)
    if _has_k222(adj1, n):
        return False
    
    # Complement graph
    adj0 = np.ones((n, n), dtype=np.int8) - adj1
    np.fill_diagonal(adj0, 0)
    
    if _has_k222(adj0, n):
        return False
    return True

def _paley_colors(n, p):
    """
    Generate colors for the first n vertices of the Paley graph on p vertices.
    Color 1 if (j-i) mod p is a quadratic residue.
    Color 0 otherwise.
    """
    if p < n:
        return None
    squares = set((x*x) % p for x in range(1, p))
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % p
            colors.append(1 if diff in squares else 0)
    return colors

def _local_search(colors, n, deadline, max_iters=5000):
    """
    Try to improve the coloring by flipping edges to eliminate K2,2,2.
    Returns (improved_colors, improved_n) or (None, n) if no improvement.
    """
    colors = list(colors)
    adj1 = _build_adj(colors, n)
    
    # Check current validity
    if _is_valid(colors, n):
        return colors, n
    
    # Find a violating K2,2,2 and try to fix it by flipping edges
    # This is a simplified local search
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Check if valid
        if _is_valid(colors, n):
            return colors, n
        
        # Try random edge flips
        num_edges = n*(n-1)//2
        for _ in range(20):  # Try up to 20 random flips per iteration
            if time.time() > deadline:
                break
            idx = np.random.randint(num_edges)
            colors[idx] = 1 - colors[idx]
            adj1 = _build_adj(colors, n)
            # Quick check
            if not _has_k222(adj1, n):
                adj0 = np.ones((n, n), dtype=np.int8) - adj1
                np.fill_diagonal(adj0, 0)
                if not _has_k222(adj0, n):
                    return colors, n
    
    return None, n

def construct():
    deadline = time.time() + 140  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try known good constructions
    candidates = []
    
    # Generate Paley 29
    c29 = _paley_colors(29, 29)
    if c29:
        candidates.append((29, c29))
        
    # Generate Paley 37 and try to find a valid subset of size 30
    c37_30 = _paley_colors(30, 37)
    if c37_30:
        candidates.append((30, c37_30))
    
    # Try other primes for Paley constructions
    for p in [31, 37, 43, 47, 59, 67, 71, 79, 83, 97, 103, 107, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]:
        if p >= 30 and time.time() < deadline - 10:
            c = _paley_colors(30, p)
            if c:
                candidates.append((30, c))
    
    # Sort by n descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Verify candidates
    for n, colors in candidates:
        if time.time() > deadline - 5:
            break
        try:
            if _is_valid(colors, n):
                best_n = n
                best_colors = colors
                if n == 30:
                    return {'n': n, 'colors': colors}
        except Exception:
            continue
    
    # If we have a valid 29-coloring, try to extend to 30
    if best_n == 29 and time.time() < deadline - 20:
        base_colors = best_colors
        # Try adding vertex 29 with different color patterns
        # The new vertex connects to 29 existing vertices
        # We need to assign colors to these 29 edges
        # Use local search to find a valid assignment
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initial assignment for new vertex edges
            new_edges = [np.random.randint(2) for _ in range(29)]
            full_colors = base_colors + new_edges
            
            if _is_valid(full_colors, 30):
                return {'n': 30, 'colors': full_colors}
            
            # Try local search on the new edges
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(29)
                new_edges[idx] = 1 - new_edges[idx]
                full_colors = base_colors + new_edges
                if _is_valid(full_colors, 30):
                    return {'n': 30, 'colors': full_colors}
    
    if best_colors is None:
        # Absolute fallback: n=1
        return {'n': 1, 'colors': []}
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
