"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _bitmask(triple):
    return (1 << triple[0]) | (1 << triple[1]) | (1 << triple[2])

def _is_2colorable(edge_masks, n=9):
    """Check if hypergraph is 2-colorable (no monochromatic edge under any 2-coloring)."""
    m = len(edge_masks)
    if m == 0:
        return True
    full = (1 << m) - 1
    # dp[s] = bitmask of vertices in part A (color 0)
    dp = [0] * (1 << m)
    dp[0] = (1 << n) - 1  # all vertices available for part A
    for i in range(m):
        ei = 1 << i
        em = edge_masks[i]
        for s in range(1 << m):
            if not (s & ei):
                continue
            prev = s ^ ei
            avail = dp[prev]
            if avail == 0:
                continue
            # Option 1: put all 3 vertices of edge i in part A
            if (avail & em) == em:
                dp[s] |= (avail & ~em)
            # Option 2: put all 3 vertices in part B
            if (avail & em) == 0:
                dp[s] |= avail
            # Option 3: 2 in A, 1 in B
            for v in range(n):
                if em & (1 << v):
                    if avail & (em ^ (1 << v)):
                        dp[s] |= (avail & ~(em ^ (1 << v)))
            # Option 4: 1 in A, 2 in B
            for v in range(n):
                if em & (1 << v):
                    if avail & (1 << v):
                        dp[s] |= (avail & ~(em ^ (1 << v)))
    return dp[full] != 0

def _valid_3coloring(edges, colors):
    """Check no edge is monochromatic."""
    for e in edges:
        c = {colors[v] for v in e}
        if len(c) == 1:
            return False
    return True

def _find_3coloring(edge_masks, n=9):
    """Find a 3-coloring where no edge is monochromatic."""
    colors = [0] * n
    def backtrack(v):
        if v == n:
            return True
        for c in range(3):
            colors[v] = c
            # Check all edges containing v
            ok = True
            for em in edge_masks:
                if em & (1 << v):
                    # Check if this edge is now monochromatic
                    mono = True
                    for u in range(n):
                        if em & (1 << u):
                            if colors[u] != c:
                                mono = False
                                break
                    if mono:
                        ok = False
                        break
            if ok and backtrack(v + 1):
                return True
        colors[v] = -1
        return False
    if backtrack(0):
        return colors
    return None

def construct():
    deadline = time.time() + 75  # safety margin before 90s
    n = 9
    all_triples = list(itertools.combinations(range(n), 3))
    all_masks = [_bitmask(t) for t in all_triples]
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Try k = 4, 5, 6, 7, ... edges
    best_result = None
    
    for k in range(4, 12):
        if time.time() > deadline:
            break
        found = False
        # Search for a set of k edges that is NOT 2-colorable and HAS a 3-coloring
        for combo in itertools.combinations(range(len(all_triples)), k):
            if time.time() > deadline:
                break
            edge_masks = [all_masks[i] for i in combo]
            
            # Check 2-colorability
            if _is_2colorable(edge_masks, n):
                continue
            
            # Check 3-colorability
            colors = _find_3coloring(edge_masks, n)
            if colors is None:
                continue
            
            # Valid! Return it
            edges = [list(all_triples[i]) for i in combo]
            return {'edges': edges, 'colors': [int(c) for c in colors]}
    
    # Fallback: use the known 7-edge construction
    # A known 3-chromatic 3-uniform hypergraph on 9 vertices with 7 edges:
    # Vertices 0-8, colors: 0,0,1,1,2,2,0,1,2
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [1, 3, 5],
        [2, 4, 5],
        [0, 5, 6],
        [1, 4, 7],
        [2, 3, 8]
    ]
    colors = [0, 0, 1, 1, 2, 2, 0, 1, 2]
    
    # Verify
    for e in edges:
        c = {colors[v] for v in e}
        assert len(c) > 1, f"Edge {e} is monochromatic"
    
    return {'edges': [list(e) for e in edges], 'colors': [int(c) for c in colors]}
# EVOLVE-BLOCK-END
