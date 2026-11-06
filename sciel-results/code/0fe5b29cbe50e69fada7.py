"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _has_k222(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains K2,2,2."""
    # K2,2,2 has 6 vertices: two parts of size 2, each vertex connected to all 4 in other part
    # Enumerate all 6-subsets
    for verts in itertools.combinations(range(n), 6):
        a, b, c, d, e, f = verts
        # Try all ways to partition into 3 pairs: (a,b),(c,d),(e,f) and permutations
        # K2,2,2: vertices {1,2,3,4,5,6} with edges between {1,2} and {3,4,5,6}, 
        # between {3,4} and {1,2,5,6}, between {5,6} and {1,2,3,4}
        # Actually K_{2,2,2} is the complete tripartite graph with parts of size 2.
        # So we need 3 pairs such that all cross-pair edges exist.
        # Partitions of 6 elements into 3 pairs:
        # (0,1)(2,3)(4,5), (0,1)(2,4)(3,5), (0,1)(2,5)(3,4),
        # (0,2)(1,3)(4,5), (0,2)(1,4)(3,5), (0,2)(1,5)(3,4),
        # (0,3)(1,2)(4,5), (0,3)(1,4)(2,5), (0,3)(1,5)(2,4),
        # (0,4)(1,2)(3,5), (0,4)(1,3)(2,5), (0,4)(1,5)(2,3),
        # (0,5)(1,2)(3,4), (0,5)(1,3)(2,4), (0,5)(1,4)(2,3)
        pairs_list = [
            ((0,1),(2,3),(4,5)), ((0,1),(2,4),(3,5)), ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)), ((0,2),(1,4),(3,5)), ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)), ((0,3),(1,4),(2,5)), ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)), ((0,4),(1,3),(2,5)), ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)), ((0,5),(1,3),(2,4)), ((0,5),(1,4),(2,3)),
        ]
        for p1, p2, p3 in pairs_list:
            v1, v2 = verts[p1[0]], verts[p1[1]]
            v3, v4 = verts[p2[0]], verts[p2[1]]
            v5, v6 = verts[p3[0]], verts[p3[1]]
            # Check all 12 cross edges
            if (adj[v1][v3] and adj[v1][v4] and adj[v1][v5] and adj[v1][v6] and
                adj[v2][v3] and adj[v2][v4] and adj[v2][v5] and adj[v2][v6] and
                adj[v3][v5] and adj[v3][v6] and adj[v4][v5] and adj[v4][v6]):
                return True
    return False

def _build_adj(colors, n):
    """Build adjacency matrix from color list."""
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == 1:
                adj[i][j] = 1
                adj[j][i] = 1
            idx += 1
    return adj

def _is_valid(colors, n):
    """Check if coloring is valid (no K2,2,2 in either color class)."""
    if len(colors) != n*(n-1)//2:
        return False
    # Color 1 graph
    adj1 = _build_adj(colors, n)
    if _has_k222(adj1, n):
        return False
    # Color 0 graph (complement)
    adj0 = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                adj0[i][j] = 1 - adj1[i][j]
    if _has_k222(adj0, n):
        return False
    return True

def _paley(n, p):
    """Paley coloring using prime p >= n. Color 0 if (j-i) mod p is a quadratic residue."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append(0 if (j-i) % p in squares else 1)
    return colors

def _try_paley(n, p):
    """Try Paley coloring with prime p for n vertices."""
    colors = _paley(n, p)
    if colors is None:
        return False
    return _is_valid(colors, n)

def _local_search(colors, n, max_iters=5000, time_limit=100):
    """Local search to find valid coloring."""
    start = time.time()
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        if _is_valid(colors, n):
            return True
        # Pick a random edge to flip
        idx = random.randint(0, len(colors)-1)
        colors[idx] = 1 - colors[idx]
    return _is_valid(colors, n)

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Try known Paley constructions for various primes
    primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    
    best_n = 0
    best_colors = None
    
    # First try direct Paley colorings
    for p in primes:
        for n in range(min(p, 31), 16, -1):
            if time.time() > deadline:
                break
            colors = _paley(n, p)
            if colors is not None and _is_valid(colors, n):
                if n > best_n:
                    best_n = n
                    best_colors = colors[:]
                break  # Found valid for this n, try smaller n with same p
    
    if best_n >= 29:
        return {'n': best_n, 'colors': best_colors}
    
    # Try random initialization + local search for larger n
    for n in range(min(best_n + 1, 31), 17, -1):
        if time.time() > deadline:
            break
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Random initial coloring
            colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
            if _local_search(colors, n, max_iters=2000, time_limit=min(30, max(1, int(deadline - time.time()) - 10))):
                if n > best_n:
                    best_n = n
                    best_colors = colors[:]
                    if n >= 29:
                        return {'n': best_n, 'colors': best_colors}
    
    # Also try starting from Paley and doing local search
    for p in primes:
        if time.time() > deadline:
            break
        n_target = min(p, 31)
        if n_target <= best_n:
            continue
        colors = _paley(n_target, p)
        if colors is None:
            continue
        if _local_search(colors, n_target, max_iters=5000, time_limit=min(40, max(1, int(deadline - time.time()) - 10))):
            if n_target > best_n:
                best_n = n_target
                best_colors = colors[:]
                if n_target >= 29:
                    return {'n': best_n, 'colors': best_colors}
    
    if best_colors is None:
        # Fallback to n=17 Paley
        squares = {x * x % 17 for x in range(1, 17)}
        best_n = 17
        best_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
