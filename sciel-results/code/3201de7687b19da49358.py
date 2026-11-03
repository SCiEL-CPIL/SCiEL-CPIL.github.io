import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _luw_construct(k, d):
    """
    Lazebnik-Ustimenko-Woldar construction CD(d, q) with q=k.
    Returns (n, edges).
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Rule for LUW CD(d, q)
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _check_girth_fast(n, adj, min_girth):
    """
    Fast girth check: if min_girth is even, no cycle of length < min_girth means
    no two vertices at distance <= min_girth//2 - 1 are adjacent via a path of length 1
    that doesn't backtrack... actually simpler:
    
    For girth >= g (even), the BFS tree from any vertex must have no back-edges 
    creating cycles < g.
    
    Key optimization: if we do BFS from vertex s, and we find a back edge (u,v) where
    dist[u] + dist[v] + 1 < g, we fail.
    
    But we can prune: if dist[u] >= g//2, we don't need to explore further from u 
    because any cycle through u would have length at least 2*dist[u] + 1 >= g.
    Actually, if dist[u] >= g//2, then for any neighbor v:
    - if dist[v] == -1, we set it, but we won't explore from v if dist[v] >= g//2
    - if dist[v] != -1, cycle length = dist[u] + dist[v] + 1. Since dist[u] >= g//2,
      and dist[v] >= 0, cycle >= g//2 + 0 + 1. For g=16, that's >= 9. Not enough to prune.
    
    Better: limit BFS depth to g//2 - 1. If we don't find a short cycle within that depth,
    we're fine for this starting vertex.
    """
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        if _time_left() < 3.0:
            return False  # timeout, assume invalid to be safe
        
        dist = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] >= max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    queue.append(neighbor)
                else:
                    # Back edge or cross edge
                    cycle_len = dist[curr] + dist[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _construct_gp(n_half, k_gp):
    """
    Construct a Generalized Petersen graph GP(n_half, k_gp).
    Vertices: 0..n_half-1 (outer), n_half..2*n_half-1 (inner).
    Returns (n, edges) or None if invalid parameters.
    """
    n = 2 * n_half
    edges = []
    
    # Outer cycle
    for i in range(n_half):
        edges.append((i, (i + 1) % n_half))
    
    # Spokes
    for i in range(n_half):
        edges.append((i, n_half + i))
    
    # Inner cycle
    for i in range(n_half):
        edges.append((n_half + i, n_half + (i + k_gp) % n_half))
    
    return n, edges

def _check_gp_girth(n_half, k_gp, g):
    """
    Check girth of GP(n_half, k_gp) is at least g.
    Returns True if girth >= g.
    """
    n = 2 * n_half
    adj = [[] for _ in range(n)]
    
    for i in range(n_half):
        # Outer
        adj[i].append((i + 1) % n_half)
        adj[i].append((i - 1) % n_half)
        # Spoke
        adj[i].append(n_half + i)
    
    for i in range(n_half):
        v = n_half + i
        adj[v].append(i)  # spoke
        adj[v].append(n_half + (i + k_gp) % n_half)
        adj[v].append(n_half + (i - k_gp) % n_half)
    
    return _check_girth_fast(n, adj, g)

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 118.0
    
    # Strategy:
    # 1. Try GP graphs for small n (256 to ~1000) - these can be smaller than LUW
    # 2. Fall back to LUW construction if GP doesn't find a valid graph in time
    # 3. Return the best (smallest n) valid graph found
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for k=3, g=16: n >= 2*(3^7 - 1)/2 + 2 = ... 
    # Actually Moore bound: 1 + k*sum_{i=0}^{g/2-1} (k-1)^i
    # For k=3, g=16: 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382
    # Wait, that's for odd girth. For even girth g=2r:
    # n >= 2 * (k * (k-1)^(r-1) - 1) / (k-2)
    # For k=3, r=8: n >= 2 * (3 * 2^7 - 1) / 1 = 2 * 383 = 766
    # Hmm, but the problem says best lower bound is 512. Let me just use a reasonable range.
    
    # GP graphs have 2*n_half vertices. Try n_half from 256 to 1000.
    # But we need to be smart: not all k_gp values work.
    
    # For GP(n, k), girth depends on n and k.
    # Known: GP(n,1) is the Petersen graph family, girth 5 for n=5.
    # For large n, girth can be larger.
    
    # Let's try a range of n_half and k_gp values.
    # We want 2*n_half to be as small as possible, so start from small n_half.
    
    # Theoretical: for girth >= 16, we need n_half to be reasonably large.
    # Let's try n_half from 300 to 800, and for each, try a few k_gp values.
    
    t_start = time.monotonic()
    
    # Phase 1: Try GP graphs
    # For efficiency, try promising (n_half, k_gp) pairs.
    # k_gp should be relatively prime to n_half for the inner cycle to be a single cycle.
    # Also, k_gp should not be too small (girth would be small) or too large (similar).
    
    # Let's try n_half values that give total n in a reasonable range.
    # Best known is 936, so n_half up to 468 might be enough.
    # But let's go a bit higher to be safe.
    
    candidates = []
    for n_half in range(280, 600, 2):  # even n_half for simplicity
        if _time_left() < 15.0:
            break
        # Try a few k_gp values
        # k_gp should be coprime to n_half
        # Good candidates: k_gp around n_half/3 to n_half/2
        for k_gp in [n_half // 3, n_half // 3 + 1, n_half // 3 + 2, 
                      n_half // 2, n_half // 2 - 1, n_half // 2 - 2,
                      (n_half * 2) // 5, (n_half * 2) // 5 + 1]:
            if k_gp <= 1 or k_gp >= n_half - 1:
                continue
            if math.gcd(n_half, k_gp) != 1:
                continue
            candidates.append((n_half, k_gp))
    
    # Sort by n_half (total vertices) to find smallest first
    candidates.sort(key=lambda x: 2 * x[0])
    
    for n_half, k_gp in candidates:
        if _time_left() < 10.0:
            break
        if 2 * n_half >= best_n:
            break
        
        # Quick girth check
        if _check_gp_girth(n_half, k_gp, g):
            n, edges = _construct_gp(n_half, k_gp)
            if n < best_n:
                best_n = n
                best_edges = edges
                # Don't break immediately; there might be smaller n_half we missed
                # But since we're iterating in order of n_half, this is the smallest so far
                # We could break if we're confident, but let's continue a bit
                # Actually, since we're going in increasing order, we can break
                # after finding the first valid one, but only if we've checked all smaller n_half
                # Since candidates is sorted, we can break here
                break
    
    # Phase 2: If no GP graph found, or as a backup, try LUW
    if best_edges is None or _time_left() > 20.0:
        # LUW construction
        # For k=3, g=16: need d such that girth >= 16
        # LUW(3, d) has girth at least 2d+2 for odd d
        # So d >= 7 for girth >= 16
        for d in range(7, 12):
            if _time_left() < 10.0:
                break
            try:
                n, edges = _luw_construct(k, d)
                if n < best_n:
                    # Verify
                    adj = [[] for _ in range(n)]
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    if _check_girth_fast(n, adj, g):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                continue
    
    # If still nothing, use a large LUW as fallback
    if best_edges is None:
        d = 10
        best_n, best_edges = _luw_construct(k, d)
    
    return best_n, best_edges
