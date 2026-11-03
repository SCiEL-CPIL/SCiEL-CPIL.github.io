import time
import math
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small circulant graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)

    # First, get the LUW result to know the baseline and ensure we have a valid fallback
    luw_n, luw_edges = _luw_construct(k, g)
    best_n = luw_n
    best_edges = luw_edges
    
    # Search for circulant graphs C_n(a, b)
    # We search from the Moore bound (243) up to the size of the LUW graph
    # The known cage is 320, so we expect to find it if it is circulant.
    # If it's not circulant, we might find a larger one or fail to find any < luw_n.
    
    # Moore bound for k=4, g=10 is 243.
    # We cap the search at 500 to be safe, or luw_n - 1 if luw_n is smaller.
    # If luw_n is very large (e.g. 2000), we still only search up to 500 because 
    # larger graphs are less likely to be "cage" candidates in this simple family 
    # and 500 is a reasonable bound for quick search. 
    # Actually, if we find a graph of size 320, we stop.
    # If we don't find one < 500, we return LUW.
    
    max_search_n = min(500, best_n - 1)
    
    for n in range(243, max_search_n + 1):
        if time.time() > deadline:
            break
            
        # Iterate over possible generators a, b
        # 1 <= a < b < n/2
        # a + b != n
        # gcd(n, a, b) == 1
        
        # Optimization: precompute valid (a,b) pairs? 
        # n is small (<=500), so n/2 is <= 250.
        # Pairs ~ 250^2 / 2 = 31250.
        # Girth check is fast.
        
        for a in range(1, n // 2):
            if (2 * a) % n == 0:
                continue
            for b in range(a + 1, n // 2):
                if (2 * b) % n == 0:
                    continue
                if a + b == n:
                    continue
                
                # Quick check: gcd(n, a, b) must be 1 for connectivity
                if math.gcd(math.gcd(n, a), b) != 1:
                    continue
                
                # Check girth
                if _check_circulant_girth(n, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    
                    # Verify exact girth just in case (filter might have false negatives? 
                    # The filter is necessary but is it sufficient? 
                    # _check_circulant_girth does a full BFS check for short cycles.
                    # It returns False if a short cycle is found.
                    # If it returns True, no short cycle was found in the BFS tree 
                    # which covers all vertices (if connected).
                    # Since we checked gcd, it is connected.
                    # So it should be valid.
                    # However, the BFS check in _check_circulant_girth is:
                    # "if du + dist[v] + 1 < g: return False"
                    # This detects any cycle of length < g.
                    # So if it returns True, girth >= g.
                    
                    return n, edges

    # If no circulant found, return LUW
    return best_n, best_edges

def _check_circulant_girth(n, a, b, g):
    """
    Fast check for girth of circulant graph C_n(a,b).
    Returns True if girth >= g, False otherwise.
    Uses BFS from 0. If any edge (u,v) has dist[u]+dist[v]+1 < g, returns False.
    This is a necessary condition for girth >= g? 
    Actually, if girth < g, there exists a cycle of length L < g.
    Does there always exist an edge (u,v) in that cycle such that dist_0(u)+dist_0(v)+1 < g?
    Yes, the cycle contains a path from 0 to some node and back?
    More simply: The BFS tree from 0 defines distances.
    Any non-tree edge (u, v) creates a cycle of length dist[u] + dist[v] + 1.
    If there is ANY cycle of length < g, it must be composed of tree paths and non-tree edges.
    The shortest cycle passing through 0 is detected by non-tree edges incident to 0? No.
    The shortest cycle in the graph might not pass through 0.
    However, for a regular graph, if there is a short cycle, it is likely to be detected 
    by BFS from a single vertex if the graph is small or the cycle is very short.
    But to be rigorous:
    The condition "min_{(u,v)} dist[u]+dist[v]+1 >= g" is NOT sufficient for girth >= g 
    if the graph is not a tree-like structure around 0.
    Wait. The cycle length is exactly dist[u] + dist[v] + 1 ONLY IF the paths from 0 to u and 0 to v 
    are disjoint except at 0. If they intersect at some node w != 0, the cycle is shorter.
    Actually, dist[u] + dist[v] + 1 is an upper bound on the cycle length formed by edge (u,v).
    If dist[u] + dist[v] + 1 >= g, it doesn't mean the cycle is >= g.
    Example: 0-1-2-3-1. dist[0]=0, dist[1]=1, dist[2]=2, dist[3]=2.
    Edge (1,3): dist[1]+dist[3]+1 = 1+2+1 = 4. Cycle 1-2-3-1 has length 3.
    So the filter "no edge has sum < g" is NOT sufficient to prove girth >= g.
    It only proves that there are no cycles of length < g that are "tight" in the BFS tree sense.
    
    Therefore, we MUST do a full verification if the filter passes, or use a more robust filter.
    Given n <= 500, a full verification from EVERY vertex is O(n^2) which is 250,000 ops.
    This is fast enough.
    So, _check_circulant_girth should just do the full check from all vertices?
    Or just from one vertex is not enough.
    
    Let's implement _verify_girth which checks from all vertices.
    """
    # For n <= 500, checking from all vertices is safe and fast.
    return _verify_girth_circulant(n, a, b, g)

def _verify_girth_circulant(n, a, b, g):
    """
    Exact girth check for circulant graph C_n(a,b).
    Returns True if girth >= g.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        adj[i].append((i + a) % n)
        adj[i].append((i - a) % n)
        adj[i].append((i + b) % n)
        adj[i].append((i - b) % n)
    
    # Check from every vertex
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= (g - 2) // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)
                else:
                    # Check if this edge creates a short cycle
                    # The cycle length is at most du + dist[v] + 1
                    if du + dist[v] + 1 < g:
                        return False
    return True

def _luw_construct(k, g):
    # Original LUW code
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
