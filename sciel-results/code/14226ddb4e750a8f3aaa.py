"""Construct a k-regular graph with girth at least g using tuned LUW construction."""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    Uses the Lazebnik-Ustimenko-Woldar construction with tuned parameter d,
    verifying girth by non-backtracking walk enumeration.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin (150s limit)

    def elapsed():
        return time.time() - start_time

    def check_deadline():
        return elapsed() > deadline

    best_n = None
    best_edges = None

    # LUW construction: q-regular bipartite graph, girth >= d+5 for odd d >= 3
    # We try smaller d values (fewer vertices) and verify girth >= g
    # d must be odd for the standard girth guarantee, but we verify anyway

    q = k
    # d must satisfy d >= 3 and be such that the graph exists
    # Try d values from small to large, stop when we find one with girth >= g
    # Start from a reasonable minimum and go up
    
    # The Moore bound gives a lower bound on n, but for LUW the size is q * (q-1)^d * 2 (roughly)
    # For k=3: n = 2 * 3 * 2^d = 6 * 2^d
    # g=16: need n >= 512 (Moore bound), so 6*2^d >= 512 => 2^d >= 85.3 => d >= 7
    # But girth guarantee is d+5, so d >= 11 for guaranteed girth 16
    # However, actual girth may be larger than guaranteed, so try d from 7 upward
    
    # For odd d: guaranteed girth = d + 5
    # For even d: the construction still works but girth guarantee is weaker
    # Let's try both odd and even d, starting from a minimum
    
    # Minimum d: must have d >= 3, and the graph must be connected
    # For k=3, g=16: try d from 7 to 11 (odd) and 8, 10 (even)
    
    # Actually, let's be systematic: try d from max(3, g-10) to g-3, both parities
    # and verify girth each time
    
    d_min = max(3, g - 10)
    d_max = g - 3  # conservative upper bound
    
    # Try d values in order of increasing n (which is increasing in d)
    # For each d, construct and verify
    for d in range(d_min, d_max + 1):
        if check_deadline():
            break
        
        try:
            n, edges = _luw_construct(k, g, d)
            if n is None:
                continue
            
            # Verify girth >= g using non-backtracking walks
            if _verify_girth_nb(edges, n, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = edges
                # If we found a valid graph, smaller d won't work (we go up in d)
                # But we started from d_min and go up, so the first valid is the smallest
                break
        except Exception:
            continue
    
    # If no tuned LUW worked, fall back to standard d = g-5 (odd)
    if best_n is None:
        d_std = g - 5
        if d_std < 3:
            d_std = 3
        if d_std % 2 == 0:
            d_std += 1
        try:
            best_n, best_edges = _luw_construct(k, g, d_std)
        except Exception:
            # Last resort: use d = g-4 if even, or g-5
            d_alt = g - 4
            if d_alt < 3:
                d_alt = 3
            if d_alt % 2 == 0:
                d_alt += 1
            best_n, best_edges = _luw_construct(k, g, d_alt)
    
    return best_n, best_edges


def _luw_construct(k, g, d):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges) or (None, None) if construction fails.
    """
    q = k
    if d < 3:
        return None, None

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    # Build the rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    n = len(order)
    return n, edges


def _verify_girth_nb(edges, n, g):
    """
    Verify girth >= g using non-backtracking walk enumeration.
    A graph has girth < g iff there exists a non-backtracking walk of length < g
    that returns to its starting vertex (i.e., a cycle of length < g).
    
    We check: for every vertex, enumerate all non-backtracking walks of length up to g-1
    and check if any returns to the start vertex.
    
    For efficiency, we use the fact that in a k-regular graph, the number of
    non-backtracking walks of length L from a vertex is k * (k-1)^(L-1).
    For k=3, L=15: 3 * 2^14 = 49152 per vertex, times n vertices.
    For n=936, that's ~46 million walks. Might be slow in Python.
    
    Alternative: BFS-based girth check. The girth is the minimum over all edges (u,v)
    of (shortest path from u to v not using edge (u,v) + 1).
    """
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check for cycles of length < g
    # For each edge, BFS from one endpoint to the other without using that edge
    # The cycle length = dist + 1. If dist + 1 < g, girth < g.
    
    from collections import deque
    
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    for u, v in edge_set:
        # BFS from u to v, not using edge (u,v)
        dist = {u: 0}
        parent = {u: None}
        queue = deque([u])
        found = False
        
        while queue and not found:
            node = queue.popleft()
            if dist[node] >= g - 1:
                break
            for neighbor in adj[node]:
                # Skip the edge (u,v) in both directions
                if node == u and neighbor == v:
                    continue
                if node == v and neighbor == u:
                    continue
                if neighbor not in dist:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    if neighbor == v:
                        found = True
                        break
                    queue.append(neighbor)
        
        if found:
            cycle_len = dist[v] + 1
            if cycle_len < g:
                return False
    
    return True
# EVOLVE-BLOCK-END
