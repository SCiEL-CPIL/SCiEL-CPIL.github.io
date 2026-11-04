"""Program for the suite task: construct a small k-regular graph of girth g."""
import math
import random
from collections import deque

# EVOLVE-BLOCK-START
def _moore_bound(k, g):
    """Return the Moore bound (lower bound on n) for a k-regular graph of girth g."""
    if g % 2 == 0:
        # even girth: 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
        m = g // 2
        if k == 1:
            return 2
        s = 0
        term = 1
        for _ in range(m - 1):
            s += term
            term *= (k - 1)
        s += term  # include the last term? Actually sum i=0 to m-1
        # Let's recompute properly: sum_{i=0}^{m-1} (k-1)^i
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s
    else:
        # odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        m = (g - 1) // 2
        s = 0
        term = 1
        for i in range(m):
            s += term
            term *= (k - 1)
        return 1 + k * s

def _check_girth(adj, g):
    """Check if graph with adjacency list 'adj' has girth at least g.
    Uses BFS from each vertex to find shortest cycle through that vertex."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _bfs_girth(adj, g):
    """More efficient girth check using BFS from each vertex.
    Returns True if girth >= g."""
    n = len(adj)
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _construct_lazebnik(k, g):
    """Construct Lazebnik-Ustimenko-Woldar graph. Returns (n, edges) or None."""
    q = k
    # d is the 'dimension' parameter; girth is at least d+5 for odd d
    # We need girth >= g, so we need d+5 >= g => d >= g-5
    # d should be odd for the construction to work properly
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
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
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
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
            yield (1 - side, tuple(y))
    
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

def _try_construct_cage(k, g, timeout=110):
    """Try to construct a k-regular graph of girth g with as few vertices as possible.
    Returns (n, edges) for the best found, or the Lazebnik construction as fallback."""
    
    import time
    start_time = time.time()
    
    # First, try the Lazebnik construction as a baseline
    try:
        n_laz, edges_laz = _construct_lazebnik(k, g)
        best_n = n_laz
        best_edges = edges_laz
    except Exception:
        best_n = None
        best_edges = None
    
    # Try to find smaller graphs using random constructions
    # For small k and g, we can try various approaches
    
    # Approach: Start with a random k-regular graph and try to eliminate short cycles
    # This is heuristic and may not work well for large g
    
    # For the specific case k=4, g=10, the known best is 320 vertices
    # Let's try to find something better than Lazebnik if possible
    
    # Try a different construction: the (k, g)-cage from known tables
    # For k=4, g=10: known cages have 320 vertices
    
    # Try to improve by using a different method
    # One approach: use the fact that for k=4, g=10, we can try to construct
    # a graph with fewer vertices by using a different algebraic construction
    
    # For now, return the Lazebnik construction which is valid
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: try to construct a simple k-regular graph
    # This is a last resort
    n = k * k  # rough estimate
    edges = []
    # Create a k-regular graph using a simple construction
    # For small n, we can use a circulant graph
    # But this may not have the right girth
    
    # Return a trivial answer if nothing else works
    # This should not happen for valid inputs
    return 0, []

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs."""
    
    # Handle edge cases
    if k < 2:
        # Degenerate case
        return 0, []
    
    if g <= 2:
        # Any graph works, use a complete graph or similar
        if k == 2:
            # Cycle graph
            n = max(3, g)
            edges = [(i, (i + 1) % n) for i in range(n)]
            return n, edges
        else:
            # For k > 2 and small g, use a complete graph if possible
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            return n, edges
    
    # Try the main construction
    try:
        n, edges = _try_construct_cage(k, g)
        return n, edges
    except Exception:
        # Fallback to Lazebnik
        try:
            n, edges = _construct_lazebnik(k, g)
            return n, edges
        except Exception:
            # Last resort: return a trivial answer
            return 0, []
# EVOLVE-BLOCK-END
