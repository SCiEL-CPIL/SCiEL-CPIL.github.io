import time
import random
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Returns (n, edges) where edges is a list of (u, v) tuples.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # 1. Get a baseline valid graph using Lazebnik-Ustimenko-Woldar (LUW)
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If the LUW graph is small, try to find a smaller one via random search
    #    Cages for small n are often found by random search or backtracking.
    if n_luw <= 100:
        # Moore bound for k-regular graph of girth g
        # For odd girth g: n >= k * (1 + k^(g//2 - 1))
        # For even girth g: n >= 2 * k * (k^(g//2 - 1) - 1) / (k - 1)
        if g % 2 == 1:
            n_min = int(k * (1 + (k ** (g // 2 - 1)))) + 1
        else:
            n_min = int(2 * k * ((k ** (g // 2 - 1) - 1) / (k - 1))) + 1
        
        # Ensure n_min is at least 3
        n_min = max(n_min, 3)
        
        # Try sizes from n_min to n_luw - 1
        for n in range(n_min, n_luw):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Try to find a valid graph of size n
            # Use a time limit for each n
            t_start = time.time()
            found = False
            attempts = 0
            max_attempts = 2000  # Adjust based on time
            
            while time.time() - t_start < 5.0: # 5 seconds per n
                # Generate a random k-regular graph
                edges = _random_k_regular(n, k)
                if edges is None:
                    continue
                
                # Check girth
                if _check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                attempts += 1
                if attempts > max_attempts:
                    break
            
            if found:
                break # Found the smallest possible (so far), stop searching for smaller
    
    return best_n, best_edges

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
    q = k
    # d is chosen such that girth is at least d + 5 (for odd d) or similar
    # The seed code logic: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Rule generation from seed
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
                
    return len(order), edges

def _random_k_regular(n, k):
    """
    Generates a random k-regular graph on n vertices.
    Returns a list of edges or None if impossible.
    """
    if n * k % 2 != 0:
        return None
    if k > n - 1:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, n * k, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            # Self-loop, retry or fail
            # Simple strategy: fail this attempt
            return None
        if (min(u, v), max(u, v)) in seen:
            # Multi-edge, retry or fail
            return None
        seen.add((min(u, v), max(u, v)))
        edges.append((u, v))
        
    return edges

def _check_girth(n, edges, g):
    """
    Checks if the graph has girth at least g.
    Uses BFS from each vertex to detect short cycles.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # For each vertex, do BFS up to depth g//2
    # If we find a node at depth d that was visited at depth d' < d 
    # such that d + d' < g, then there is a cycle of length d + d' < g.
    # Actually, simpler: BFS from each vertex, track distance.
    # If we encounter a neighbor that is not the parent and has been visited,
    # check if dist[u] + dist[v] + 1 < g.
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g // 2:
                continue # No cycle of length < g can be formed from here
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
