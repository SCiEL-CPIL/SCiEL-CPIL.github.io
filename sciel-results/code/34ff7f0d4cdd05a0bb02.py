"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # First, generate the base graph using Lazebnik-Ustimenko-Woldar construction
    base_n, base_edges = _generate_luw_graph(k, g)
    
    if time.time() > deadline - 10:
        return base_n, base_edges
    
    # Try to improve by finding smaller valid graphs through random search
    best_n = base_n
    best_edges = base_edges
    
    # Generate random 3-regular graphs and check girth
    attempts = 0
    while time.time() < deadline - 15 and attempts < 50:
        attempts += 1
        
        # Try different sizes starting from Moore bound
        moore_bound = _moore_bound(k, g)
        
        # Try a few sizes between Moore bound and current best
        for n in range(moore_bound, min(best_n, moore_bound + 200)):
            if time.time() > deadline - 15:
                break
            
            # Generate random k-regular graph
            n_try, edges_try = _generate_random_k_regular(n, k)
            if n_try is None:
                continue
            
            # Check girth
            girth = _compute_girth(n_try, edges_try)
            if girth >= g:
                best_n = n_try
                best_edges = edges_try
                break
    
    return best_n, best_edges

def _generate_luw_graph(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def _moore_bound(k, g):
    # Moore bound for k-regular graphs of girth g
    if g % 2 == 0:
        # Even girth: n >= k + k(k-1)^(g/2 - 1)
        return k + k * (k - 1) ** (g // 2 - 1)
    else:
        # Odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        total = 1
        term = 1
        for i in range((g - 3) // 2 + 1):
            total += k * term
            term *= (k - 1)
        return total

def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if n * k % 2 != 0:
        return None, None
    
    # Create stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    
    # Try multiple times to get a simple graph
    for attempt in range(100):
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            edge = (min(u, v), max(u, v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return n, list(edges)
    
    return None, None

def _compute_girth(n, edges):
    """Compute the girth (shortest cycle length) using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS from start, tracking parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    min_cycle = min(min_cycle, cycle_len)
    
    return min_cycle if min_cycle != float('inf') else 0
# EVOLVE-BLOCK-END
