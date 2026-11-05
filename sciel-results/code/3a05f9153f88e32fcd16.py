"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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


def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    for _ in range(100):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * k)

        rng.shuffle(half_edges)

        adj = {v: set() for v in range(n)}
        valid = True

        for i in range(0, n * k, 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)

        if valid:
            return [(min(u, v), max(u, v)) for u in adj for v in sorted(adj[u]) if u < v]

    return None


def _check_girth(adj, n, g):
    """Verify girth >= g by checking no cycle shorter than g exists.
    For each vertex, BFS and check non-tree edges for cycle length < g."""
    half = g // 2
    
    for root in range(n):
        parent = [-1] * n
        dist = [-1] * n
        parent[root] = -2
        dist[root] = 0
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    if dist[node] + 1 <= half:
                        queue.append(neighbor)
                elif parent[node] != neighbor and parent[neighbor] != node:
                    # Non-tree edge: potential cycle
                    if dist[node] + dist[neighbor] + 1 < g:
                        return False
    
    return True


def _construct_lift(base_edges, base_n, m, voltage_seed, k):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    n = base_n * m
    adj = {i: set() for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            adj[u_lift].add(v_lift)
            adj[v_lift].add(u_lift)
    
    # Check degree and simplicity
    for i in range(n):
        if len(adj[i]) != k:
            return None, None
    
    edges = [(min(u, v), max(u, v)) for u in adj for v in sorted(adj[u]) if u < v]
    return n, edges


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    best_n = float('inf')
    best_edges = None
    
    # Try voltage graph lifts with small base graphs
    base_sizes = [10, 12, 14, 16]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            m_values = []
            for target_n in [500, 600, 700, 800, 900, 1000, 1100]:
                m = target_n // base_n
                if m >= 2:
                    m_values.append(m)
            m_values = list(set(m_values))
            m_values.sort()
            
            for m in m_values:
                if time.time() > deadline:
                    break
                
                for vs in range(5):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, base_n, m, seed + vs, k)
                    if n is None:
                        continue
                    
                    adj = {i: set() for i in range(n)}
                    for u, v in edges:
                        adj[u].add(v)
                        adj[v].add(u)
                    
                    if _check_girth(adj, n, g) and n < best_n:
                        best_n = n
                        best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges
# EVOLVE-BLOCK-END
