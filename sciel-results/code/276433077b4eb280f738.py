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
    
    # Configuration model: create n*k half-edges, pair them randomly
    half_edges = []
    for v in range(n):
        half_edges.extend([v] * k)
    
    rng.shuffle(half_edges)
    
    edges = set()
    adj = {v: set() for v in range(n)}
    
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            continue  # skip loops
        if u in adj[v]:
            continue  # skip multi-edges
        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if we got a valid k-regular graph
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    return list(edges)


def _check_girth_bfs(adj, g):
    """Check if graph has girth at least g using BFS from each vertex up to depth g//2."""
    n = len(adj)
    half_g = g // 2
    
    for start in range(n):
        # BFS from start
        visited = {start: (None, 0)}  # node -> (parent, depth)
        queue = deque([(start, None, 0)])
        
        while queue:
            node, parent, depth = queue.popleft()
            
            if depth >= half_g:
                continue
            
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a cycle
                    # The cycle length is depth + 1 + visited[neighbor][1]
                    # Actually need to be more careful
                    pass
                else:
                    visited[neighbor] = (node, depth + 1)
                    queue.append((neighbor, node, depth + 1))
    
    # More precise girth check: find shortest cycle
    # For each edge, temporarily remove it and check if there's a path between its endpoints
    # But that's O(E * (V+E)) which might be slow for large graphs
    
    # Alternative: BFS from each vertex, tracking parents
    for start in range(n):
        parent = [-1] * n
        dist = [-1] * n
        parent[start] = -2
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif parent[node] != neighbor and parent[neighbor] != node:
                    # Found a back edge indicating a cycle
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    
    return True


def _construct_lift(base_edges, m, voltage_seed):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    # Get base vertices
    base_vertices = set()
    for u, v in base_edges:
        base_vertices.add(u)
        base_vertices.add(v)
    base_vertices = sorted(base_vertices)
    base_n = len(base_vertices)
    base_idx = {v: i for i, v in enumerate(base_vertices)}
    
    # Assign voltages: each edge gets a random voltage in Z_m
    # Use the edge index to seed
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    # Construct lift: n = base_n * m vertices
    n = base_n * m
    
    # Lift vertex (v, a) maps to v * m + a
    lift_edges = set()
    adj = {i: [] for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = base_idx[u] * m + a
            v_lift = base_idx[v] * m + (a + vol) % m
            lift_edges.add((min(u_lift, v_lift), max(u_lift, v_lift)))
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None, None  # multi-edges exist
        if len(adj[i]) < 5:
            return None, None
    
    return n, list(lift_edges)


def _check_lift_girth(base_edges, m, voltage_seed, g):
    """Construct lift and check girth."""
    n, edges = _construct_lift(base_edges, m, voltage_seed)
    if n is None:
        return None, None, False
    
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check girth using BFS
    if not _check_girth_bfs(adj, g):
        return n, edges, False
    
    return n, edges, True


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # First try voltage graph lifts with small base graphs
    # The base graph should be k-regular
    # Try different base sizes and lift sizes
    
    best_n = float('inf')
    best_edges = None
    
    # Try base sizes from small to larger
    # For k=5, g=9: Moore bound is around 428, best known is 1116
    # A base of size 10 with lift m=100 gives n=1000
    # A base of size 12 with lift m=84 gives n=1008
    
    base_sizes = [10, 12, 14, 16, 18, 20]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        # Generate several random base graphs
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try different lift sizes
            # We want n = base_n * m to be around the target
            # Try m values that give reasonable total sizes
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
                
                # Try several voltage assignments
                for vs in range(10):
                    if time.time() > deadline:
                        break
                    
                    n, edges, valid = _check_lift_girth(base_edges, m, seed + vs, g)
                    if valid and n < best_n:
                        best_n = n
                        best_edges = edges
    
    # If we found a valid lift, return it
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges
# EVOLVE-BLOCK-END
