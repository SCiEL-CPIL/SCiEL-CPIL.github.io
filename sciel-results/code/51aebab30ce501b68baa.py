"""Voltage-graph lift search for 3-regular graphs of large girth.
Starts from small 3-regular base graphs, assigns voltage labels in Z_m,
and uses randomized hill-climbing to maximize the girth of the lifted graph.
Falls back to an explicit LUW construction if the search fails."""
import random
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    if k != 3:
        return _luw_construct(k, g)
    # Try voltage-graph lift search
    best_n, best_edges = None, None
    deadline = time.time() + 100  # leave 20s for fallback
    # Try various base graph sizes and voltage group sizes
    base_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    # For each base size, try voltage groups that give reasonable lifted sizes
    for base_n in base_sizes:
        if base_n < 4:
            continue
        # Base graph must be 3-regular, so base_n must be even
        if base_n % 2 != 0:
            continue
        # Try voltage group sizes
        # Lifted size = base_n * m
        # We want lifted size to be not too large (say < 2000) for efficiency
        max_m = max(2, 2000 // base_n)
        for m in range(2, max_m + 1):
            if time.time() > deadline:
                break
            lifted_n = base_n * m
            # Quick lower bound: if lifted_n is already worse than best, skip
            if best_n is not None and lifted_n >= best_n:
                continue
            # Generate a random base graph and voltage assignment
            result = _try_lift_search(base_n, m, g, deadline)
            if result is not None:
                n, edges = result
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
        if time.time() > deadline:
            break
    if best_n is not None:
        return best_n, best_edges
    # Fallback
    return _luv_construct(k, g)


def _generate_random_3regular(n):
    """Generate a random 3-regular simple graph on n vertices (n even, n>=4)."""
    if n < 4 or n % 2 != 0:
        return None
    # Use configuration model with retries to avoid multiedges/loops
    for _ in range(1000):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * 3)
        random.shuffle(half_edges)
        edges = []
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        if valid and len(edges) == n * 3 // 2:
            return edges
    return None


def _try_lift_search(base_n, m, g, deadline):
    """Try to find a voltage assignment on a base graph that gives girth >= g."""
    # Generate a few random base graphs and try voltage assignments
    for trial in range(50):
        if time.time() > deadline:
            return None
        base_edges = _generate_random_3regular(base_n)
        if base_edges is None:
            continue
        # Build base adjacency
        base_adj = [[] for _ in range(base_n)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
        # Initialize random voltage assignment
        # Each directed edge has a voltage in Z_m
        # We store voltage for each (u,v) where u < v, and use -voltage for (v,u)
        voltages = {}
        for u, v in base_edges:
            voltages[(min(u, v), max(u, v))] = random.randint(0, m - 1)
        # Try to improve via hill climbing
        best_girth = 0
        best_voltages = None
        for iteration in range(200):
            if time.time() > deadline:
                break
            girth = _compute_lifted_girth(base_n, base_adj, voltages, m, limit=g)
            if girth >= g:
                # Success! Build the lifted graph
                return _build_lifted_graph(base_n, base_adj, voltages, m)
            if girth > best_girth:
                best_girth = girth
                best_voltages = {k: v for k, v in voltages.items()}
            # Perturb: change a random voltage
            edge_keys = list(voltages.keys())
            if not edge_keys:
                break
            ek = random.choice(edge_keys)
            old_v = voltages[ek]
            new_v = (old_v + random.randint(1, m - 1)) % m
            voltages[ek] = new_v
        # If we didn't succeed, try next trial
    return None


def _compute_lifted_girth(base_n, base_adj, voltages, m, limit=None):
    """Compute the girth of the lifted graph using BFS in the covering graph.
    limit: if provided, stop early if we find a cycle of length < limit (return that length)
    or return limit if no short cycle found."""
    # The lifted graph has base_n * m vertices
    # Vertex (u, t) where u in 0..base_n-1, t in 0..m-1
    # Edge from (u, t) to (v, t + voltage(u,v) mod m)
    # We need to find the shortest cycle
    # Use BFS from each vertex, but that's expensive.
    # Alternative: use the fact that cycles in the lift correspond to closed walks in the base
    # with voltage sum 0 mod m.
    # For efficiency, we'll do BFS from each vertex up to depth limit-1
    if limit is None:
        limit = 100
    n_lift = base_n * m
    # Build lifted adjacency
    lifted_adj = [[] for _ in range(n_lift)]
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                volt = voltages[(u, v)]
            else:
                volt = (-voltages[(v, u)]) % m
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + ((t + volt) % m)
                lifted_adj[u_lift].append(v_lift)
    # BFS from each vertex to find shortest cycle
    # For each vertex, BFS up to depth limit-1, tracking parent
    best = limit
    for start in range(n_lift):
        # BFS
        dist = [-1] * n_lift
        parent = [-1] * n_lift
        q = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            if dist[u] >= limit - 1:
                continue
            for v in lifted_adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != start and parent[u] != v:
                    # Found a cycle: u -> v is a back edge
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best:
                        best = cycle_len
                        if best < limit:
                            return best
        if best < limit:
            return best
    return best


def _build_lifted_graph(base_n, base_adj, voltages, m):
    """Build the lifted graph edges."""
    n_lift = base_n * m
    edges = []
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                volt = voltages[(u, v)]
            else:
                volt = (-voltages[(v, u)]) % m
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + ((t + volt) % m)
                # Only add each edge once
                if u_lift < v_lift:
                    edges.append((u_lift, v_lift))
    return n_lift, edges


def _luv_construct(k, g):
    """Lazebnik-Ustimenko-Woldar explicit construction (fallback, always valid)."""
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
# EVOLVE-BLOCK-END
