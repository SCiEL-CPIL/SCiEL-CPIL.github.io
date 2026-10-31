"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g using random voltage lifts of K_{3,3}."""
    deadline = time.time() + 115
    
    if k != 3:
        # Fallback for k != 3: LUW construction
        return _luw_construct(k, g)
    
    # K_{3,3} base graph: 6 vertices, 3-regular, girth 4
    # Part A: vertices 0, 1, 2
    # Part B: vertices 3, 4, 5
    # Edges: (A_i, B_j) for all i, j
    base_edges = [(0, 3), (0, 4), (0, 5),
                  (1, 3), (1, 4), (1, 5),
                  (2, 3), (2, 4), (2, 5)]
    
    def check_girth_bounded(adj, n, min_girth):
        """Check girth >= min_girth using BFS from each vertex, bounded depth."""
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def build_k33_lift(m, voltage_func):
        """Build a voltage lift of K_{3,3} over Z_m.
        voltage_func: for each base edge (u,v), returns a permutation of Z_m
        such that vertex (u, i) connects to (v, perm(i))."""
        n = 6 * m
        adj = [[] for _ in range(n)]
        
        for u, v in base_edges:
            perm = voltage_func(u, v)
            for i in range(m):
                u_idx = u * m + i
                v_idx = v * m + perm[i]
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
        
        return adj
    
    def random_voltage_func(m):
        """Generate random permutation voltage labels for each base edge of K_{3,3}."""
        perms = {}
        for u, v in base_edges:
            p = list(range(m))
            random.shuffle(p)
            perms[(u, v)] = p
            inv = [0] * m
            for i in range(m):
                inv[p[i]] = i
            perms[(v, u)] = inv
        return lambda u, v: perms[(u, v)]
    
    best_result = None
    
    # Try increasing lift sizes m. n = 6*m for K_{3,3} lifts.
    # For girth 16, need to find m such that random lift has girth >= 16.
    m_values = sorted(set([
        30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
        240, 280, 320, 360, 400, 450, 500, 550, 600, 700, 800, 900,
        1000, 1100, 1200, 1400, 1600, 1800, 2000, 2500, 3000
    ]))
    
    for m in m_values:
        if time.time() > deadline:
            break
        
        n = 6 * m
        # More trials for smaller m where success probability is lower
        n_trials = max(1, int(40 / max(1, m / 100)))
        n_trials = min(n_trials, 60)
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            voltage_func = random_voltage_func(m)
            adj = build_k33_lift(m, voltage_func)
            
            if check_girth_bounded(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """LUW construction as fallback."""
    d, q = _luw_params(k, g)
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

def _luw_params(k, g):
    """Compute LUW parameters."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return d, q
# EVOLVE-BLOCK-END
