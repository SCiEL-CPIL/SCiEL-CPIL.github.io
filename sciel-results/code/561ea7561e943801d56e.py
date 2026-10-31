"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g using random lifts of Petersen graph."""
    deadline = time.time() + 115
    
    if k != 3:
        # Fallback for k != 3: LUW construction
        return _luw_construct(k, g)
    
    # Petersen graph: 10 vertices, 3-regular, girth 5
    # Vertices 0-4: outer pentagon, 5-9: inner star
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),  # spokes
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)   # inner star
    ]
    petersen_adj = [set() for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].add(v)
        petersen_adj[v].add(u)
    
    # Sort adjacency lists for consistent labeling
    petersen_adj_sorted = [sorted(adj) for adj in petersen_adj]
    
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
    
    def build_lift(m, label_func):
        """Build a random lift of Petersen graph of size m.
        label_func: for each base edge (u,v), returns a permutation of Z_m
        such that vertex (u, i) connects to (v, perm(i))."""
        n = 10 * m
        adj = [[] for _ in range(n)]
        
        for u, v in petersen_edges:
            perm = label_func(u, v)
            for i in range(m):
                u_idx = u * m + i
                v_idx = v * m + perm[i]
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
        
        return adj
    
    def random_label_func(m):
        """Generate random permutation labels for each base edge."""
        import random as _rnd
        perms = {}
        for u, v in petersen_edges:
            p = list(range(m))
            _rnd.shuffle(p)
            perms[(u, v)] = p
            # Also store reverse for (v, u)
            inv = [0] * m
            for i in range(m):
                inv[p[i]] = i
            perms[(v, u)] = inv
        return lambda u, v: perms[(u, v)]
    
    # Strategy: try increasing lift sizes m
    # For girth 16, we need girth >= 16, so BFS depth up to 7 from each vertex
    # Random lifts of Petersen: expected girth grows as ~log(m)
    # Try m from small to larger values
    
    best_result = None
    
    # Start with reasonable m values. For girth 16, m needs to be large enough.
    # Empirically, m ~ 100-500 should give girth ~ 12-20 for random lifts.
    # We'll try a range of m values.
    
    m_values = []
    # Start from smaller m and go up
    for m in [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]:
        m_values.append(m)
    
    # Also include some intermediate values
    for m in [70, 90, 110, 130, 140, 160, 170, 180, 190, 220, 240, 260, 280, 350, 450, 550, 700, 900, 1100, 1300, 1400, 1600, 1700, 1800, 1900]:
        m_values.append(m)
    
    # Sort and deduplicate
    m_values = sorted(set(m_values))
    
    for m in m_values:
        if time.time() > deadline:
            break
        
        n = 10 * m
        # Try multiple random lifts at this size
        n_trials = max(1, int(30 / (m / 100)))  # More trials for smaller m
        n_trials = min(n_trials, 50)
        
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            label_func = random_label_func(m)
            adj = build_lift(m, label_func)
            
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
