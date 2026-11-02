<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a two-phase approach:
    1. Start with a deterministic base (LUW-like structure for small k, or random 3-regular for k=3)
    2. Improve girth via 2-opt swaps (edge exchanges) that maintain regularity and simplicity
    """
    deadline = time.time() + 100  # 100s safety margin
    
    # Phase 1: Generate initial graph
    if k == 3:
        n, edges = _generate_random_3_reg(g)
    else:
        n, edges = _generate_luw_base(k, g)
    
    if n == 0:
        # Fallback to LUW if random generation failed
        n, edges = _generate_luw_base(k, g)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
        adj[u].append(v)
        adj[v].append(u)
    
    # Phase 2: Improve girth via 2-opt swaps
    best_girth = _compute_girth(adj, n, g)
    best_edges = [(u, v) for u, v in edge_set if u < v]
    
    while time.time() < deadline:
        improved = False
        # Try random 2-opt swaps
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            # Pick two random edges
            if len(edge_set) < 2:
                break
            e1, e2 = random.sample(list(edge_set), 2)
            u1, v1 = e1
            u2, v2 = e2
            
            # Check if swapping creates valid edges
            # Remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
            for new_e1, new_e2 in [((u1, u2), (v1, v2)), ((u1, v2), (v1, u2))]:
                a1, b1 = new_e1
                a2, b2 = new_e2
                if a1 == b1 or a2 == b2:
                    continue
                if (a1, b1) in edge_set or (a2, b2) in edge_set:
                    continue
                if (min(a1, b1), max(a1, b1)) == e1 or (min(a2, b2), max(a2, b2)) == e2:
                    continue
                
                # Apply swap
                edge_set.remove(e1)
                edge_set.remove(e2)
                edge_set.add((min(a1, b1), max(a1, b1)))
                edge_set.add((min(a2, b2), max(a2, b2)))
                
                # Rebuild adjacency
                adj = [[] for _ in range(n)]
                for u, v in edge_set:
                    adj[u].append(v)
                    adj[v].append(u)
                
                # Check girth
                new_girth = _compute_girth(adj, n, g)
                if new_girth > best_girth:
                    best_girth = new_girth
                    best_edges = [(u, v) for u, v in edge_set if u < v]
                    improved = True
                    if best_girth >= g:
                        return n, best_edges
                else:
                    # Revert
                    edge_set.remove((min(a1, b1), max(a1, b1)))
                    edge_set.remove((min(a2, b2), max(a2, b2)))
                    edge_set.add(e1)
                    edge_set.add(e2)
        
        if best_girth >= g:
            break
    
    return n, best_edges


def _generate_luw_base(k, g):
    """Generate base graph using LUW construction."""
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


def _generate_random_3_reg(g):
    """Generate a random 3-regular graph with good initial girth."""
    random.seed(42)
    
    # Start with a cycle and add chords
    n = max(16, g * 2)  # Start with enough vertices
    n += (n % 2)  # Ensure even
    
    edges = set()
    # Create a cycle
    for i in range(n):
        edges.add((i, (i + 1) % n))
    
    # Add random edges to make 3-regular
    degree = [1] * n
    for u in range(n):
        degree[u] = 2  # Each vertex has 2 neighbors in cycle
    
    attempts = 0
    while any(d < 3 for d in degree) and attempts < 10000:
        attempts += 1
        u = random.randint(0, n - 1)
        if degree[u] >= 3:
            continue
        v = random.randint(0, n - 1)
        if v == u or v == (u + 1) % n or v == (u - 1) % n:
            continue
        if (min(u, v), max(u, v)) in edges:
            continue
        if degree[v] >= 3:
            continue
        
        edges.add((min(u, v), max(u, v)))
        degree[u] += 1
        degree[v] += 1
    
    if any(d != 3 for d in degree):
        return 0, []
    
    return n, list(edges)


def _compute_girth(adj, n, limit):
    """Compute girth using BFS from each vertex, stopping at depth limit//2."""
    max_depth = limit // 2 + 1
    min_cycle = float('inf')
    
    for start in range(n):
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for v in queue:
            if dist[v] >= max_depth:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    parent[u] = v
                    queue.append(u)
                elif u != parent[v] and dist[u] < dist[v]:
                    # Found a cycle
                    cycle_len = dist[v] + dist[u] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle <= limit:
                            return min_cycle
    
    return min_cycle if min_cycle < float('inf') else limit + 1
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
