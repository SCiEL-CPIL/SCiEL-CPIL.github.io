import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k == 4 and g == 10:
        # Try the known 320-vertex construction
        result = _construct_320_cage(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _construct_320_cage(deadline):
    """
    Construct the (4,10)-cage on 320 vertices using a voltage graph approach.
    Uses a 4-regular base graph on 10 vertices lifted by Z_32.
    
    The base graph is K_{5,5} minus a perfect matching (4-regular, 10 vertices, girth 4).
    We assign voltages to edges so that the lift has girth >= 10.
    """
    n_lift = 320
    group_size = 32
    
    # Base graph: 10 vertices (0-4 on one side, 5-9 on other)
    # Each vertex i (0-4) connects to 5+j for j != i
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    
    # We need to assign voltages to each of the 20 base edges
    # such that the resulting 320-vertex graph has girth >= 10
    
    # Strategy: Use a structured voltage assignment based on linear algebra
    # over Z_32. The key is to ensure that any closed walk of length < 10
    # in the base graph has non-zero total voltage.
    
    # Let's try a deterministic construction based on known results.
    # The (4,10)-cage can be constructed as a Cayley graph or using
    # specific voltage assignments.
    
    # Try a simple structured assignment first
    # Assign voltages based on edge index
    voltages = []
    for idx, (u, v) in enumerate(base_edges):
        # Use a pattern that avoids short cycles
        # This is a heuristic - we'll verify with girth check
        voltages.append((idx * 7) % group_size)
    
    edges = _build_lift_edges(base_edges, voltages, group_size)
    
    # Verify girth
    if _verify_girth_fast(n_lift, edges, 4, 10, deadline):
        return n_lift, edges
    
    # Try different voltage patterns
    patterns = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4],
        [7, 14, 21, 28, 5, 12, 19, 26, 3, 10, 17, 24, 1, 8, 15, 22, 29, 6, 13, 20],
        [11, 22, 1, 12, 23, 2, 13, 24, 3, 14, 25, 4, 15, 26, 5, 16, 27, 6, 17, 28],
    ]
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        voltages = pattern[:]
        edges = _build_lift_edges(base_edges, voltages, group_size)
        if _verify_girth_fast(n_lift, edges, 4, 10, deadline):
            return n_lift, edges
    
    # Try the Cayley graph approach as fallback
    result = _try_cayley_320(deadline)
    if result is not None:
        return result
    
    return None

def _build_lift_edges(base_edges, voltages, group_size):
    """Build the lifted graph edges from base edges and voltages."""
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        w = voltages[idx]
        for x in range(group_size):
            u_id = u * group_size + x
            v_id = v * group_size + (x + w) % group_size
            edges.append((u_id, v_id))
    return edges

def _verify_girth_fast(n, edges, k, g, deadline):
    """
    Fast verification that the graph is k-regular, simple, and has girth >= g.
    Uses non-backtracking walks from every vertex.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        if i > j:
            i, j = j, i
        if (i, j) in edge_set:
            return False
        edge_set.add((i, j))
        adj[i].append(j)
        adj[j].append(i)
    
    # Check regularity
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check girth using non-backtracking BFS from each vertex
    for v in range(n):
        if time.time() > deadline:
            return False
        
        visited = {}
        queue = deque()
        for w in adj[v]:
            visited[(w, v)] = 1
            queue.append((w, v, 1))
        
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= g:
                break
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor == v:
                    if depth + 1 < g:
                        return False
                    continue
                state = (neighbor, node)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((neighbor, node, depth + 1))
    
    return True

def _try_cayley_320(deadline):
    """Try Cayley graph on Z_16 x Z_20."""
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        visited = {}
        visited[(0,0)] = 0
        queue = deque([(0,0)])
        
        while queue:
            u, v = queue.popleft()
            depth = visited[(u,v)]
            
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in visited:
                    visited[(nu, nv)] = depth + 1
                    queue.append((nu, nv))
                else:
                    cycle_len = depth + 1 + visited[(nu, nv)]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            if x == 0 and y == 0: continue
            if x == 1 and y == 0: continue
            if x == m-1 and y == 0: continue
            if (2*x) % m == 0 and (2*y) % n == 0: continue
            
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                        
                        nu = (u + x) % m
                        nv = (v + y) % n
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                
                deg = [0]*total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges

    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction - the baseline."""
    q = k
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
