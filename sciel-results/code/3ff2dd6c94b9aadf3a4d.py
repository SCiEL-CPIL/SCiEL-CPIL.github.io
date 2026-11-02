import time
import sys

def luw_construct(k, d):
    """
    Constructs the Lazebnik-Ustimenko-Woldar graph CD(d, k).
    Returns (n, edges).
    """
    q = k
    # d must be at least 2
    if d < 2:
        d = 2
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    # Define the rule for coordinates
    # rule[c] = (a, b) such that y[c] depends on y[a] and x[b] (or vice versa)
    # Based on the seed program logic
    rule = {}
    if d >= 2: rule[2] = (1, 1)
    if d >= 3: rule[3] = (2, 1)
    if d >= 4: rule[4] = (1, 2)
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        # The list in seed: [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        # Index r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # Seed: start = (0, (None,) + (0,) * d) -> length d+1.
        # Index 0 is None, indices 1..d are values.
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                    # y is line, x is point.
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line l ~ point p: p[c] - l[c] = l[a] * p[b] ? 
                    # Seed: y[c] = sub(x[c], mul(x[a], y[b]))
                    # Here x is line, y is point.
                    # p[c] = l[c] + l[a] * p[b] ? No.
                    # The relation is symmetric in a specific way.
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def check_girth(n, edges, g):
    """
    Checks if the graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = (g - 1) // 2
    
    for start in range(n):
        # BFS on non-backtracking walks: state is (node, prev_node)
        # Track depth and whether we've seen this state at a shallower depth
        visited = {}  # (node, prev) -> depth
        queue = [(start, -1, 0)]
        while queue:
            node, prev, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                state = (nb, node)
                if state in visited:
                    # We've been here before at some depth
                    # A cycle exists of length (depth+1) - visited[state] + 1? 
                    # Actually, the cycle length is depth + 1 - visited[state]
                    # Wait: we're at depth+1 from start via non-backtracking walk.
                    # We were at this state at depth visited[state].
                    # The cycle length is (depth + 1) - visited[state]
                    cycle_len = (depth + 1) - visited[state]
                    if cycle_len < g:
                        return False
                else:
                    visited[state] = depth + 1
                    queue.append((nb, node, depth + 1))
    return True

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s kill
    
    # Determine candidate d values
    # LUW girth >= d+4, so d >= g-4 guarantees validity without checking.
    candidates = []
    for d in range(max(2, g - 4), g + 5):
        candidates.append(d)
    
    best_n = float('inf')
    best_edges = None
    
    for d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges = luw_construct(k, d)
        except Exception:
            continue
        
        if d + 4 >= g:
            # Theoretically guaranteed valid
            if n < best_n:
                best_n = n
                best_edges = edges
        else:
            # Need to verify girth
            if time.time() > deadline:
                break
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: construct with largest d
    max_d = candidates[-1]
    n, edges = luw_construct(k, max_d)
    return n, edges
