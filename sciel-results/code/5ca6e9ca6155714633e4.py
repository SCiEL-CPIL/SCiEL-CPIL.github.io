import time
import math
import random

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small random 4-regular graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    if k != 4 or g != 10:
        # Fallback to generic LUW for other parameters if needed, 
        # but the task specifies this task's degree and girth.
        return _luw_construct(k, g)

    # Target size 320 (known smallest cage)
    target_n = 320
    
    def _gen_random_4reg(n):
        """Generates a random 4-regular graph on n vertices using rejection with retry on individual edges."""
        # Use a smarter approach: generate stubs, shuffle, then pair with local repair
        for attempt in range(50):
            stubs = []
            for i in range(n):
                for _ in range(4):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if u > v:
                    u, v = v, u
                if (u, v) in edges:
                    valid = False
                    break
                edges.add((u, v))
            
            if valid and len(edges) == 2 * n:
                return list(edges)
        
        return None

    def _has_girth_at_least(n, edges, g):
        """Checks if graph has girth at least g using non-backtracking walk enumeration."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # For girth >= g, we need no cycles of length < g.
        # For g=10, we need no cycles of length 3,4,5,6,7,8,9.
        # Equivalent: for every vertex s, the non-backtracking walks of length up to g-1
        # should not return to s via a different path.
        # 
        # Simpler approach: BFS from each vertex, checking for early returns.
        # A cycle of length L exists iff there is a vertex s and two distinct 
        # non-backtracking paths of total length L-1 from s back to s.
        # 
        # For g=10: check that no vertex is reachable from itself via non-backtracking
        # walks of length 2..8 (which would give cycles of length 3..9).
        # 
        # Efficient check: for each vertex s, do a BFS tracking (node, prev) pairs.
        # If we reach (s, x) where x != parent of s in the BFS tree, and the path length + 1 < g,
        # we have a short cycle.
        
        for s in range(n):
            # BFS tracking (current_node, previous_node) to avoid backtracking
            # dist[(u,v)] = length of non-backtracking walk from s ending at u coming from v
            visited = set()
            visited.add((s, -1))  # start at s with no previous
            queue = [(s, -1, 0)]  # (node, prev, depth)
            qi = 0
            while qi < len(queue):
                u, prev, d = queue[qi]
                qi += 1
                if d >= g - 1:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue  # non-backtracking
                    if v == s and d + 1 < g:
                        return False
                    if (v, u) not in visited:
                        visited.add((v, u))
                        queue.append((v, u, d + 1))
        return True

    # Search loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        edges = _gen_random_4reg(target_n)
        if edges is None:
            continue
        
        if _has_girth_at_least(target_n, edges, g):
            return target_n, edges
            
    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW code
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
