import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Target: 4-regular, girth >= 10
    # Known optimal: 320 vertices (Boutin graph)
    
    def check_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each vertex to find shortest non-backtracking cycle
        # Since graph is bipartite, we only need to check even girths
        # We check up to min_girth - 2
        
        for start in range(n):
            # Queue: (current_node, prev_node, depth)
            # We start by going to all neighbors
            visited = {} # (node, prev) -> min_depth? No, just need to detect cycle
            # Actually, standard girth check:
            # For each start, do BFS.
            # To prevent backtracking, we track the previous node.
            # State: (node, prev)
            # If we reach a node 'u' from 'v' and 'u' is the start node, and depth >= 3 (cycle length >= 4? No, depth is edges)
            # Cycle length = depth + 1?
            # Walk: start -> ... -> u. If u == start, cycle length is depth.
            # But we must not backtrack.
            
            # Optimized:
            # For each start, for each neighbor n1, do BFS from n1 with prev=start.
            # If we hit start, cycle length is current_depth + 1.
            
            for n1 in adj[start]:
                # BFS from n1, prev is start
                # Queue: (node, prev, depth)
                q = [(n1, start, 1)]
                visited = set()
                visited.add((n1, start))
                while q:
                    curr, prev, depth = q.pop(0)
                    if curr == start and depth >= min_girth:
                        return depth
                    if depth >= min_girth:
                        break
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if (nxt, curr) in visited:
                            continue
                        # If nxt is start, we found a cycle of length depth + 1
                        if nxt == start:
                            return depth + 1
                        visited.add((nxt, curr))
                        q.append((nxt, curr, depth + 1))
        return min_girth + 1 # Assuming no shorter cycle found
        
    def luw_construct(k, g):
        # Fallback to LUW
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

    # Try to construct the 320-vertex graph
    # Bipartite, 160+160
    # Connection set S in Z_160
    # Try known candidate
    candidates = [
        [0, 1, 2, 4],
        [0, 1, 2, 8],
        [0, 1, 3, 4],
        [0, 1, 4, 8],
        [0, 2, 4, 8]
    ]
    
    for S in candidates:
        if time.time() > deadline:
            break
        n = 320
        half = 160
        edges = []
        for p in range(half):
            for s in S:
                l = (p + s) % half
                # p is in 0..159, l is in 160..319
                edges.append((p, half + l))
        
        # Check girth
        # We only need to check if girth >= 10
        # If girth < 10, this candidate is invalid
        # check_girth returns the smallest girth found or min_girth+1 if none smaller
        girth = check_girth(n, edges, 4)
        if girth >= g:
            return n, edges

    # If no candidate worked, fallback to LUW
    return luw_construct(k, g)
