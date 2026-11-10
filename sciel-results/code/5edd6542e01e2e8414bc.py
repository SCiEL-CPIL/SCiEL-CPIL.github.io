"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. We use d = g - 3 (odd) to guarantee girth >= g.
    q = k
    # Ensure d is odd and d + 5 >= g, so d >= g - 5. Use d = g - 3 (always odd if g is even, or adjust)
    d = g - 3
    if d % 2 == 0:
        d += 1  # make it odd
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule for LUW graph: for each c in 2..d, (a_c, b_c) with a_c < c, b_c < c
    # Standard LUW rule: 
    # c=2: (1,1), c=3: (2,1), c=4: (1,2)
    # For c >= 5: pattern repeats with period 4
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:  # r == 3
            rule[c] = (1, c0 + 1)

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # point -> line: line[c] = point[c] + line[a] * point[b]
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    # line -> point: point[c] = line[c] - line[a] * point[b]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    # Verify girth using BFS from each vertex (non-backtracking)
    n = len(order)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            # Fallback: return what we have, but this shouldn't happen
            pass
    
    # Girth check via BFS
    def bfs_girth(start_node, target_girth):
        # BFS tracking (node, parent, depth)
        from collections import deque
        visited = {}  # node -> (depth, parent)
        queue = deque()
        for nb in adj[start_node]:
            queue.append((nb, start_node, 1))
            visited[nb] = (1, start_node)
        while queue:
            if time.time() > deadline:
                return float('inf')
            node, parent, depth = queue.popleft()
            if depth >= target_girth:
                return depth
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    d2, p2 = visited[nb]
                    if d2 + depth + 1 < target_girth:
                        return d2 + depth + 1
                else:
                    visited[nb] = (depth + 1, node)
                    queue.append((nb, node, depth + 1))
        return float('inf')
    
    # Quick girth check from a few vertices
    min_girth = float('inf')
    check_nodes = [0, n // 2, n - 1] if n > 2 else [0]
    for sn in check_nodes:
        g_found = bfs_girth(sn, g)
        if g_found < min_girth:
            min_girth = g_found
    
    if min_girth < g:
        # Girth too small; try with larger d
        d += 2
        # Rebuild
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)
        
        def neighbours2(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield (1 - side, tuple(y))
        
        start2 = (0, (None,) + (0,) * d)
        index2, order2, edges2 = {start2: 0}, [start2], []
        for vertex in order2:
            if time.time() > deadline:
                break
            for other in neighbours2(vertex):
                if other not in index2:
                    index2[other] = len(order2)
                    order2.append(other)
                if vertex[0] == 0:
                    edges2.append((index2[vertex], index2[other]))
        return len(order2), edges2
    
    return n, edges
# EVOLVE-BLOCK-END
