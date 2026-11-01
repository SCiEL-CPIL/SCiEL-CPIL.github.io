# EVOLVE-BLOCK-START
def construct(k, g):
    import time
    from collections import deque

    deadline = time.time() + 140.0

    def make_rule(d):
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return rule

    def luw_graph(q, d, rule):
        # Convert rule to list for faster access
        rule_list = [None] * (d + 1)
        for c in range(2, d + 1):
            rule_list[c] = rule[c]

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # Pre-allocate y list? No, size depends on d.
        
        for i in range(len(order)):
            vertex = order[i]
            side, x = vertex
            # x is tuple
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                if side == 0:
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    for c in range(2, d + 1):
                        a, b = rule_list[c]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1 - side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for v in range(n):
            visited = set()
            visited.add((v, -1))
            queue = deque([(v, -1, 1)])
            
            while queue:
                cur, prev, length = queue.popleft()
                if length >= g:
                    continue
                for w in adj[cur]:
                    if w == prev:
                        continue
                    new_state = (w, cur)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    if w == v and length + 1 >= 2:
                        return False
                    if length + 1 < g:
                        queue.append((w, cur, length + 1))
        return True

    q = k
    min_d = (g - 1) // 2 + (1 if (g - 1) % 2 else 0)

    best_n = float('inf')
    best_edges = None

    for d in range(min_d, min_d + 5):
        if time.time() > deadline:
            break
        rule = make_rule(d)
        n, edges = luw_graph(q, d, rule)
        if n < best_n:
            if check_girth(n, edges, g):
                best_n = n
                best_edges = edges

    if best_edges is None:
        d = min_d
        rule = make_rule(d)
        best_n, best_edges = luw_graph(q, d, rule)

    return best_n, best_edges
# EVOLVE-BLOCK-END
