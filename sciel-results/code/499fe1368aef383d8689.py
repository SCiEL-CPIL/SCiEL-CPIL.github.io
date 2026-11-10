import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Moore bound for lower bound
    if k == 1:
        moore = 2
    else:
        if g % 2 == 0:
            R = g // 2
            moore = 2 * k * (k - 1) ** (R - 1)
        else:
            R = (g - 1) // 2
            moore = k * (k - 1) ** R + 1

    # 1. LUW construction
    def get_luw_graph():
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    try:
        luw_n, luw_edges = get_luw_graph()
    except Exception:
        luw_n, luw_edges = None, None

    best_n = luw_n if luw_n is not None else 10 ** 9
    best_edges = luw_edges if luw_edges is not None else []

    # 2. Try to find smaller graph via edge-swap local search
    # Start near the LUW size or a bit below
    if luw_n is not None and luw_n > moore:
        target_n = max(moore, luw_n - 50)
    else:
        target_n = max(moore, 100)

    if target_n < best_n and target_n >= 2 * k:
        # Generate random k-regular graph
        def gen_random_reg(n, k):
            for _ in range(20):
                half_edges = [i for i in range(n) for _ in range(k)]
                random.shuffle(half_edges)
                adj = [set() for _ in range(n)]
                edges = []
                valid = True
                for i in range(0, n * k, 2):
                    u = half_edges[i]
                    v = half_edges[i + 1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                if valid and len(edges) == n * k // 2:
                    return edges
            return None

        edges = gen_random_reg(target_n, k)
        if edges:
            adj = [set() for _ in range(target_n)]
            edge_set = set()
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((min(u, v), max(u, v)))

            # Girth check via non-backtracking BFS from each vertex
            def has_short_cycle(limit):
                for start in range(target_n):
                    # BFS with parent tracking to avoid backtracking
                    visited = {}
                    queue = [(start, -1, 0)]  # (node, parent, dist)
                    while queue:
                        node, parent, dist = queue.pop(0)
                        if dist >= limit:
                            continue
                        for nb in adj[node]:
                            if nb == parent:
                                continue
                            if nb not in visited:
                                visited[nb] = (node, dist + 1)
                                queue.append((nb, node, dist + 1))
                            else:
                                # Found a cycle
                                pnode, pdist = visited[nb]
                                if pnode != node:
                                    cycle_len = dist + 1 + pdist
                                    if cycle_len < limit:
                                        return True
                    if not visited:
                        continue
                return False

            # Local search: swap edges to eliminate short cycles
            t = 1.0
            cooling = 0.995
            while time.time() < deadline:
                if not has_short_cycle(g):
                    best_n = target_n
                    best_edges = list(edge_set)
                    break
                # Find a short cycle and try to break it
                # For simplicity, do random swaps
                swapped = False
                for _ in range(100):
                    u = random.randint(0, target_n - 1)
                    v = random.choice(list(adj[u]))
                    x = random.randint(0, target_n - 1)
                    if x == u or x == v:
                        continue
                    y = random.choice(list(adj[x]))
                    if y == u or y == v or x == u or x == v:
                        continue
                    if (min(u, x), max(u, x)) in edge_set or (min(v, y), max(v, y)) in edge_set:
                        continue
                    # Perform swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    adj[u].add(x)
                    adj[x].add(u)
                    adj[v].add(y)
                    adj[y].add(v)
                    edge_set.remove((min(u, v), max(u, v)))
                    edge_set.remove((min(x, y), max(x, y)))
                    edge_set.add((min(u, x), max(u, x)))
                    edge_set.add((min(v, y), max(v, y)))
                    swapped = True
                    break
                if not swapped:
                    t *= cooling
                    if t < 0.01:
                        break
                else:
                    t *= cooling

    return best_n, best_edges
