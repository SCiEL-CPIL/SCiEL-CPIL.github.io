"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    if k != 3:
        return _fallback_construct(k, g)

    # Voltage lift of K_{3,3} over Z_m
    # n = 6m
    
    def count_4_cycles(m, voltages):
        count = 0
        for i in range(3):
            for k in range(3):
                if i == k: continue
                for j in range(3):
                    for l in range(3):
                        if j == l: continue
                        s = (voltages[i][j] - voltages[k][j] + voltages[k][l] - voltages[i][l]) % m
                        if s == 0:
                            count += 1
        return count

    def check_girth(m, voltages, required_girth):
        n = 6 * m
        adj = [[] for _ in range(n)]
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    adj[u].append(w)
                    adj[w].append(u)
                    
        half_girth = required_girth // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= half_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < required_girth:
                            return False
        return True

    def build_graph(m, voltages):
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    edges.append((u, w))
        return n, edges

    random.seed(42)
    
    # Moore bound for k=3, g=16 is 512. m_min = 86.
    # Best known 936 (m=156).
    # We search m from 86 to 160.
    
    for m in range(86, 161):
        # Local search to minimize 4-cycles
        voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
        current_cost = count_4_cycles(m, voltages)
        
        for step in range(500):
            if current_cost == 0:
                break
            i = random.randint(0, 2)
            j = random.randint(0, 2)
            old_v = voltages[i][j]
            new_v = random.randint(0, m - 1)
            voltages[i][j] = new_v
            new_cost = count_4_cycles(m, voltages)
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                voltages[i][j] = old_v
        
        if current_cost == 0:
            if check_girth(m, voltages, g):
                return build_graph(m, voltages)
        
    return _fallback_construct(k, g)

def _fallback_construct(k, g):
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
# EVOLVE-BLOCK-END
