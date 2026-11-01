import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 115.0

    # Fallback: LUW graph construction
    def luw_construct(k, g):
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

    # Q4 base graph for (4,10) cage
    def get_q4():
        # Q4 has 16 vertices, 4-regular, girth 4
        edges = []
        for i in range(16):
            for j in range(4):
                if i & (1 << j):
                    continue
                v = i | (1 << j)
                edges.append((i, v))
        return edges

    def get_adj(edges, n):
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        return adj

    def enumerate_closed_walks(adj, n, max_len):
        walks = []
        for start in range(n):
            stack = [(start, [])]
            while stack:
                v, path = stack.pop()
                L = len(path)
                if L > 0 and v == start and L < max_len:
                    walks.append((L, list(path)))
                if L < max_len:
                    for w, eidx in adj[v]:
                        if not path or w != path[-1]:
                            stack.append((w, path + [eidx]))
        return walks

    if k == 4 and g == 10:
        q4_edges = get_q4()
        q4_n = 16
        q4_adj = get_adj(q4_edges, q4_n)
        
        # Enumerate closed walks of length 5..9
        closed_walks = enumerate_closed_walks(q4_adj, q4_n, 10)
        closed_walks = [(L, eidxs) for L, eidxs in closed_walks if L >= 5]
        
        walk_seqs = [eidxs for L, eidxs in closed_walks]
        
        def check_valid(m, voltages):
            for seq in walk_seqs:
                s = 0
                for e in seq:
                    s += voltages[e]
                if s % m == 0:
                    return False
            return True

        # Search for m=20, voltages in Z_20
        m = 20
        random.seed(42)
        best_voltages = None
        
        # Try random assignments
        for trial in range(10000):
            if time.time() > deadline:
                break
            voltages = [random.randint(1, m - 1) for _ in range(len(q4_edges))]
            if check_valid(m, voltages):
                best_voltages = voltages
                break
        
        if best_voltages is None:
            # If random fails, try local search
            voltages = [random.randint(1, m - 1) for _ in range(len(q4_edges))]
            def count_bad(m, voltages):
                count = 0
                for seq in walk_seqs:
                    s = 0
                    for e in seq:
                        s += voltages[e]
                    if s % m == 0:
                        count += 1
                return count
            
            best_count = count_bad(m, voltages)
            for iteration in range(10000):
                if time.time() > deadline:
                    break
                if best_count == 0:
                    best_voltages = voltages
                    break
                improved = False
                for eidx in range(len(q4_edges)):
                    if time.time() > deadline:
                        break
                    old_val = voltages[eidx]
                    for new_val in range(1, m):
                        if new_val == old_val:
                            continue
                        voltages[eidx] = new_val
                        cnt = count_bad(m, voltages)
                        if cnt < best_count:
                            best_count = cnt
                            improved = True
                        else:
                            voltages[eidx] = old_val
                            break
                    if improved:
                        break
                    voltages[eidx] = old_val
                if not improved:
                    voltages = [random.randint(1, m - 1) for _ in range(len(q4_edges))]
                    best_count = count_bad(m, voltages)
            
            if best_count == 0:
                best_voltages = voltages

        if best_voltages is not None:
            n = q4_n * m
            edges = []
            for idx, (u, v) in enumerate(q4_edges):
                w = best_voltages[idx]
                for g_val in range(m):
                    src = u * m + g_val
                    dst = v * m + ((g_val + w) % m)
                    if src < dst:
                        edges.append((src, dst))
            return n, edges

    # Fallback to LUW
    return luw_construct(k, g)
