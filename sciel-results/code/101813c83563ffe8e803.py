"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def luw_construct(k, g):
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

    if k == 4:
        def check_girth(n, a, b, g_target):
            if (2 * a) % n == 0 or (2 * b) % n == 0:
                return False
            if (a + b) % n == 0 or (a - b) % n == 0:
                return False
            if (3 * a) % n == 0 or (3 * b) % n == 0:
                return False
            if (2 * a + b) % n == 0 or (2 * a - b) % n == 0:
                return False
            if (a + 2 * b) % n == 0 or (a - 2 * b) % n == 0:
                return False
            steps = (a, n - a, b, n - b)
            INF = 10**9
            dist = [[INF] * 4 for _ in range(n)]
            queue = []
            for d in range(4):
                v = steps[d] % n
                dist[v][d] = 1
                queue.append((v, d))
            min_cycle = INF
            qi = 0
            while qi < len(queue):
                v, d = queue[qi]
                qi += 1
                length = dist[v][d]
                if length >= g_target:
                    break
                for nd in range(4):
                    if (d ^ 1) == nd:
                        continue
                    nv = (v + steps[nd]) % n
                    if nv == 0:
                        cycle_len = length + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g_target:
                                return False
                    elif dist[nv][nd] == INF:
                        dist[nv][nd] = length + 1
                        queue.append((nv, nd))
            return min_cycle >= g_target

        # Phase 1: Search n from 243 to 319
        for n in range(243, 320):
            if time.time() > deadline:
                break
            t0 = time.time()
            max_a = min(30, n // 2)
            for a in range(1, max_a + 1):
                if time.time() - t0 > 1.0:
                    break
                for b in range(a + 1, n):
                    if b == n - a:
                        continue
                    if check_girth(n, a, b, g):
                        edges = []
                        for i in range(n):
                            for s in (a, n - a, b, n - b):
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges

        # Phase 2: Exhaustive search for n=320
        if time.time() < deadline - 5:
            n = 320
            for a in range(1, n // 2 + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, n):
                    if b == n - a:
                        continue
                    if check_girth(n, a, b, g):
                        edges = []
                        for i in range(n):
                            for s in (a, n - a, b, n - b):
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges

        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
# EVOLVE-BLOCK-END
