import time
import sys
import random

def construct(k, g):
    deadline = time.time() + 140.0

    def verify_girth(n, edges, g):
        """Verify girth >= g using non-backtracking walks from every vertex.
        Also verifies simplicity and regularity."""
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n:
                return False
            if u == v:
                return False
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Non-backtracking walk BFS
        for start in range(n):
            dist = {}
            queue = [(start, -1, 0)]
            head = 0
            while head < len(queue):
                u, prev, d = queue[head]
                head += 1
                if d >= g // 2:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    key = (v, u)
                    if key not in dist:
                        dist[key] = d + 1
                        queue.append((v, u, d + 1))
                    else:
                        cycle_len = dist[key] + d + 1
                        if cycle_len < g:
                            return False
        return True

    def luw_construct(d, q):
        """Lazebnik-Ustimenko-Woldar construction."""
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
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
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

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Main construction for k=4, g=10
    if k == 4 and g == 10:
        # Voltage graph construction on K5,5 minus perfect matching
        base_edges = []
        edge_index = {}
        for i in range(5):
            for j in range(5):
                if i != j:
                    edge_index[(i, 5 + j)] = len(base_edges)
                    base_edges.append((i, 5 + j))
        n_base_edges = len(base_edges)  # 20 edges

        # Enumerate 4-cycles
        four_cycles = []
        for i1 in range(5):
            for i2 in range(i1 + 1, 5):
                for j1 in range(5):
                    if j1 == i1 or j1 == i2:
                        continue
                    for j2 in range(j1 + 1, 5):
                        if j2 == i1 or j2 == i2:
                            continue
                        e1 = edge_index.get((i1, 5 + j1))
                        e2 = edge_index.get((i2, 5 + j1))
                        e3 = edge_index.get((i2, 5 + j2))
                        e4 = edge_index.get((i1, 5 + j2))
                        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                            four_cycles.append((e1, e2, e3, e4))

        # Enumerate 6-cycles
        six_cycles = []
        for i1 in range(5):
            for i2 in range(5):
                if i2 == i1:
                    continue
                for i3 in range(5):
                    if i3 == i1 or i3 == i2:
                        continue
                    for j1 in range(5):
                        if j1 == i1:
                            continue
                        for j2 in range(5):
                            if j2 == i2 or j2 == j1:
                                continue
                            for j3 in range(5):
                                if j3 == i3 or j3 == j1 or j3 == j2:
                                    continue
                                e1 = edge_index.get((i1, 5 + j1))
                                e2 = edge_index.get((i2, 5 + j1))
                                e3 = edge_index.get((i2, 5 + j2))
                                e4 = edge_index.get((i3, 5 + j2))
                                e5 = edge_index.get((i3, 5 + j3))
                                e6 = edge_index.get((i1, 5 + j3))
                                if None not in (e1, e2, e3, e4, e5, e6):
                                    six_cycles.append((e1, e2, e3, e4, e5, e6))

        # Enumerate 8-cycles
        eight_cycles = []
        for i1 in range(5):
            for i2 in range(5):
                if i2 == i1:
                    continue
                for i3 in range(5):
                    if i3 == i1 or i3 == i2:
                        continue
                for i4 in range(5):
                    if i4 == i1 or i4 == i2 or i4 == i3:
                        continue
                    for j1 in range(5):
                        if j1 == i1:
                            continue
                        for j2 in range(5):
                            if j2 == i2 or j2 == j1:
                                continue
                            for j3 in range(5):
                                if j3 == i3 or j3 == j1 or j3 == j2:
                                    continue
                                for j4 in range(5):
                                    if j4 == i4 or j4 == j1 or j4 == j2 or j4 == j3:
                                        continue
                                    e1 = edge_index.get((i1, 5 + j1))
                                    e2 = edge_index.get((i2, 5 + j1))
                                    e3 = edge_index.get((i2, 5 + j2))
                                    e4 = edge_index.get((i3, 5 + j2))
                                    e5 = edge_index.get((i3, 5 + j3))
                                    e6 = edge_index.get((i4, 5 + j3))
                                    e7 = edge_index.get((i4, 5 + j4))
                                    e8 = edge_index.get((i1, 5 + j4))
                                    if None not in (e1, e2, e3, e4, e5, e6, e7, e8):
                                        eight_cycles.append((e1, e2, e3, e4, e5, e6, e7, e8))

        def build_lift(voltages):
            """Build the 320-vertex lift from voltage assignment."""
            n_lift = 320
            lift_edges = []
            for u, v in base_edges:
                idx = edge_index[(u, v)]
                w = voltages[idx]
                for x in range(32):
                    u_lift = u * 32 + x
                    v_lift = v * 32 + (x ^ w)
                    lift_edges.append((u_lift, v_lift))
            return n_lift, lift_edges

        def check_voltages(voltages):
            """Check if voltage assignment kills all short cycles."""
            for cycle in four_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            for cycle in six_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            for cycle in eight_cycles:
                s = 0
                for e in cycle:
                    s ^= voltages[e]
                if s == 0:
                    return False
            return True

        # Randomized search for voltage assignment
        random.seed(42)
        found = None
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            voltages = [random.randint(0, 31) for _ in range(n_base_edges)]
            if check_voltages(voltages):
                found = voltages
                break

        if found is not None:
            n_lift, lift_edges = build_lift(found)
            if verify_girth(n_lift, lift_edges, 10):
                return n_lift, lift_edges

        # Try structured assignments
        for a_seed in range(32):
            if time.time() > deadline - 20:
                break
            for b_seed in range(32):
                if time.time() > deadline - 15:
                    break
                a_vecs = [(a_seed * (i + 1)) & 31 for i in range(5)]
                b_vecs = [(b_seed * (j + 1)) & 31 for j in range(5)]
                for twist_type in range(8):
                    voltages = [0] * n_base_edges
                    for idx, (u, v) in enumerate(base_edges):
                        i = u
                        j = v - 5
                        if twist_type == 0:
                            c = 0
                        elif twist_type == 1:
                            c = i ^ j
                        elif twist_type == 2:
                            c = (i * j) & 31
                        elif twist_type == 3:
                            c = (i + j) & 31
                        elif twist_type == 4:
                            c = (i * 3 + j * 5) & 31
                        elif twist_type == 5:
                            c = (i * 7 + j * 11) & 31
                        elif twist_type == 6:
                            c = (i * 13 + j * 17) & 31
                        else:
                            c = (i * 19 + j * 23) & 31
                        voltages[idx] = a_vecs[i] ^ b_vecs[j] ^ c
                    if check_voltages(voltages):
                        n_lift, lift_edges = build_lift(voltages)
                        if verify_girth(n_lift, lift_edges, 10):
                            return n_lift, lift_edges

        # Fallback: LUW construction
        for d_try in [5, 6, 7]:
            if time.time() > deadline - 10:
                break
            n, edges = luw_construct(d_try, 4)
            if verify_girth(n, edges, 10):
                return n, edges

    # General case
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n, edges = luw_construct(d, k)
    if verify_girth(n, edges, g):
        return n, edges

    return n, edges
