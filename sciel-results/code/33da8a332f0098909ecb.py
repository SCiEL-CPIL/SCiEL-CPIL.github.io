import time
import random
from collections import deque


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        res = _voltage_lift_search(start_time, deadline, g)
        if res is not None:
            return res
        # Fallback: guaranteed-valid Lazebnik-Ustimenko-Woldar construction.
        return _luw_construct(k, g)

    # Generic fallback for other (k, g).
    return _luw_construct(k, g)


# ---------------------------------------------------------------------------
# Voltage-lift search over the K_{5,5} minus perfect matching base, Z_32 group.
# Lift is 4-regular on 320 vertices. We search a voltage assignment whose lift
# has girth >= g.
# ---------------------------------------------------------------------------

def _voltage_lift_search(start_time, deadline, g):
    M = 32
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    E = len(base_edges)
    n = 10 * M
    eidx = [(u, v - 5) for (u, v) in base_edges]

    # Base adjacency (10 nodes) - precompute as lists for speed
    badj = [[] for _ in range(10)]
    for u, v in base_edges:
        badj[u].append(v)
        badj[v].append(u)
    
    # Precompute: for each directed edge (u,v), which voltage matrix entry it uses
    # and whether it's "forward" (u<5, v>=5) or "backward"
    # For u<5, v>=5: voltage = v[u][v-5], forward
    # For u>=5, v<5: voltage = M - v[v][u-5], backward
    directed_edges = []  # (u, v, mat_row, mat_col, is_forward)
    for u in range(10):
        for v in badj[u]:
            if u < 5:
                directed_edges.append((u, v, u, v - 5, True))
            else:
                directed_edges.append((u, v, v, u - 5, False))
    
    # Build adjacency as list of directed edge indices per node
    adj_dir = [[] for _ in range(10)]
    for idx, (u, v, mr, mc, fwd) in enumerate(directed_edges):
        adj_dir[u].append(idx)

    def build(v):
        """Build the lift edge list (list of (a,b) tuples) from voltage matrix v."""
        edges = []
        for idx in range(E):
            u, j = eidx[idx]
            w = v[u][j]
            vbase = 5 + j
            base_u = u * M
            base_v = vbase * M
            if w == 0:
                for x in range(M):
                    edges.append((base_u + x, base_v + x))
            else:
                for x in range(M):
                    y = x + w
                    if y >= M:
                        y -= M
                    edges.append((base_u + x, base_v + y))
        return edges

    # Fast girth check: returns True if no cycle shorter than min_girth
    def girth_ok(v, min_girth):
        max_depth = min_girth // 2
        for s in range(10):
            # seen[d][node] = set of predecessors, or use a set of (node, pred)
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for de_idx in adj_dir[u]:
                    u2, v2, mr, mc, fwd = directed_edges[de_idx]
                    if v2 == p:
                        continue
                    # Compute voltage
                    if fwd:
                        w = v[mr][mc]
                    else:
                        w = M - v[mr][mc]
                    key = (v2, u)
                    if key in seen:
                        if d + 1 < min_girth:
                            return False
                    else:
                        seen.add(key)
                        q.append((v2, u, d + 1))
        return True

    def shortest_nb(v, min_girth):
        """Length of the shortest non-backtracking cycle in the lift (edges),
        or min_girth if none shorter found. Larger is better."""
        max_depth = min_girth // 2
        best = min_girth
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for de_idx in adj_dir[u]:
                    u2, v2, mr, mc, fwd = directed_edges[de_idx]
                    if v2 == p:
                        continue
                    key = (v2, u)
                    if key in seen:
                        L = d + 1
                        if L < best:
                            best = L
                            if best == 4:  # Can't do better than 4 for 4-regular
                                return 4
                    else:
                        seen.add(key)
                        q.append((v2, u, d + 1))
        return best

    def random_v():
        return [[random.randint(0, M - 1) for _ in range(5)] for _ in range(5)]

    def copy_v(v):
        return [row[:] for row in v]

    # 1) Try a small set of deterministic structured voltage assignments.
    candidates = []
    # All-zero.
    candidates.append([[0] * 5 for _ in range(5)])
    # Linear: v[i][j] = (a*i + b*j) mod M for various (a,b).
    for a in range(0, M):
        for b in range(0, M):
            if (a, b) in [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (3, 1), (5, 1), (7, 3), (3, 5), (11, 7), (13, 9), (17, 5), (19, 11), (23, 7), (29, 13)]:
                candidates.append([[((a * i + b * j) % M) for j in range(5)] for i in range(5)])
    # More structured: quadratic
    for a in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        for b in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
            for c in [0, 1, 5, 10, 15, 20, 25, 30]:
                candidates.append([[(a * i * i + b * i * j + c) % M for j in range(5)] for i in range(5)])
    # De-duplicate.
    seen_cand = set()
    unique_cands = []
    for c in candidates:
        key = tuple(tuple(row) for row in c)
        if key not in seen_cand:
            seen_cand.add(key)
            unique_cands.append(c)
    for c in unique_cands:
        if time.time() > deadline - 3.0:
            break
        if girth_ok(c, g):
            return n, build(c)

    # 2) Randomized local search
    best_score = -1
    best_v = None
    attempts = 0
    while time.time() < deadline - 3.0:
        attempts += 1
        v = random_v()
        cur_score = shortest_nb(v, g)
        if cur_score >= g:
            return n, build(v)
        if cur_score > best_score:
            best_score = cur_score
            best_v = copy_v(v)
        # Local search with simulated annealing-like acceptance
        cur_v = copy_v(v)
        cur_s = cur_score
        steps = 0
        max_steps = 60
        temp = 2.0  # "temperature" for accepting worse moves
        while steps < max_steps and time.time() < deadline - 3.0:
            steps += 1
            idx = random.randrange(E)
            u, j = eidx[idx]
            old = cur_v[u][j]
            new = random.randint(0, M - 1)
            if new == old:
                continue
            cur_v[u][j] = new
            new_s = shortest_nb(cur_v, g)
            if new_s >= g:
                return n, build(cur_v)
            if new_s > cur_s:
                cur_s = new_s
                temp = max(0.5, temp * 0.95)
                if cur_s > best_score:
                    best_score = cur_s
                    best_v = copy_v(cur_v)
            else:
                # Accept worse move with probability based on "temperature"
                delta_s = cur_s - new_s
                if random.random() < 0.1 / (1 + delta_s * temp):
                    cur_s = new_s
                    temp = max(0.3, temp * 0.9)
                else:
                    cur_v[u][j] = old
        if best_v is not None and best_score >= g:
            return n, build(best_v)

    # 3) Return best valid candidate if we ever found one; else None -> LUW.
    if best_v is not None and best_score >= g:
        return n, build(best_v)
    return None


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar construction CD(d, q) with q = k.
# ---------------------------------------------------------------------------

def _luw_construct(k, g):
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
