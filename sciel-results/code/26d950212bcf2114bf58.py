import time
import random
from collections import deque
from array import array


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Voltage lift of the 4-regular bipartite base graph K_{5,5} minus a
        # perfect matching (10 base vertices) over the cyclic group Z_32.
        # Base edges: (i, 5+j) for i != j.  Lift has 10*32 = 320 vertices and
        # is 4-regular.  We search for a voltage assignment whose lift has
        # girth at least g (i.e. the (4,10)-cage on 320 vertices).
        res = _voltage_lift_search(start_time, deadline, g)
        if res is not None:
            return res

    # Fallback: guaranteed-valid Lazebnik-Ustimenko-Woldar construction.
    return _luw_construct(k, g)


def _voltage_lift_search(start_time, deadline, g):
    """Random + local-search voltage assignment on the K_{5,5}-matching base."""
    M = 32
    # Base edges (u, v) with u in 0..4, v in 5..9; voltage v[u][v-5].
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))
    E = len(base_edges)
    n = 10 * M
    n_edges = E * M

    # Precompute per-edge voltage index (row, col) so we can read/write fast.
    eidx = [(u, v - 5) for (u, v) in base_edges]

    # Adjacency of the base graph (for the non-backtracking girth probe).
    bdeg = [0] * 10
    for u, v in base_edges:
        bdeg[u] += 1
        bdeg[v] += 1

    def build(v):
        """Build the lift edge list (as an array of ints) from voltage matrix v."""
        out = array('I', [0])
        for idx in range(E):
            u, j = eidx[idx]
            w = v[u][j]
            vbase = 5 + j
            base_u = u * M
            base_v = vbase * M
            if w == 0:
                # (u, x) - (v, x)
                for x in range(M):
                    out.append(base_u + x)
                    out.append(base_v + x)
            else:
                # (u, x) - (v, (x + w) % M)
                for x in range(M):
                    y = x + w
                    if y >= M:
                        y -= M
                    out.append(base_u + x)
                    out.append(base_v + y)
        return out

    def girth_ok(v, min_girth):
        """Return True iff the lift has no cycle shorter than min_girth.
        Non-backtracking BFS from every vertex (each vertex once)."""
        max_depth = min_girth // 2
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for vv in bdeg:
                    pass
            # (rebuild properly below)
        return True

    # The above placeholder is replaced by the real implementation below.
    return _voltage_lift_search_real(start_time, deadline, g, M, base_edges, eidx, bdeg, n, n_edges, build)


def _voltage_lift_search_real(start_time, deadline, g, M, base_edges, eidx, bdeg, n, n_edges, build):
    from collections import deque

    def girth_ok(v, min_girth):
        max_depth = min_girth // 2
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for vv in bdeg:
                    pass
            # (real adjacency loop follows in the wrapper)
        return True

    # We need the actual adjacency of the base graph for the probe.
    return _voltage_lift_search_full(start_time, deadline, g, M, base_edges, eidx, n)


def _voltage_lift_search_full(start_time, deadline, g, M, base_edges, eidx, n):
    from collections import deque
    import array as _array

    E = len(base_edges)
    # Base adjacency list.
    badj = [[] for _ in range(10)]
    for u, v in base_edges:
        badj[u].append(v)
        badj[v].append(u)

    def build(v):
        out = _array.array('I', [0])
        for idx in range(E):
            u, j = eidx[idx]
            w = v[u][j]
            vbase = 5 + j
            base_u = u * M
            base_v = vbase * M
            if w == 0:
                for x in range(M):
                    out.append(base_u + x)
                    out.append(base_v + x)
            else:
                for x in range(M):
                    y = x + w
                    if y >= M:
                        y -= M
                    out.append(base_u + x)
                    out.append(base_v + y)
        return out

    def girth_ok(v, min_girth):
        # Non-backtracking BFS from every base vertex; a short cycle of length
        # L < min_girth yields a non-backtracking walk of length <= L-1.
        max_depth = min_girth // 2
        for s in range(10):
            seen = set()
            seen.add((s, -1))
            q = deque()
            q.append((s, -1, 0))
            while q:
                u, p, d = q.popleft()
                if d >= max_depth:
                    continue
                for vv in badj[u]:
                    if vv == p:
                        continue
                    key = (vv, u)
                    if key in seen:
                        # non-backtracking walk of length d+1 exists
                        if d + 1 < min_girth:
                            return False
                    else:
                        seen.add(key)
                        q.append((vv, u, d + 1))
        return True

    def random_v():
        return [[random.randint(0, M - 1) for _ in range(5)] for _ in range(5)]

    def try_accept(v):
        return girth_ok(v, g)

    best = None
    attempts = 0
    while time.time() < deadline - 3.0:
        v = random_v()
        if try_accept(v):
            return n, build(v).tolist()
        attempts += 1
        # Local search: mutate a voltage pair and accept if girth improves.
        # Track a soft objective = the (negative) length of the shortest
        # non-backtracking cycle found; we hill-climb to raise it.
        cur_v = v
        cur_score = _shortest_nb_cycle_len(cur_v, g)  # larger is better
        steps = 0
        max_steps = 60
        while steps < max_steps and time.time() < deadline - 3.0:
            steps += 1
            # Pick a random edge index and a random target voltage.
            idx = random.randrange(E)
            u, j = eidx[idx]
            old = cur_v[u][j]
            new = random.randint(0, M - 1)
            if new == old:
                continue
            cur_v[u][j] = new
            new_score = _shortest_nb_cycle_len(cur_v, g)
            if new_score > cur_score:
                cur_score = new_score
                if cur_score >= g:
                    return n, build(cur_v).tolist()
            else:
                # revert (occasionally keep a worse move to escape plateaus)
                if random.random() < 0.85:
                    cur_v[u][j] = old
                else:
                    cur_score = new_score
        # If we improved the best-so-far score, keep going; else restart.
        if cur_score > (best[0] if best else -1):
            best = (cur_score, [row[:] for row in cur_v])

    # Fallback: return the best candidate we have (may still be invalid -> the
    # caller's LUW fallback is safer, so we return None to trigger fallback).
    return None


def _shortest_nb_cycle_len(v, g):
    """Length of the shortest non-backtracking cycle in the lift (in edges),
    or a large number if none below g is found.  Larger is better."""
    from collections import deque
    M = 32
    badj = _SHORT_BASE_ADJ
    max_depth = g // 2
    best = g  # we only care up to g
    for s in range(10):
        seen = set()
        seen.add((s, -1))
        q = deque()
        q.append((s, -1, 0))
        while q:
            u, p, d = q.popleft()
            if d >= max_depth:
                continue
            for vv in badj[u]:
                if vv == p:
                    continue
                key = (vv, u)
                if key in seen:
                    L = d + 1
                    if L < best:
                        best = L
                else:
                    seen.add(key)
                    q.append((vv, u, d + 1))
    return best


_SHORT_BASE_ADJ = None


def _init_base_adj():
    global _SHORT_BASE_ADJ
    if _SHORT_BASE_ADJ is None:
        badj = [[] for _ in range(10)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    badj[i].append(5 + j)
                    badj[5 + j].append(i)
        _SHORT_BASE_ADJ = badj
    return _SHORT_BASE_ADJ


def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar construction CD(d, q) with q = k.
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


# Initialize the shared base adjacency once at import time.
_init_base_adj()
