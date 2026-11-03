"""Cage solver: random k-regular init + greedy stub matching + local 2-switch
repair + exact non-backtracking girth (vectorized). Entry: construct(k, g)."""
import time
import random
import numpy as np


def construct(k, g):
    """Construct a simple k-regular graph with girth >= g on as few vertices as
    possible. Returns (n, edges)."""
    deadline = time.time() + 140.0  # safety margin before the 150 s kill

    # Moore bound / feasibility lower bound
    def _moore(kk, gg):
        R = (gg - 1) // 2
        return 1 + kk * sum((kk - 1) ** i for i in range(R))

    def _min_n(kk, gg):
        n = _moore(kk, gg)
        while (n * kk) % 2 != 0 or n < kk + 1:
            n += 1
        return n

    lo = _min_n(k, g)

    # Candidate sizes: start near the lower bound, grow as needed.
    n_list = []
    n = lo
    while n < 12000 and len(n_list) < 12:
        if n * k % 2 == 0 and n >= k + 1:
            n_list.append(n)
        n += 2
    for extra in (1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000):
        if extra > n_list[-1] and extra * k % 2 == 0:
            n_list.append(extra)

    # The LUW graph is a guaranteed-valid (if large) fallback.
    luw = _luw_construct(k, g)
    best = luw  # (n, edges)

    for n_attempt in n_list:
        if time.time() > deadline:
            break
        # Skip if we already have something at least this small
        if best is not None and best[0] <= n_attempt:
            continue
        result = _try_construct(k, g, n_attempt, deadline)
        if result is not None:
            n, edges = result
            if best is None or n < best[0]:
                best = (n, edges)
            if n <= lo + 5:  # essentially optimal, stop
                break

    return best


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q), q = k (guaranteed valid)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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


def _try_construct(k, g, n, deadline):
    """Try to build a k-regular graph on n vertices with girth >= g."""
    max_attempts = 20
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        edges = _random_regular_graph(k, n)
        if edges is None:
            continue
        adj = _build_adj(edges, n)
        girth = _girth_np(adj, n, g)
        if girth >= g:
            return (n, edges)
        # Repair
        edges = _fix_short_cycles(n, edges, k, g, deadline)
        if edges is None:
            continue
        adj = _build_adj(edges, n)
        girth = _girth_np(adj, n, g)
        if girth >= g:
            return (n, edges)
    return None


def _random_regular_graph(k, n):
    """Random simple k-regular graph via greedy stub matching (config model)."""
    if n * k % 2 != 0 or n < k + 1:
        return None
    for _ in range(300):
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not ok:
            continue
        # verify exact degree
        if any(len(a) != k for a in adj):
            continue
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges
    return None


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    return [sorted(a) for a in adj]


def _girth_np(adj_lists, n, g):
    """Exact girth (or g if none shorter) via non-backtracking walks, vectorized."""
    deg = np.array([len(a) for a in adj_lists], dtype=np.int32)
    m = int(deg.sum())
    k = int(deg.max())
    pos = np.zeros(m, dtype=np.int32)
    for u in range(n):
        du = deg[u]
        pos[u * k: u * k + du] = np.arange(du, dtype=np.int32)
    nbr = np.zeros((n, k), dtype=np.int32)
    for u in range(n):
        nbr[u, :deg[u]] = np.array(adj_lists[u], dtype=np.int32)

    u_idx = np.repeat(np.arange(n, dtype=np.int32), deg)
    p_idx = pos
    v_idx = nbr[u_idx, p_idx]
    init_state = v_idx * k + p_idx
    init_origin = u_idx

    visited_origin = np.full(n * k, -1, dtype=np.int32)
    visited_dist = np.zeros(n * k, dtype=np.int32)
    visited_origin[init_state] = init_origin
    visited_dist[init_state] = 1

    cur_state = init_state
    cur_origin = init_origin
    dist = 1

    while len(cur_state) > 0:
        v_cur = cur_state // k
        p_cur = cur_state % k
        r_all = np.tile(np.arange(k, dtype=np.int32), len(v_cur))
        w_all = nbr[v_cur, r_all]
        mask = (r_all != p_cur)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        hit = (w_all == o_all)
        if hit.any():
            return dist + 1

        new_state = w_all * k + r_all
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            diff = (is_new == False)
            cand = (dist + 1 + visited_dist[new_state[diff]])
            if cand.min() < g:
                return int(cand.min())

        new_state = new_state[is_new]
        new_origin = o_all[is_new]
        if len(new_state) == 0:
            break

        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1
        if dist >= g:
            break

    return g


def _short_cycle_edges(n, adj, g):
    """Return a set of edges that lie on some cycle of length < g, using
    non-backtracking walks (meetings of distinct origins at the same state)."""
    deg = np.array([len(a) for a in adj], dtype=np.int32)
    k = int(deg.max())
    nbr = np.zeros((n, k), dtype=np.int32)
    for u in range(n):
        nbr[u, :deg[u]] = np.array(adj[u], dtype=np.int32)

    # state (v, p): v = current vertex, p = position of parent in adj[v]
    # origin = u
    init_state = []
    init_origin = []
    for u in range(n):
        for p, v in enumerate(adj[u]):
            init_state.append(v * k + p)
            init_origin.append(u)
    init_state = np.array(init_state, dtype=np.int32)
    init_origin = np.array(init_origin, dtype=np.int32)

    visited_origin = np.full(n * k, -1, dtype=np.int32)
    visited_dist = np.zeros(n * k, dtype=np.int32)
    visited_origin[init_state] = init_origin
    visited_dist[init_state] = 1

    bad = set()
    cur_state = init_state
    cur_origin = init_origin
    dist = 1

    while len(cur_state) > 0:
        v_cur = cur_state // k
        p_cur = cur_state % k
        r_all = np.tile(np.arange(k, dtype=np.int32), len(v_cur))
        w_all = nbr[v_cur, r_all]
        mask = (r_all != p_cur)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        # closing a cycle: w == origin
        hit = (w_all == o_all)
        if hit.any():
            for idx in np.where(hit)[0]:
                u = int(o_all[idx])
                v = int(w_all[idx])
                bad.add((min(u, v), max(u, v)))

        new_state = w_all * k + r_all
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            diff = (is_new == False)
            for idx in np.where(diff)[0]:
                st = int(new_state[idx])
                # meeting of two walks -> cycle; mark the edge entering this state
                v = st // k
                r = st % k
                w = int(w_all[idx])
                # the edge (v, w) is part of a short cycle
                bad.add((min(v, w), max(v, w)))

        new_state = new_state[is_new]
        new_origin = o_all[is_new]
        if len(new_state) == 0:
            break
        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1
        if dist >= g:
            break

    return bad


def _fix_short_cycles(n, edges, k, g, deadline):
    """2-switch repair: target edges inside short cycles, then random."""
    edge_set = set()
    for (u, v) in edges:
        edge_set.add((min(u, v), max(u, v)))
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)

    def _would_create_short_cycle(u, v):
        # BFS from u to v up to depth g-2; if found, adding (u,v) makes a cycle < g
        if v in adj[u]:
            return True
        depth = g - 2
        seen = {u}
        frontier = [u]
        for _ in range(depth):
            nxt = []
            for x in frontier:
                for y in adj[x]:
                    if y == v:
                        return True
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
            if not frontier:
                break
        return False

    def _do_swap(e1, e2, variant):
        # e1=(u,v), e2=(c,d) -> (u,c),(v,d) [variant 0] or (u,d),(v,c) [variant 1]
        u, v = e1
        c, d = e2
        if variant == 0:
            new1, new2 = (u, c), (v, d)
        else:
            new1, new2 = (u, d), (v, c)
        # validity: no loops, no multi-edges, distinct from existing
        for a, b in (new1, new2):
            if a == b:
                return False
            if b in adj[a]:
                return False
        if new1 == new2:
            return False
        # would these create short cycles?
        if _would_create_short_cycle(*new1) or _would_create_short_cycle(*new2):
            return False
        # apply
        adj[u].discard(v); adj[v].discard(u)
        adj[c].discard(d); adj[d].discard(c)
        for a, b in (new1, new2):
            adj[a].add(b); adj[b].add(a)
        edges.remove(e1); edges.remove(e2)
        edges.append(tuple(new1)); edges.append(tuple(new2))
        return True

    # collect candidate edges for swapping (prefer those in short cycles)
    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > deadline:
            return None
        bad = _short_cycle_edges(n, [sorted(a) for a in adj], g)
        if not bad:
            return edges
        # candidate edge list
        cand = [e for e in edges if (min(e[0], e[1]), max(e[0], e[1])) in bad]
        if not cand:
            cand = list(edges)
        improved = False
        for _ in range(60):
            if time.time() > deadline:
                return None
            e1 = random.choice(cand)
            e2 = random.choice(edges)
            if e1 == e2:
                continue
            if e1[0] in e2 or e1[1] in e2:
                continue
            if _do_swap(e1, e2, 0):
                improved = True
                break
            if _do_swap(e1, e2, 1):
                improved = True
                break
        # occasional global random 2-switch
        if not improved and random.random() < 0.5:
            for _ in range(20):
                if time.time() > deadline:
                    return None
                e1 = random.choice(edges)
                e2 = random.choice(edges)
                if e1 == e2 or e1[0] in e2 or e1[1] in e2:
                    continue
                if _do_swap(e1, e2, random.randint(0, 1)):
                    break

        # check girth
        girth = _girth_np([sorted(a) for a in adj], n, g)
        if girth >= g:
            return edges
    return None
