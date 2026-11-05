"""4-cycle flip local search with bit-parallel C6 bookkeeping for R_4(C6)."""
import time
import random

_DEADLINE = time.time() + 110.0


def _edge_index(n):
    m = n * (n - 1) // 2
    idx = [[-1] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = idx[j][i] = k
            k += 1
    return idx, m


def _c6_sets(n, m, idx):
    c6s = []
    for i in range(n - 1):
        for j in range(i + 2, n):
            e0 = idx[i][j]
            for a in range(n):
                if a == i or a == j:
                    continue
                e1 = idx[i][a]
                for b in range(n):
                    if b == i or b == j or b == a:
                        continue
                    e2 = idx[a][b]
                    for c in range(n):
                        if c == i or c == j or c == a or c == b:
                            continue
                        e3 = idx[b][c]
                        for d in range(n):
                            if d == i or d == j or d == a or d == b or d == c:
                                continue
                            e4 = idx[c][d]
                            if d == j:
                                continue
                            e5 = idx[d][j]
                            s = frozenset((e0, e1, e2, e3, e4, e5))
                            if len(s) == 6:
                                c6s.append(s)
    return c6s


_EDGE_CACHE = {}


def _get_edge_struct(n):
    if n in _EDGE_CACHE:
        return _EDGE_CACHE[n]
    idx, m = _edge_index(n)
    c6s = _c6_sets(n, m, idx)
    edge_c6s = [[] for _ in range(m)]
    for ci, s in enumerate(c6s):
        for e in s:
            edge_c6s[e].append(ci)
    _EDGE_CACHE[n] = (idx, m, c6s, edge_c6s)
    return _EDGE_CACHE[n]


def _init_coloring(n, idx):
    if n == 17:
        S = [1, 4, 8, 13]
    else:
        S = [1, 2, 3, 4]
    m = n * (n - 1) // 2
    colors = [0] * m
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            for c, s in enumerate(S):
                if d == s:
                    colors[idx[i][j]] = c
                    break
    return colors


def _build_state(colors, n, idx, c6s, edge_c6s):
    m = n * (n - 1) // 2
    present = [0, 0, 0, 0]
    for e, c in enumerate(colors):
        present[c] |= (1 << e)
    cnt = [[0] * m for _ in range(4)]
    c6_count = [0, 0, 0, 0]
    for c in range(4):
        p = present[c]
        for ci, s in enumerate(c6s):
            ok = True
            for e in s:
                if not (p >> e) & 1:
                    ok = False
                    break
            if ok:
                c6_count[c] += 1
                for e in s:
                    cnt[c][e] += 1
    return present, cnt, c6_count


def _validate(colors, n, idx, c6s):
    present = [0, 0, 0, 0]
    for e, c in enumerate(colors):
        present[c] |= (1 << e)
    for c in range(4):
        p = present[c]
        for s in c6s:
            ok = True
            for e in s:
                if not (p >> e) & 1:
                    ok = False
                    break
            if ok:
                return False
    return True


def construct():
    random.seed(20240517)
    n = 17
    idx, m, c6s, edge_c6s = _get_edge_struct(n)

    colors0 = _init_coloring(n, idx)
    if _validate(colors0, n, idx, c6s):
        return {'n': n, 'colors': colors0}

    present, cnt, c6_count = _build_state(colors0, n, idx, c6s, edge_c6s)
    total0 = sum(c6_count)
    verts = list(range(n))
    best = None
    best_total = total0

    def run_search(start_colors, time_budget):
        nonlocal best, best_total
        colors = start_colors[:]
        present_l, cnt_l, c6_count_l = _build_state(colors, n, idx, c6s, edge_c6s)
        cur_total = sum(c6_count_l)

        temp = 0.2
        no_improve = 0
        max_no_improve = 2000
        step = 0

        while time.time() < time_budget:
            step += 1
            if step % 1000 == 0:
                no_improve = min(no_improve, 200)
                temp = max(0.05, temp * 0.95)

            p, q, r, s = random.sample(verts, 4)
            e_pq = idx[p][q]
            e_rs = idx[r][s]
            e_qr = idx[q][r]
            e_sp = idx[s][p]
            c_pq = colors[e_pq]
            c_rs = colors[e_rs]
            c_qr = colors[e_qr]
            c_sp = colors[e_sp]

            if c_pq == c_rs and c_qr == c_sp:
                continue

            edges = [e_pq, e_rs, e_qr, e_sp]
            old_c = [c_pq, c_rs, c_qr, c_sp]
            new_c = [c_rs, c_pq, c_sp, c_qr]

            # Compute delta using original colors
            # Affected C6s: union of edge_c6s[e] for e in edges
            affected_c6s = set()
            for e in edges:
                affected_c6s.update(edge_c6s[e])

            delta_total = 0
            for ci in affected_c6s:
                s6 = c6s[ci]
                # was_in_old for each color
                was = [False, False, False, False]
                for cc in range(4):
                    all_old = True
                    for e2 in s6:
                        if e2 in (e_pq, e_rs, e_qr, e_sp):
                            # This edge is being changed; use old color
                            oi = edges.index(e2)
                            if old_c[oi] != cc:
                                all_old = False
                                break
                        else:
                            if colors[e2] != cc:
                                all_old = False
                                break
                    was[cc] = all_old
                # will_be_in_new for each color
                will = [False, False, False, False]
                for cc in range(4):
                    all_new = True
                    for e2 in s6:
                        if e2 in (e_pq, e_rs, e_qr, e_sp):
                            ni = edges.index(e2)
                            if new_c[ni] != cc:
                                all_new = False
                                break
                        else:
                            if colors[e2] != cc:
                                all_new = False
                                break
                    will[cc] = all_new
                for cc in range(4):
                    d = (1 if will[cc] else 0) - (1 if was[cc] else 0)
                    if d != 0:
                        c6_count_l[cc] += d
                        delta_total += d
                        for e2 in s6:
                            cnt_l[cc][e2] += d

            if delta_total < 0 or (delta_total == 0 and random.random() < 0.3) or (
                delta_total <= 2 and random.random() < temp
            ):
                # Apply
                for e, oc, nc in zip(edges, old_c, new_c):
                    present_l[oc] &= ~(1 << e)
                    present_l[nc] |= (1 << e)
                    colors[e] = nc
                cur_total += delta_total
                no_improve = 0
                if cur_total < best_total:
                    best_total = cur_total
                    if cur_total == 0:
                        best = colors[:]
                        return
            else:
                no_improve += 1

            if no_improve > max_no_improve:
                break

        if cur_total < best_total:
            best_total = cur_total
            if cur_total == 0:
                best = colors[:]
                return
        return colors

    start = colors0[:]
    while time.time() < _DEADLINE - 5.0:
        result = run_search(start, min(time.time() + 4.0, _DEADLINE - 5.0))
        if best is not None:
            return {'n': n, 'colors': best}
        if result is not None:
            start = result
        if time.time() > _DEADLINE - 8.0:
            break
        for _ in range(10):
            e1 = random.randint(0, m - 1)
            e2 = random.randint(0, m - 1)
            if e1 != e2:
                start[e1], start[e2] = start[e2], start[e1]

    if best is not None:
        return {'n': n, 'colors': best}
    return {'n': n, 'colors': colors0}
