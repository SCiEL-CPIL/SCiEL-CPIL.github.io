import time
import random
import math
from collections import deque
import numpy as np


def _moore_bound(k, g):
    if g % 2 == 0:
        h = g // 2
        return 1 + k * sum(k ** i for i in range(h - 1))
    else:
        h = (g - 1) // 2
        return k * sum(k ** i for i in range(h))


def construct(k, g):
    # ---- LUW construction (fallback / general k,g) ----
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = sub = lambda x, y: x ^ y
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

    if not (k == 4 and g == 10):
        return luw_construct(k, g)

    deadline = time.time() + 115.0
    n_target = 320
    mb = _moore_bound(4, 10)
    if n_target < mb:
        return luw_construct(k, g)

    # ---------- vectorized girth (BFS) ----------
    def get_girth(adj, limit=None):
        n = len(adj)
        if limit is None:
            limit = float('inf')
        best = float('inf')
        A = np.asarray(adj, dtype=np.int32)
        for s in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            dist[s] = 0
            frontier = np.array([s], dtype=np.int32)
            d = 0
            while frontier.size:
                d += 1
                if d >= best:
                    break
                nxt = A[frontier].ravel()
                nd = dist[nxt]
                cyc = nd[nd >= 0] + d + 1
                if cyc.size:
                    cm = int(cyc.min())
                    if cm < best:
                        best = cm
                        if best <= limit:
                            return best
                unv = nd < 0
                if unv.any():
                    newf = nxt[unv]
                    dist[newf] = d
                    frontier = np.unique(newf)
        return best

    # ---------- exact non-backtracking girth + short cycle ----------
    def find_short_cycle(adj, limit):
        n = len(adj)
        best = float('inf')
        best_cycle = None
        for s in range(n):
            prev, cur = -1, s
            depth = 0
            seen = {(prev, cur): 0}
            while True:
                depth += 1
                if depth >= best:
                    break
                found = False
                for v in adj[cur]:
                    if v == prev:
                        continue
                    if v == s and depth + 1 >= 3:
                        if depth + 1 < best:
                            best = depth + 1
                            best_cycle = [s] + [cur]
                            if best <= limit:
                                return best, best_cycle
                        found = True
                    else:
                        key = (cur, v)
                        if key in seen:
                            cl = seen[key] + depth + 1
                            if cl < best:
                                best = cl
                                best_cycle = [s, cur, v]
                                if best <= limit:
                                    return best, best_cycle
                        else:
                            seen[key] = depth
                            prev, cur = cur, v
                            found = True
                if not found:
                    break
        return best, best_cycle

    # ---------- random 4-regular via configuration model ----------
    def random_4_regular(n):
        items = list(range(n))
        random.shuffle(items)
        deg = [0] * n
        adj = [[] for _ in range(n)]
        ok = True
        for i in range(0, n, 2):
            u, v = items[i], items[i + 1]
            if u == v:
                ok = False
                break
            adj[u].append(v)
            adj[v].append(u)
        if not ok:
            return None
        for u in range(n):
            if len(set(adj[u])) < 2:
                return None
        return adj

    def is_valid_4reg(adj):
        n = len(adj)
        for u in range(n):
            if len(adj[u]) != 4:
                return False
            if len(set(adj[u])) < 4:
                return False
            for v in adj[u]:
                if v == u:
                    return False
                if u not in adj[v]:
                    return False
        return True

    def edges_of(adj):
        out = []
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    out.append((u, v))
        return out

    # ---------- 2-switch ----------
    def do_swap(adj, a, b, c, d):
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)

    def undo_swap(adj, a, b, c, d):
        adj[a].remove(c)
        adj[c].remove(a)
        adj[b].remove(d)
        adj[d].remove(b)
        adj[a].append(b)
        adj[b].append(a)
        adj[c].append(d)
        adj[d].append(c)

    # ---------- initial graph ----------
    adj = None
    for _ in range(400):
        if time.time() > deadline - 30:
            break
        cand = random_4_regular(n_target)
        if cand and is_valid_4reg(cand):
            adj = cand
            break
    if adj is None:
        return luw_construct(k, g)

    best_adj = None
    best_girth = 0
    while time.time() < deadline - 15:
        girth, cycle = find_short_cycle(adj, g)
        if girth >= g:
            best_adj = adj
            best_girth = girth
            break
        if best_adj is None or girth > best_girth:
            best_adj = [list(x) for x in adj]
            best_girth = girth

        # target a short cycle with a 2-switch
        if cycle and len(cycle) >= 3:
            nodes = cycle
            # pick two non-adjacent-ish edges on the cycle to swap
            L = len(nodes)
            i = random.randrange(L)
            j = (i + 1 + random.randrange(max(1, L - 2))) % L
            a = nodes[i]
            b = nodes[(i + 1) % L]
            c = nodes[j]
            d = nodes[(j + 1) % L]
            if len({a, b, c, d}) < 4:
                a = random.randrange(n_target)
                b = random.choice(adj[a])
                c = random.randrange(n_target)
                d = random.choice(adj[c])
                if len({a, b, c, d}) < 4:
                    continue
        else:
            a = random.randrange(n_target)
            b = random.choice(adj[a])
            c = random.randrange(n_target)
            d = random.choice(adj[c])
            if len({a, b, c, d}) < 4:
                continue

        do_swap(adj, a, b, c, d)
        new_girth, _ = find_short_cycle(adj, g)
        # annealing acceptance
        if new_girth >= girth:
            pass
        else:
            T = 1.5 * math.exp(-3.0 * (time.time() - (deadline - 115.0)) / 115.0)
            if random.random() > math.exp(-(girth - new_girth) / max(T, 1e-6)):
                undo_swap(adj, a, b, c, d)

    if best_adj is not None:
        return n_target, edges_of(best_adj)
    # fallback: return current if it happens to be valid, else LUW
    if is_valid_4reg(adj):
        gg, _ = find_short_cycle(adj, g)
        if gg >= g:
            return n_target, edges_of(adj)
    return luw_construct(k, g)
