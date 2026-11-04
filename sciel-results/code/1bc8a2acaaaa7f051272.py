import time
import random
from collections import deque


def construct(k, g):
    deadline = time.time() + 114.0

    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
            ]
            add = sub = lambda x, y: x ^ y
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
                yield 1 - side, tuple(y)

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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    n = 320

    def random_regular():
        for _ in range(120):
            m1 = {}
            m2 = {}
            for _ in range(3):
                perm = list(range(n))
                random.shuffle(perm)
                cand1 = {}
                cand2 = {}
                for i in range(0, n, 2):
                    u, v = perm[i], perm[i + 1]
                    cand1[u] = cand1[v] = v if u == perm[i] else u
                    cand2[u] = cand2[v] = v if u == perm[i] else u
                ok = True
                adj = [set() for _ in range(n)]
                for u in range(n):
                    for m in (cand1, cand2):
                        v = m[u]
                        if v in adj[u]:
                            ok = False
                            break
                        adj[u].add(v)
                        adj[v].add(u)
                    if not ok:
                        break
                if ok and all(len(adj[u]) == 4 for u in range(n)):
                    return [list(adj[u]) for u in range(n)]
        return None

    def girth(adj, upper=None):
        n = len(adj)
        best = upper if upper is not None else float("inf")
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] * 2 >= best:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif dist[v] < dist[u]:
                        cyc = dist[u] + dist[v] + 1
                        if cyc < best:
                            best = cyc
                            if best <= g:
                                return best, True
            if best <= g:
                return best, True
        return (best if best != float("inf") else 0), (best < g)

    def cycle_adj(adj, girth_limit):
        n = len(adj)
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] * 2 >= girth_limit:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif dist[v] < dist[u]:
                        cyc = dist[u] + dist[v] + 1
                        if cyc < girth_limit:
                            cyc_nodes = []
                            cur = u
                            while cur != v:
                                cyc_nodes.append(cur)
                                cur = parent[cur]
                            cyc_nodes.append(v)
                            cyc_set = set(cyc_nodes)
                            for i in range(len(cyc_nodes)):
                                a = cyc_nodes[i]
                                b = cyc_nodes[(i + 1) % len(cyc_nodes)]
                                if a in cyc_set and b in cyc_set:
                                    return cyc_nodes
        return None

    def valid(adj):
        for u in range(n):
            if len(adj[u]) != 4:
                return False
            if len(set(adj[u])) != 4:
                return False
            for v in adj[u]:
                if u == v:
                    return False
                if u not in adj[v]:
                    return False
        gr, bad = girth(adj, g)
        return not bad

    def swap_adj(adj, a, b, c, d, adj_sets):
        if c in adj_sets[a] or d in adj_sets[b] or c == d or a == b:
            return False
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        adj_sets[a].discard(b)
        adj_sets[b].discard(a)
        adj_sets[c].discard(d)
        adj_sets[d].discard(c)

        adj[a].append(c)
        adj[c].append(a)
        adj[b].append(d)
        adj[d].append(b)
        adj_sets[a].add(c)
        adj_sets[c].add(a)
        adj_sets[b].add(d)
        adj_sets[d].add(b)
        return True

    def revert_swap(adj, a, b, c, d, adj_sets):
        adj[a].remove(c)
        adj[c].remove(a)
        adj[b].remove(d)
        adj[d].remove(b)
        adj_sets[a].discard(c)
        adj_sets[c].discard(a)
        adj_sets[b].discard(d)
        adj_sets[d].discard(b)

        adj[a].append(b)
        adj[b].append(a)
        adj[c].append(d)
        adj[d].append(c)
        adj_sets[a].add(b)
        adj_sets[b].add(a)
        adj_sets[c].add(d)
        adj_sets[d].add(c)

    adj = random_regular()
    if adj is None:
        return luw_construct(k, g)

    adj_sets = [set(adj[u]) for u in range(n)]
    cur_gr, _ = girth(adj, g)
    best_adj = [list(adj[u]) for u in range(n)]
    best_gr = cur_gr

    while time.time() < deadline:
        if cur_gr >= g:
            break

        cyc = cycle_adj(adj, min(cur_gr, g))
        if not cyc:
            cur_gr, _ = girth(adj, g)
            continue

        improved = False
        tried = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            for x in adj[u]:
                if x == v:
                    continue
                for y in adj[v]:
                    if y == u or y == x:
                        continue
                    if swap_adj(adj, u, v, x, y, adj_sets):
                        ng, _ = girth(adj, cur_gr)
                        if ng >= cur_gr:
                            cur_gr = ng
                            best_gr = max(best_gr, cur_gr)
                            best_adj = [list(adj[a]) for a in range(n)]
                            improved = True
                        else:
                            revert_swap(adj, u, v, x, y, adj_sets)
                        tried += 1
                        if tried > 200 or time.time() > deadline - 6:
                            break
                    if tried > 200 or time.time() > deadline - 6:
                        break
                if tried > 200 or time.time() > deadline - 6:
                    break
            if improved or tried > 200 or time.time() > deadline - 6:
                break

        if not improved:
            for _ in range(80):
                if time.time() > deadline - 6:
                    break
                a = random.randrange(n)
                b = random.choice(adj[a])
                c = random.randrange(n)
                d = random.choice(adj[c])
                if a == b or c == d or a == c or a == d or b == c or b == d:
                    continue
                if swap_adj(adj, a, b, c, d, adj_sets):
                    ng, _ = girth(adj, cur_gr)
                    if ng >= cur_gr:
                        cur_gr = ng
                        best_gr = max(best_gr, cur_gr)
                        best_adj = [list(adj[a]) for a in range(n)]
                    else:
                        revert_swap(adj, a, b, c, d, adj_sets)

    final_adj = best_adj
    final_gr, bad = girth(final_adj, g)
    if not bad and valid(final_adj):
        edges = []
        for u in range(n):
            for v in final_adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    return luw_construct(k, g)
