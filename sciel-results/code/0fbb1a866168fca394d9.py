import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 50.0

    def time_left():
        return DEADLINE - time.time()

    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def build(edges):
        adj = [set() for _ in range(n)]
        deg = [0] * n
        eset = set()
        for (u, v) in edges:
            if u == v:
                continue
            e = (u, v) if u < v else (v, u)
            if e in eset:
                continue
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            eset.add(e)
        return adj, deg, eset

    def is_valid(adj, deg):
        for s in range(n):
            ds = deg[s]
            ns = adj[s]
            for t in range(s + 1, n):
                if deg[t] != ds:
                    continue
                nt = adj[t]
                for u in ns:
                    if u == t:
                        continue
                    au = adj[u]
                    for v in nt:
                        if v == s or v == u:
                            continue
                        if u in au:
                            return False
        return True

    def add_edge(adj, deg, eset, e):
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        eset.add(e)

    def remove_edge(adj, deg, eset, e):
        u, v = e
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1
        eset.discard(e)

    best = None
    best_count = -1

    def update_best(eset, count):
        nonlocal best, best_count
        if count > best_count:
            best_count = count
            best = sorted(eset)

    def greedy_add(adj, deg, eset, count):
        non_edges = [e for e in all_edges if e not in eset]
        random.shuffle(non_edges)
        improved = True
        while improved and time_left() > 4:
            improved = False
            for e in non_edges:
                if e in eset:
                    continue
                add_edge(adj, deg, eset, e)
                if is_valid(adj, deg):
                    count += 1
                    improved = True
                else:
                    remove_edge(adj, deg, eset, e)
            non_edges = [e for e in non_edges if e not in eset]
        return count

    def local_search(adj, deg, eset, count):
        nonlocal best, best_count
        stall = 0
        while time_left() > 4:
            non_edges = [e for e in all_edges if e not in eset]
            moved = False
            # try add one
            if non_edges:
                random.shuffle(non_edges)
                for e in non_edges[:20]:
                    if e in eset:
                        continue
                    add_edge(adj, deg, eset, e)
                    if is_valid(adj, deg):
                        count += 1
                        moved = True
                        break
                    else:
                        remove_edge(adj, deg, eset, e)
            if moved:
                stall = 0
                if count > best_count:
                    best_count = count
                    best = sorted(eset)
                continue
            # try remove one, add two
            found = False
            if non_edges and len(eset) > 0 and len(non_edges) >= 2:
                for _ in range(30):
                    e_rm = random.choice(list(eset))
                    cands = [e for e in non_edges if e != e_rm]
                    if len(cands) < 2:
                        break
                    e1, e2 = random.sample(cands, 2)
                    remove_edge(adj, deg, eset, e_rm)
                    add_edge(adj, deg, eset, e1)
                    add_edge(adj, deg, eset, e2)
                    if is_valid(adj, deg):
                        count += 1
                        found = True
                        break
                    else:
                        remove_edge(adj, deg, eset, e2)
                        remove_edge(adj, deg, eset, e1)
                        add_edge(adj, deg, eset, e_rm)
            if found:
                stall = 0
                if count > best_count:
                    best_count = count
                    best = sorted(eset)
                continue
            # kick: remove random edge, greedy re-add
            if len(eset) > 0:
                e_rm = random.choice(list(eset))
                remove_edge(adj, deg, eset, e_rm)
                count -= 1
                count = greedy_add(adj, deg, eset, count)
                if count > best_count:
                    best_count = count
                    best = sorted(eset)
            stall += 1
            if stall > 50:
                break
        return count

    # seeds
    seeds = []
    for a in range(1, n):
        b = n - a
        if a == b:
            continue
        edges = []
        for i in range(a):
            for j in range(a, n):
                edges.append((i, j))
        seeds.append(edges)
    # star
    seeds.append([(0, v) for v in range(1, n)])

    # sort seeds by edge count desc so we start with best
    seeds.sort(key=lambda es: -len(es))

    for seed_edges in seeds:
        if time_left() < 6:
            break
        adj, deg, eset = build(seed_edges)
        if not is_valid(adj, deg):
            continue
        count = len(eset)
        # a few greedy passes with different random orders
        for _ in range(3):
            count = greedy_add(adj, deg, eset, count)
        count = local_search(adj, deg, eset, count)
        if count > best_count:
            best_count = count
            best = sorted(eset)

    if best is None:
        best = [(0, v) for v in range(1, n)]

    # final defensive check
    adj, deg, eset = build(best)
    if not is_valid(adj, deg):
        best = [(0, v) for v in range(1, n)]

    return [[u, v] for (u, v) in best]
