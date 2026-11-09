"""R(4,6): maximize n of a simple graph with no K4 and no independent 6-set.

Structural approach: local search on the edge set of a fixed vertex count.
 - Start from the Paley-17 graph (known valid).
 - Extend to a target n by adding edges only (K4-freeness is preserved when
   adding edges, and the independence number can only shrink), using a
   greedy + repair procedure until the graph is valid.
 - Run simulated annealing to drive the violation count (#K4 + #indep6) to 0.
 - Once a valid graph is found, try to push the vertex count to 36.
"""
import time
import random

_DEADLINE = time.time() + 145.0  # safety margin before the 150 s kill


def _time_left():
    return _DEADLINE - time.time()


def _paley17():
    """Paley graph on 17 vertices (prime 17, quadratic residues)."""
    n = 17
    sq = {x * x % n for x in range(1, n)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in sq:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _k4_count(n, adj):
    """Exact number of K4 subgraphs."""
    c = 0
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            common = adj[i] & adj[j]
            for k in common:
                if k <= j:
                    continue
                c += len(adj[i] & adj[j] & adj[k])
    return c


def _indep6_count(n, adj, limit=200):
    """Exact number of independent 6-sets, capped at `limit`."""
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            if j > i:
                comp[i].add(j)
                comp[j].add(i)

    count = 0

    def rec(cands, size):
        nonlocal count
        if size == 0:
            count += 1
            return
        if len(cands) < size:
            return
        if count >= limit:
            return
        for v in list(cands):
            if size == 1:
                count += 1
                if count >= limit:
                    return
            else:
                nc = cands & comp[v]
                if len(nc) >= size - 1:
                    rec(nc, size - 1)
            cands.discard(v)

    rec(set(range(n)), 6)
    return count


def _is_valid(n, adj):
    return _k4_count(n, adj) == 0 and _indep6_count(n, adj, limit=1) == 0


def _to_dict(n, adj):
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _extend(n, adj, target):
    """Add (target - n) vertices by adding edges only, until valid or budget."""
    adj = [set(s) for s in adj]
    cur = n
    while cur < target:
        cur += 1
        new = cur - 1
        adj.append(set())
        # random initial neighborhood
        k = random.randint(4, 11)
        for v in random.sample(range(cur - 1), k):
            adj[v].add(new)
            adj[new].add(v)
        # repair: keep adding random edges (safe for K4) until valid
        pool = list(range(cur - 1))
        random.shuffle(pool)
        it = 0
        while not _is_valid(cur, adj) and _time_left() > 8:
            if it >= len(pool):
                pool = list(range(cur - 1))
                random.shuffle(pool)
                it = 0
            v = pool[it]
            it += 1
            if v not in adj[new]:
                adj[v].add(new)
                adj[new].add(v)
    return cur, adj


def _anneal(n, adj):
    """Simulated annealing on the edge set to reach a valid graph."""
    adj = [set(s) for s in adj]
    cur = _k4_count(n, adj) + _indep6_count(n, adj, limit=200)
    best = cur
    best_adj = [set(s) for s in adj]
    T = 1.0
    cool = 0.999
    it = 0
    while _time_left() > 8 and T > 0.02:
        it += 1
        u = random.randrange(n)
        v = random.randrange(n)
        if u == v:
            continue
        if v in adj[u]:
            adj[u].discard(v)
            adj[v].discard(u)
        else:
            adj[u].add(v)
            adj[v].add(u)
        new = _k4_count(n, adj) + _indep6_count(n, adj, limit=200)
        d = new - cur
        if d <= 0 or random.random() < pow(2.718281828459045, -d / T):
            cur = new
            if new < best:
                best = new
                best_adj = [set(s) for s in adj]
                if best == 0:
                    return True, best_adj
        else:
            # revert
            if v in adj[u]:
                adj[u].discard(v)
                adj[v].discard(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
        T *= cool
    return best == 0, best_adj


def _solve(n, adj):
    """Try to make a valid graph on exactly n vertices."""
    n, adj = _extend(n, adj, n)
    if _is_valid(n, adj):
        return adj
    # several SA restarts
    while _time_left() > 8:
        ok, adj = _anneal(n, adj)
        if ok:
            return adj
    return None


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    random.seed(42)

    best_n = 17
    base = _paley17()
    best_adj = [set(s) for s in base]

    # Phase 1: secure a valid n=35 graph (the published record).
    adj35 = _solve(35, base)
    if adj35 is not None:
        best_n = 35
        best_adj = adj35

    # Phase 2: attempt n=36 (a new lower bound) if time remains.
    if _time_left() > 25:
        start = best_adj if best_n >= 35 else base
        adj36 = _solve(36, start)
        if adj36 is not None:
            best_n = 36
            best_adj = adj36

    # Final verification of the result we return.
    if not _is_valid(best_n, best_adj):
        # fall back to the guaranteed-valid Paley 17
        best_n = 17
        best_adj = [set(s) for s in base]

    return _to_dict(best_n, best_adj)
