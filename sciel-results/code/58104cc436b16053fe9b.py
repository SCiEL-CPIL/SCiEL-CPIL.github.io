import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 5.0
_SEED = 20240601


def _score(adj, n):
    """Return max over triangles of deg(u)+deg(v)+deg(w)."""
    deg = [adj[i].__len__() if False else len(adj[i]) for i in range(n)]
    best = 0
    for u in range(n):
        au = adj[u]
        du = deg[u]
        for i in range(len(au)):
            v = au[i]
            av = adj[v]
            s = du + deg[v]
            if s <= best:
                continue
            # common neighbours of u and v
            for w in au[i + 1:]:
                if w in av:
                    val = s + deg[w]
                    if val > best:
                        best = val
    return best


def _edge_swap(adj, n, rng):
    """Pick a non-edge and an edge, swap. Returns (u,v,w,x) meaning
    remove (w,x), add (u,v)."""
    while True:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v or v in adj[u]:
            continue
        w = rng.randrange(n)
        x = rng.randrange(n)
        if w == x or x not in adj[w]:
            continue
        if (u, v) == (w, x) or (v, u) == (w, x):
            continue
        break
    return u, v, w, x


def construct():
    t0 = time.time()
    rng = random.Random(_SEED)
    n = 16

    # Seed: K_{8,8} plus one internal edge
    adj = [set() for _ in range(n)]
    for u in range(8):
        for v in range(8, 16):
            adj[u].add(v)
            adj[v].add(u)
    adj[0].add(1)
    adj[1].add(0)

    cur_score = _score(adj, n)
    best_adj = [set(s) for s in adj]
    best_score = cur_score

    def elapsed():
        return time.time() - t0

    def time_left():
        return _DEADLINE - _MARGIN - elapsed()

    T0 = 12.0
    Tend = 0.15
    total_steps = 200000
    step = 0
    best_step = 0

    while step < total_steps:
        if time_left() <= 0:
            break
        step += 1
        u, v, w, x = _edge_swap(adj, n, rng)

        adj[u].add(v)
        adj[v].add(u)
        adj[w].discard(x)
        adj[x].discard(w)

        new_score = _score(adj, n)
        delta = new_score - cur_score

        if delta <= 0:
            cur_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = [set(s) for s in adj]
                best_step = step
        else:
            frac = step / max(1, total_steps)
            T = T0 * (Tend / T0) ** frac
            if rng.random() < math.exp(-delta / max(T, 1e-9)):
                cur_score = new_score
            else:
                # revert
                adj[u].discard(v)
                adj[v].discard(u)
                adj[w].add(x)
                adj[x].add(w)

    # Final polish: simple local search from best
    adj = [set(s) for s in best_adj]
    cur_score = best_score
    improved = True
    while improved and time_left() > 0:
        improved = False
        for u in range(n):
            for v in list(adj[u]):
                if v <= u:
                    continue
                for w in range(n):
                    for x in range(n):
                        if time_left() <= 0:
                            return _finalize(best_adj, n)
                        if w == x or (w, x) == (u, v) or (x, w) == (u, v):
                            continue
                        if x not in adj[w]:
                            continue
                        if v in adj[w] and x in adj[u]:
                            continue
                        # try remove (w,x) add (u,v)
                        adj[w].discard(x)
                        adj[x].discard(w)
                        adj[u].add(v)
                        adj[v].add(u)
                        ns = _score(adj, n)
                        if ns < cur_score:
                            cur_score = ns
                            if ns < best_score:
                                best_score = ns
                                best_adj = [set(s) for s in adj]
                            improved = True
                        else:
                            adj[w].add(x)
                            adj[x].add(w)
                            adj[u].discard(v)
                            adj[v].discard(u)

    return _finalize(best_adj, n)


def _finalize(adj, n):
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    edges.sort()
    return {'n': n, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print(len(res['edges']))
