import time
import random

_N = 15
_ALL = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START = time.time()
_DEADLINE = _START + 50.0


def _validity(g):
    deg = [0] * _N
    adj = [set() for _ in range(_N)]
    for (u, v) in g:
        deg[u] += 1
        deg[v] += 1
        adj[u].add(v)
        adj[v].add(u)
    
    # Check all 3-edge paths
    for (u, v) in g:
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if deg[s] == deg[t]:
                    return (False, (s, u, v, t))
    return (True, None)


def _solve_once(rng):
    # Start with K_7,8 (Bipartite, valid)
    g = set()
    for u in range(7):
        for v in range(7, 15):
            g.add((u, v))
    
    # Try to add intra-partition edges
    remaining = [e for e in _ALL if e not in g]
    rng.shuffle(remaining)
    
    for e in remaining:
        if time.time() > _DEADLINE:
            break
        g2 = g | {e}
        if _validity(g2)[0]:
            g = g2
            
    # Local search
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for (u, v) in g:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def edge_violations(u, v):
        # Count paths s-u-v-t or s-v-u-t where deg(s)==deg(t)
        # This is an approximation for search
        a = [deg[x] for x in adj[u] if x != v]
        b = [deg[x] for x in adj[v] if x != u]
        ca = {}
        for d in a:
            ca[d] = ca.get(d, 0) + 1
        cb = {}
        for d in b:
            cb[d] = cb.get(d, 0) + 1
        tot = 0
        for d, cnt in ca.items():
            if d in cb:
                tot += cnt * cb[d]
        return tot

    def add_edge(u, v):
        g.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1

    def remove_edge(u, v):
        g.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
        deg[u] -= 1
        deg[v] -= 1

    best_g = set(g)
    best_sz = len(g)
    
    for _ in range(1000):
        if time.time() > _DEADLINE:
            break
        # Try adding a random edge
        e = rng.choice(_ALL)
        if e not in g:
            g2 = g | {e}
            if _validity(g2)[0]:
                g = g2
                if len(g) > best_sz:
                    best_sz = len(g)
                    best_g = set(g)
        else:
            pass
            
    if len(best_g) >= len(g):
        g = best_g
    return g


def construct():
    best = None
    best_sz = -1
    base_seed = 12345
    restart = 0
    while time.time() < _DEADLINE:
        rng = random.Random(base_seed + 1000 * restart)
        g = _solve_once(rng)
        if _validity(set(g))[0]:
            if len(g) > best_sz:
                best_sz = len(g)
                best = list(g)
        restart += 1
    if best is None:
        best = [(u, v) for u in range(7) for v in range(7, 15)]
    out = [[int(u), int(v)] for (u, v) in best]
    return out
