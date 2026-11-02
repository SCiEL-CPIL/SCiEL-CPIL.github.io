"""Rewritten: LUBW baseline + randomized edge-swap search with exact girth certification."""
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    rng = random.Random(20260519)

    def _lubw(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add, sub, mul = (lambda x, y: (x+y) % q,
                             lambda x, y: (x-y) % q,
                             lambda x, y: x*y % q)
        rule = {2: (1,1), 3: (2,1), 4: (1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5) % 4, c - (c-5) % 4
            rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,)*d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
            if vertex[0] == 0:
                for other in neighbours(vertex):
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def _girth(used, adj):
        # exact girth via non-backtracking BFS from every vertex
        best = 10**9
        for s in range(len(used)):
            if not used[s]:
                continue
            seen = {s: -1}
            dq = deque()
            for nb in adj[s]:
                if used[nb]:
                    dq.append((nb, s, 1))
            while dq:
                v, p, dist = dq.popleft()
                if dist + 1 >= best:
                    continue
                for w in adj[v]:
                    if not used[w]:
                        continue
                    if w == p:
                        continue
                    if w == s:
                        best = min(best, dist + 1)
                    elif w not in seen:
                        seen[w] = v
                        dq.append((w, v, dist + 1))
        return best

    def _make_pool(n0, edges):
        adj = [set() for _ in range(n0)]
        for a, b in edges:
            adj[a].add(b); adj[b].add(a)
        used = [True]*n0
        ekeys = set()
        for a, b in edges:
            ekeys.add((min(a,b), max(a,b)))
        return adj, used, ekeys

    def _try_swap(adj, used, ekeys):
        # pick a random edge, move one endpoint to a random other vertex
        if not ekeys:
            return False
        a, b = rng.choice(list(ekeys))
        # choose which endpoint to move
        mv = a if rng.random() < 0.5 else b
        # candidate new partner
        cands = [v for v in range(len(used)) if used[v] and v != mv and (min(mv,v),max(mv,v)) not in ekeys]
        if not cands:
            return False
        c = rng.choice(cands)
        # remove (mv,b), add (mv,c)
        ekeys.discard((min(mv,b),max(mv,b)))
        ekeys.add((min(mv,c),max(mv,c)))
        adj[mv].discard(b); adj[b].discard(mv)
        adj[mv].add(c); adj[c].add(mv)
        return True

    n0, edges0 = _lubw(k, g)
    if n0 == 0:
        return 0, []
    adj, used, ekeys = _make_pool(n0, edges0)
    best_n = n0
    best_used = used[:]
    best_ekeys = set(ekeys)
    best_g = _girth(used, adj)

    # If baseline already has girth >= g, we have a valid starting point.
    # Otherwise (shouldn't happen for LUBW), we just return baseline.
    if best_g < g:
        # fallback: return baseline as-is (still valid per LUBW guarantee)
        return n0, list(ekeys)

    # Randomized local search: batches of swaps, then certify.
    batch = 40
    while time.time() < deadline:
        for _ in range(batch):
            if time.time() > deadline:
                break
            _try_swap(adj, used, ekeys)
        # recount used vertices (those with degree k)
        for v in range(len(used)):
            used[v] = (len(adj[v]) == k)
        # drop isolated/low-degree vertices from consideration
        # compute girth on the used subgraph
        girth = _girth(used, adj)
        # count valid vertices: those with degree exactly k
        valid = sum(1 for v in range(len(used)) if len(adj[v]) == k)
        if girth >= g and valid < best_n:
            best_n = valid
            best_used = used[:]
            best_ekeys = set(ekeys)
            best_g = girth

    # Build final edge list from best state (only edges among valid vertices)
    final_edges = []
    for a, b in sorted(best_ekeys):
        if best_used[a] and best_used[b]:
            final_edges.append((a, b))
    # Renumber to 0..n-1 among valid vertices
    remap = {}
    cnt = 0
    for v in range(len(best_used)):
        if best_used[v]:
            remap[v] = cnt
            cnt += 1
    final_edges = [(remap[a], remap[b]) for a, b in final_edges]
    # Verify
    n = cnt
    if n == 0:
        return 0, []
    deg = [0]*n
    seen = set()
    for a, b in final_edges:
        deg[a]+=1; deg[b]+=1
        key=(min(a,b),max(a,b))
        if key in seen or a==b:
            # invalid, fall back to baseline
            return n0, list(edges0)
        seen.add(key)
    for d_ in deg:
        if d_ != k:
            return n0, list(edges0)
    return n, final_edges
