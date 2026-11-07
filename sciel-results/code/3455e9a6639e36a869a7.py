# EVOLVE-BLOCK-START
import time as _time
import itertools as _it

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _valid_partition(parts, n, adj):
    seen = [False] * n
    for part in parts:
        if len(part) == 0:
            return False
        for v in part:
            if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= n or seen[v]:
                return False
            seen[v] = True
        ok_e = True
        ok_n = True
        for a, b in _it.combinations(part, 2):
            if adj[a][b]:
                ok_n = False
            else:
                ok_e = False
        if not ok_e and not ok_n:
            return False
    return all(seen)

def construct():
    t0 = _time.time()
    deadline = t0 + 85.0
    n = int(P['n'])
    adj = [[False] * n for _ in range(n)]
    for a, b in P['edges']:
        ai, bi = int(a), int(b)
        adj[ai][bi] = True
        adj[bi][ai] = True

    best = None
    best_len = n + 1

    # Greedy initial upper bound.
    def greedy():
        nonlocal best, best_len
        rem = set(range(n))
        gs = []
        while rem:
            mis, mc = [], []
            for v in list(rem):
                if all(not adj[v][u] for u in mis):
                    mis.append(v)
                if all(adj[v][u] for u in mc):
                    mc.append(v)
            s = mis if len(mis) >= len(mc) else mc
            gs.append(s)
            for v in s:
                rem.discard(v)
        if len(gs) < best_len:
            best = [list(s) for s in gs]
            best_len = len(gs)

    greedy()

    import sys
    sys.setrecursionlimit(100000)

    def set_valid(part):
        if len(part) <= 1:
            return True
        ok_e = True
        ok_n = True
        for a, b in _it.combinations(part, 2):
            if adj[a][b]:
                ok_n = False
            else:
                ok_e = False
            if not ok_e and not ok_n:
                return False
        return True

    deg = [sum(1 for j in range(n) if adj[i][j]) for i in range(n)]
    order = sorted(range(n), key=lambda v: (-deg[v], v))

    def rec(idx, parts, cur_len):
        nonlocal best, best_len
        if _time.time() > deadline:
            return
        if cur_len > best_len:
            return
        if idx == n:
            if cur_len < best_len:
                best = [list(p) for p in parts]
                best_len = cur_len
            return
        v = order[idx]
        # Try placing v into each existing part.
        for pi in range(len(parts)):
            trial = parts[pi] + [v]
            if set_valid(trial):
                new_parts = list(parts)
                new_parts[pi] = trial
                rec(idx + 1, new_parts, cur_len)
        # Try a new part.
        if cur_len + 1 < best_len:
            new_parts = list(parts)
            new_parts.append([v])
            rec(idx + 1, new_parts, cur_len + 1)

    rec(0, [], 0)

    if best is None:
        best = [[v] for v in range(n)]

    out = [[int(x) for x in part] for part in best]
    if not _valid_partition(out, n, adj):
        out = [[int(v)] for v in range(n)]
    return out
# EVOLVE-BLOCK-END
