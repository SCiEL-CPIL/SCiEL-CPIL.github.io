"""Exact solver: scan range length W upward with fast bipartite-matching feasibility."""
import time
from math import gcd

# EVOLVE-BLOCK-START
_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAX = 1000000
_N = len(_DIVS)

def _max_matching_with_assign(adj):
    n = len(adj)
    match = [-1] * n
    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match[v] == -1 or dfs(match[v], seen):
                match[v] = u
                return True
        return False
    for u in range(n):
        if not dfs(u, [False] * n):
            return None
    assign = [0] * n
    for r in range(n):
        assign[match[r]] = r
    return assign

def _try_W(W, deadline):
    upper = _MAX - W
    if upper < 1:
        return None
    covered = bytearray(upper + 1)
    for d in _DIVS:
        for m in range(d, upper + 1, d):
            covered[m] = 1
    for m in range(1, upper + 1):
        if time.time() > deadline:
            return None
        if not covered[m]:
            continue
        hi = m + W
        adj = []
        ok = True
        for d in _DIVS:
            first = ((m + d - 1) // d) * d
            vals = []
            v = first
            while v <= hi:
                vals.append(v)
                v += d
            if not vals:
                ok = False
                break
            adj.append(vals)
        if not ok:
            continue
        all_vals = sorted(set(v for vals in adj for v in vals))
        if len(all_vals) < _N:
            continue
        v2i = {v: i for i, v in enumerate(all_vals)}
        adj2 = [[v2i[v] for v in vals] for vals in adj]
        assign = _max_matching_with_assign(adj2)
        if assign is not None:
            return [all_vals[r] for r in assign]
    return None

def construct():
    deadline = time.time() + 80
    for W in range(9, 2000):
        if time.time() > deadline:
            break
        assign = _try_W(W, deadline)
        if assign is not None:
            if len(set(assign)) == _N and all(1 <= x <= _MAX for x in assign):
                good = True
                for i, d in enumerate(_DIVS):
                    if assign[i] % d != 0:
                        good = False
                        break
                if good:
                    return [int(x) for x in assign]
    return list(_DIVS)
# EVOLVE-BLOCK-END
