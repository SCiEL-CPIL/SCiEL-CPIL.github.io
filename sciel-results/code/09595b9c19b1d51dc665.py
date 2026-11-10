import time
from math import gcd
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 80.0  # explicit wall-clock guard (evaluator kills at 90s)

    N = 512
    vals = np.arange(1, N + 1, dtype=np.int64)

    # ---- Precompute the full compatibility matrix (lcm(a,b) <= N) ----
    G = np.empty((N, N), dtype=bool)
    for i in range(N):
        a = vals[i]
        row = (a // np.gcd(a, vals)) * vals
        G[i] = row <= N
    # diagonal (self) is always True; keep it, harmless.

    # ---- Divisor-set seeds (all divisors of L form a valid clique) ----
    def get_divisors(n):
        d = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                d.add(i)
                d.add(n // i)
            i += 1
        return d

    div_sets = []
    for L in range(1, N + 1):
        divs = get_divisors(L)
        div_sets.append((len(divs), L, divs))
    div_sets.sort(key=lambda t: (-t[0], t[1]))

    # ---- Deterministic greedy expansion using the compatibility matrix ----
    def expand(s):
        s = set(s)
        while time.time() < deadline:
            if len(s) == N:
                break
            arr = np.fromiter(s, dtype=np.int64, count=len(s))
            mask = G[arr - 1].all(axis=0)   # True where candidate compatible with all of s
            mask[arr - 1] = False           # exclude already-present
            cand = np.flatnonzero(mask)
            if cand.size == 0:
                break
            # score = number of compatible partners in s; take the best (stable -> deterministic)
            scores = G[arr - 1][:, cand].sum(axis=0)
            order = np.argsort(-scores, kind="stable")
            best = int(cand[order[0]])
            s.add(best)
        return s

    # ---- Local search: try removals, re-expand, keep improvement ----
    def local_search(s):
        s = set(s)
        best = len(s)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            lst = list(s)
            # 1-exchange: remove one element, re-expand
            for rc in lst:
                if time.time() > deadline:
                    break
                t = set(s)
                t.discard(rc)
                e = expand(t)
                if len(e) > best:
                    s, best = e, len(e)
                    improved = True
                    break
            if improved or time.time() > deadline:
                continue
            # 2-exchange: remove two elements, re-expand (escape 1-exchange optima)
            m = len(lst)
            for i in range(m):
                if time.time() > deadline:
                    break
                for j in range(i + 1, m):
                    t = set(s)
                    t.discard(lst[i])
                    t.discard(lst[j])
                    e = expand(t)
                    if len(e) > best:
                        s, best = e, len(e)
                        improved = True
                        break
                if improved:
                    break
        return s

    # ---- Build candidate seeds: top divisor sets + a few random starts ----
    seeds = []
    for _, L, divs in div_sets[:6]:
        seeds.append(set(divs))
    for _ in range(8):
        rs = set()
        for x in range(1, 129):
            if random.random() < 0.35:
                arr = np.fromiter(rs, dtype=np.int64, count=len(rs))
                if arr.size == 0 or bool(G[arr - 1, x - 1].all()):
                    rs.add(x)
        seeds.append(rs)

    # ---- Evaluate each seed: expand -> local search, keep the best ----
    best_set = None
    best_size = -1
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        e = expand(set(seed))
        ls = local_search(e)
        if len(ls) > best_size:
            best_size, best_set = len(ls), ls

    # ---- Final polish on the best set ----
    if best_set is not None:
        best_set = expand(set(best_set))

    if best_set is None or len(best_set) == 0:
        # guaranteed-valid fallback: divisors of 512
        best_set = get_divisors(512)

    return list(best_set)
