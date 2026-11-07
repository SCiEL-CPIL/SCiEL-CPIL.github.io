"""Rewritten solver: maximize a subset S of 2..256 with no triple a|b*c."""
import time
import math
import random
import sys
from collections import Counter, defaultdict

# ---------- fixed problem data ----------
N = 256
UNIVERSE = list(range(2, N + 1))

def _factorize(m):
    d = {}
    t = m
    p = 2
    while p * p <= t:
        while t % p == 0:
            d[p] = d.get(p, 0) + 1
            t //= p
        p += 1
    if t > 1:
        d[t] = d.get(t, 0) + 1
    return d

FACT = {m: _factorize(m) for m in UNIVERSE}
IDX = {v: i for i, v in enumerate(UNIVERSE)}

# conflict[a][b] = True if a|b*c can hold for some c (=> a,b can't both be in S)
CONFLICT = [None] * N
for a in range(2, N + 1):
    fa = FACT[a]
    row = bytearray(N)
    for b in range(2, N + 1):
        if b == a:
            continue
        fb = FACT[b]
        ok = True
        for p, e in fa.items():
            if fb.get(p, 0) < e:
                ok = False
                break
        if ok:
            row[b] = 1
    CONFLICT[a] = row

def _conflict_count(i, S):
    r = CONFLICT[UNIVERSE[i]]
    c = 0
    for j in S:
        if r[UNIVERSE[j]]:
            c += 1
    return c

def _is_valid(S):
    for i in S:
        r = CONFLICT[UNIVERSE[i]]
        for j in S:
            if j != i and r[UNIVERSE[j]]:
                return False
    return True

def _greedy():
    S = set()
    for a in range(2, N + 1):
        fa = FACT[a]
        ok = True
        for b in range(2, N + 1):
            if b == a or b in S:
                continue
            fb = FACT[b]
            if all(fb.get(p, 0) >= e for p, e in fa.items()):
                ok = False
                break
        if ok:
            S.add(a)
    return S

def _local_search(S, deadline, rng):
    S = set(S)
    best = len(S)
    while time.time() < deadline:
        improved = False
        # 1) single removals
        for a in sorted(S, key=lambda x: _conflict_count(IDX[x], S), reverse=True):
            cand = S - {a}
            if len(cand) > best:
                S = cand
                best = len(cand)
                improved = True
        # 2) single additions
        for a in range(2, N + 1):
            if a in S:
                continue
            if _conflict_count(IDX[a], S) == 0:
                S.add(a)
                best = len(S)
                improved = True
        # 3) 2-exchanges: remove one, add one
        for a in sorted(S, key=lambda x: _conflict_count(IDX[x], S), reverse=True):
            for b in range(2, N + 1):
                if b in S or b == a:
                    continue
                if _conflict_count(IDX[b], S - {a}) == 0:
                    S = S - {a} | {b}
                    best = len(S)
                    improved = True
        if not improved:
            break
    return S

def construct():
    t0 = time.time()
    deadline = t0 + 50.0          # safety margin before the 90s kill
    rng = random.Random(12345)

    best = _greedy()
    best = _local_search(best, deadline, rng)

    # extra random-restart local searches while time remains
    while time.time() < deadline - 1.0:
        S = set(UNIVERSE)
        # random perturbation: drop a random subset, then greedily refill
        for _ in range(rng.randint(3, 9)):
            S.discard(rng.choice(list(S)))
        S = _local_search(S, deadline, rng)
        if len(S) > len(best):
            best = S

    res = [int(x) for x in sorted(best)]
    # final safety: never exceed the deadline
    if time.time() > t0 + 85.0:
        return res
    return res

# ---------- self-test (skipped by evaluator) ----------
if __name__ == "__main__":
    _s = construct()
    assert _is_valid(_s), "invalid set produced"
    print("size:", len(_s))
    print(_s)
