import time
import random
from itertools import combinations

_DEADLINE = 110.0  # safety margin under 120s kill

def _validate(triples):
    # any two triples share at most one point; points in 0..15
    for a in triples:
        if len(set(a)) != 3:
            return False
        for x in a:
            if x < 0 or x > 15:
                return False
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            if len(set(triples[i]) & set(triples[j])) > 1:
                return False
    return True

def _independent_set_size(triples):
    # largest subset of {0..15} containing no triple (exact, n=16)
    tset = [set(t) for t in triples]
    n = len(tset)
    best = 0
    for mask in range(1 << 16):
        bits = bin(mask).count("1")
        if bits <= best:
            continue
        ok = True
        for s in tset:
            if (mask & s) == s:
                ok = False
                break
        if ok:
            best = bits
    return best

def _base_sts():
    # AG(2,4) lines -> Steiner triple system on 16 points (every pair in exactly one line)
    # GF(4): 0,1,w,w2 with w2=w+1, w^3=1
    def add(a, b):
        t = a + b
        if t == 3:
            t = 1
        return t
    pts = [0, 1, 2, 3]
    lines = []
    # 6 parallel classes, each 4 lines
    for m in [0, 1, 2, 3]:
        for b in [0, 1, 2, 3]:
            line = []
            for x in [0, 1, 2, 3]:
                y = (m * x + b) % 4
                line.append(4 * x + y)
            lines.append(line)
    for line in lines:
        lines.append([add(line[0], 4), add(line[1], 4), add(line[2], 4)])
    return lines  # 32 triples

def _swap(triples, i, a, b):
    # swap point a with b inside triple i (b must be free in triple i)
    t = triples[i]
    if b in t:
        return None
    nt = [b if x == a else x for x in t]
    triples[i] = nt
    return (i, a, b)

def _local_search(triples, deadline):
    triples = [list(t) for t in triples]
    cur_alpha = _independent_set_size(triples)
    best_alpha = cur_alpha
    best = [list(t) for t in triples]
    rng = random.Random(12345)
    while time.time() < deadline:
        i = rng.randrange(len(triples))
        a = rng.choice(triples[i])
        b = rng.randrange(16)
        if b in triples[i]:
            continue
        old = _swap(triples, i, a, b)
        if old is None:
            continue
        if not _validate(triples):
            triples[i] = [a if x == b else x for x in triples[i]]
            continue
        na = _independent_set_size(triples)
        if na <= cur_alpha:
            cur_alpha = na
            if na < best_alpha:
                best_alpha = na
                best = [list(t) for t in triples]
        else:
            triples[i] = [a if x == b else x for x in triples[i]]
    return best, best_alpha

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    base = _base_sts()
    # deterministic first pass (fixed seed) to guarantee reproducibility
    best, best_alpha = _local_search(base, deadline)
    # if time remains, try a few more seeds
    s = 101
    while time.time() < deadline - 5:
        rng = random.Random(s)
        tri = [list(t) for t in base]
        # small random perturbation
        for _ in range(4):
            i = rng.randrange(len(tri))
            a = rng.choice(tri[i])
            b = rng.randrange(16)
            if b in tri[i]:
                continue
            nt = [b if x == a else x for x in tri[i]]
            tri[i] = nt
            if not _validate(tri):
                tri[i] = [a if x == b else x for x in tri[i]]
        cand, ca = _local_search(tri, deadline)
        if ca < best_alpha:
            best, best_alpha = cand, ca
        s += 1
    return [list(t) for t in best]
