"""Direct approach for minimum maximal no-isosceles set."""
N = 16
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

def _precompute():
    pts = [_idx_point(i) for i in range(TOTAL)]
    iso_bits = [[0] * TOTAL for _ in range(TOTAL)]
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                continue
            a = pts[i]; b = pts[j]
            da = (a[0]-b[0])**2 + (a[1]-b[1])**2
            bits = 0
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                c = pts[k]
                db = (a[0]-c[0])**2 + (a[1]-c[1])**2
                dc = (b[0]-c[0])**2 + (b[1]-c[1])**2
                if da == db or da == dc or db == dc:
                    bits |= (1 << k)
            iso_bits[i][j] = bits
    return iso_bits

_iso_bits = _precompute()

def _greedy(order):
    chosen = []
    blocked = 0
    for p in order:
        if (blocked >> p) & 1:
            continue
        new_block = 0
        for q in chosen:
            new_block |= _iso_bits[p][q]
        blocked |= new_block
        blocked |= (1 << p)
        chosen.append(p)
    return chosen, blocked

def _is_maximal(chosen, blocked):
    cmask = 0
    for c in chosen:
        cmask |= (1 << c)
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~cmask
    return (blocked & unchosen) == unchosen

def _try_remove(chosen):
    n = len(chosen)
    for i in range(n):
        new_chosen = chosen[:i] + chosen[i+1:]
        blocked = 0
        for a in new_chosen:
            blocked |= (1 << a)
        for a in new_chosen:
            for b in new_chosen:
                if a == b:
                    continue
                blocked |= _iso_bits[a][b]
        cmask = 0
        for c in new_chosen:
            cmask |= (1 << c)
        all_mask = (1 << TOTAL) - 1
        unchosen = all_mask & ~cmask
        if (blocked & unchosen) == unchosen:
            return new_chosen
    return None

def construct():
    all_idx = list(range(TOTAL))
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 325

    random.seed(42)
    order = all_idx[:]
    chosen, blocked = _greedy(order)
    if _is_maximal(chosen, blocked):
        best_size = len(chosen)
        best = chosen[:]

    for attempt in range(1000000):
        if time.time() > deadline - 5:
            break
        order = all_idx[:]
        random.shuffle(order)
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen[:]
                if best_size <= 12:
                    break

    if best is not None and time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            result = _try_remove(best)
            if result is not None:
                best = result
                best_size = len(best)
                improved = True

    if best is None:
        best = all_idx[:]
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
