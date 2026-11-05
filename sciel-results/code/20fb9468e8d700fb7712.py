"""Bit-parallel greedy snake-in-the-box in Q_9.

Validity is tracked with a single 512-bit integer `forbidden` whose bit v is set
iff v is at Hamming distance 1 from some path vertex. A candidate nxt is valid
iff bit nxt is clear in `forbidden` AND nxt not in `used`. This makes the
9-neighbor check a single native integer AND per candidate instead of a Python
set scan, so many randomized greedy passes fit in the time budget.
"""
import random
import time

N = 9
SIZE = 1 << N  # 512


def _precompute():
    adj_mask = [0] * SIZE
    for v in range(SIZE):
        m = 0
        for b in range(N):
            m |= 1 << (v ^ (1 << b))
        adj_mask[v] = m
    return adj_mask


ADJ = _precompute()


def _greedy(seed, rng):
    path = [seed]
    used = {seed}
    forbidden = ADJ[seed]
    cur = seed
    while True:
        found = False
        for b in rng.sample(range(N), N):
            v = cur ^ (1 << b)
            if (forbidden >> v) & 1:
                continue
            if v in used:
                continue
            path.append(v)
            used.add(v)
            forbidden |= ADJ[v]
            forbidden &= ~ADJ[cur]
            cur = v
            found = True
            break
        if not found:
            return path


def construct():
    rng = random.Random(123456789)
    deadline = time.time() + 112.0
    best = _greedy(0, rng)
    best_len = len(best)

    # Phase 1: many randomized greedy passes from the all-zero vertex.
    while time.time() < deadline:
        p = _greedy(0, rng)
        if len(p) > best_len:
            best, best_len = p, len(p)

    # Phase 2: greedy from random seeds (diverse starts can escape the
    # "0-start" basin and occasionally produce longer snakes).
    while time.time() < deadline:
        s = rng.randrange(SIZE)
        p = _greedy(s, rng)
        if len(p) > best_len:
            best, best_len = p, len(p)

    return best
