"""Fresh design: greedy construction + multi-move local search for K_q(n, radius).

Entry point: construct(q, n, radius) -> list of codewords (lists of ints in 0..q-1),
distinct, minimizing the count while covering the whole space within Hamming radius.
"""
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    start = time.time()
    DEADLINE = start + 115.0          # safety margin before the 150s kill
    rng = random.Random(20240517)

    # ---------- helpers ----------
    def encode(w):
        v = 0
        for x in w:
            v = v * q + x
        return v

    def decode(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    def neighbors(w):
        """All words within Hamming distance radius of w (incl. w)."""
        res = [w]
        for i in range(n):
            for a in range(q):
                if a == w[i]:
                    continue
                w[i] = a
                res.append(w[:])
        w[i] = w[i]  # (loop var) restore handled below
        # restore properly
        w2 = w
        return res

    # rebuild neighbors correctly (the above is buggy; use clean version)
    def neigh(w):
        res = [w[:]]
        for i in range(n):
            orig = w[i]
            for a in range(q):
                if a == orig:
                    continue
                w[i] = a
                res.append(w[:])
            w[i] = orig
        return res

    # ---------- initial greedy ----------
    best = None
    best_count = None

    # seed: start with a few random codewords, then greedy add
    seed_size = max(1, min(8, (q ** n) // 100))
    for trial in range(6):
        if time.time() > DEADLINE:
            break
        # fresh random start
        cw = []
        used = set()
        for _ in range(seed_size):
            while True:
                w = [rng.randrange(q) for _ in range(n)]
                k = encode(w)
                if k not in used:
                    used.add(k)
                    cw.append(w)
                    break
        # greedy: repeatedly add the word covering the most uncovered
        covered = set()
        for w in cw:
            for g in neigh(w):
                covered.add(encode(g))
        total = q ** n
        while len(covered) < total:
            if time.time() > DEADLINE:
                break
            # pick the uncovered word with the largest ball (tie-break random)
            # sample a subset of uncovered words for speed when space is large
            uncovered = [v for v in range(total) if v not in covered]
            if len(uncovered) > 4000:
                cand = rng.sample(uncovered, 4000)
            else:
                cand = uncovered
            best_v = None
            best_gain = -1
            for v in cand:
                ball = neigh(decode(v))
                gain = sum(1 for g in ball if g not in covered)
                if gain > best_gain or (gain == best_gain and rng.random() < 0.5):
                    best_gain = gain
                    best_v = v
            if best_v is None or best_gain <= 0:
                # force add an uncovered word
                best_v = cand[0]
            w = decode(best_v)
            k = encode(w)
            if k in used:
                # pick another uncovered
                w = decode(uncovered[0])
                k = encode(w)
            used.add(k)
            cw.append(w)
            for g in neigh(w):
                covered.add(g)
        if len(covered) < total:
            continue
        if best is None or len(cw) < best_count:
            best = cw
            best_count = len(cw)

    if best is None:
        # fallback: all words (valid, large)
        return [list(t) for t in _itertools_product(range(q), repeat=n)]

    # ---------- local search on best ----------
    def is_covered(cw):
        covered = set()
        for w in cw:
            for g in neigh(w):
                covered.add(encode(g))
        return len(covered) == total

    def ball_set(cw):
        s = set()
        for w in cw:
            for g in neigh(w):
                s.add(encode(g))
        return s

    cur = [w[:] for w in best]
    cur_keys = set(encode(w) for w in cur)
    cur_covered = ball_set(cur)

    def try_removal(i):
        """Remove codeword i if still covered. Return new covered set or None."""
        new_cw = [w for j, w in enumerate(cur) if j != i]
        nc = ball_set(new_cw)
        if len(nc) == total:
            return new_cw, nc
        return None

    def try_swap(i, new_w):
        """Replace codeword i with new_w (must be distinct from others)."""
        k = encode(new_w)
        if k in cur_keys and k != encode(cur[i]):
            return None
        new_cw = [w if j != i else new_w for j, w in enumerate(cur)]
        nc = ball_set(new_cw)
        if len(nc) == total:
            return new_cw, nc
        return None

    def try_add(new_w):
        k = encode(new_w)
        if k in cur_keys:
            return None
        new_cw = cur + [new_w]
        nc = ball_set(new_cw)
        if len(nc) == total:
            return new_cw, nc
        return None

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # 1) removals
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            r = try_removal(i)
            if r:
                cur, cur_covered = r
                cur_keys = set(encode(w) for w in cur)
                improved = True
                break
        if improved:
            continue
        # 2) 1-for-1 swaps (random candidate words)
        if time.time() > DEADLINE:
            break
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            for _ in range(30):
                cand = [rng.randrange(q) for _ in range(n)]
                # bias toward words that are far / in sparse regions
                r = try_swap(i, cand)
                if r:
                    cur, cur_covered = r
                    cur_keys = set(encode(w) for w in cur)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # 3) 2-for-1: replace two codewords with one
        if time.time() > DEADLINE:
            break
        for _ in range(200):
            if time.time() > DEADLINE:
                break
            i, j = rng.sample(range(len(cur)), 2)
            # candidate: a word covering the union of their balls' uncovered parts
            cand = [rng.randrange(q) for _ in range(n)]
            new_cw = [w for k, w in enumerate(cur) if k != i and k != j] + [cand]
            nc = ball_set(new_cw)
            if len(nc) == total and len(new_cw) < len(cur):
                cur, cur_covered = new_cw, nc
                cur_keys = set(encode(w) for w in cur)
                improved = True
                break
        if improved:
            continue
        # 4) 3-for-2: replace three codewords with two
        if time.time() > DEADLINE:
            break
        for _ in range(100):
            if time.time() > DEADLINE:
                break
            idx = rng.sample(range(len(cur)), 3)
            kept = [w for k, w in enumerate(cur) if k not in idx]
            a = [rng.randrange(q) for _ in range(n)]
            b = [rng.randrange(q) for _ in range(n)]
            new_cw = kept + [a, b]
            nc = ball_set(new_cw)
            if len(nc) == total and len(new_cw) < len(cur):
                cur, cur_covered = new_cw, nc
                cur_keys = set(encode(w) for w in cur)
                improved = True
                break

    # final safety: ensure validity
    if not is_covered(cur):
        # top up greedily
        covered = ball_set(cur)
        while len(covered) < total and time.time() < DEADLINE:
            uncovered = [v for v in range(total) if v not in covered]
            cand = uncovered if len(uncovered) <= 2000 else rng.sample(uncovered, 2000)
            bv, bg = None, -1
            for v in cand:
                ball = neigh(decode(v))
                gain = sum(1 for g in ball if g not in covered)
                if gain > bg:
                    bg, bv = gain, v
            w = decode(bv)
            if encode(w) in cur_keys:
                w = decode(uncovered[0])
            cur_keys.add(encode(w))
            cur.append(w)
            for g in neigh(w):
                covered.add(g)

    return [w[:] for w in cur]


def _itertools_product(*args, **kwargs):
    import itertools
    return itertools.product(*args, **kwargs)
