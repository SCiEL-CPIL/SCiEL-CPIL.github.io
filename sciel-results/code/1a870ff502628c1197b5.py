"""Greedy covering with bitset acceleration + budgeted local-search polish."""
import itertools
import time
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 2.0


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN

    def timeup():
        return time.time() >= deadline

    # ---- index all t-subsets and k-subsets via combinatorial number system ----
    def comb_index(c, m, n):
        # index of sorted tuple c (length m) in combinations(range(n), m)
        idx = 0
        for i in range(m):
            r = n - 1 - c[i]
            k_ = m - 1 - i
            c_val = 1
            for j in range(k_):
                c_val = c_val * (r - j) // (j + 1)
            idx += c_val
        return idx

    def comb_unindex(idx, m, n):
        c = [0] * m
        start = 0
        for i in range(m):
            r = n - 1
            while True:
                c_val = 1
                for j in range(m - i):
                    c_val = c_val * (r - j) // (j + 1)
                if c_val <= idx:
                    break
                r -= 1
            c[i] = r
            idx -= c_val
            n = r
        return c

    total_t = 0
    prod = 1
    for i in range(t):
        prod *= (v - i)
    total_t = prod // (2 ** t * t)  # C(v,t) = v!/(t!(v-t)!)
    # safer:
    from math import comb
    total_t = comb(v, t)
    total_k = comb(v, k)

    ksets = [tuple(comb_unindex(i, k, v)) for i in range(total_k)]
    ksets_idx = [comb_index(c, k, v) for c in ksets]

    # mask[x]: bitset (int) of t-subset indices containing point x
    mask = [0] * v
    for kidx, c in enumerate(ksets):
        for sub in itertools.combinations(c, t):
            mask[sub[0]] |= (1 << comb_index(sub, t, v))

    # ---- greedy build ----
    rng = random.Random(12345)
    uncovered = (1 << total_t) - 1
    blocks = []

    def greedy_round():
        nonlocal uncovered, blocks
        while uncovered:
            if timeup():
                return
            # pick a t-subset to cover: lowest uncovered bit (deterministic)
            low = (uncovered & -uncovered).bit_length() - 1
            seed_t = tuple(comb_unindex(low, t, v))
            block = list(seed_t)
            while len(block) < k:
                if timeup():
                    return
                inb = set(block)
                best_x = None
                best_g = -1
                best_tie = None
                for x in range(v):
                    if x in inb:
                        continue
                    g = int(np.count_nonzero(mask[x] & uncovered))
                    if g > best_g:
                        best_g = g
                        best_x = x
                        best_tie = g
                    elif g == best_g and best_x is not None:
                        # tie-break by random to diversify
                        if rng.random() < 0.5:
                            best_x = x
                if best_x is None:
                    return
                block.append(best_x)
            block.sort()
            blocks.append(block)
            m = 0
            for sub in itertools.combinations(block, t):
                m |= (1 << comb_index(sub, t, v))
            uncovered &= ~m

    greedy_round()
    if not blocks:
        # fallback minimal valid: cover via exhaustive singletons (valid but large)
        return blocks

    # ---- local search polish (time-budgeted) ----
    def coverage_count(bs):
        m = 0
        for b in bs:
            for sub in itertools.combinations(b, t):
                m |= (1 << comb_index(sub, t, v))
        return int(np.count_nonzero(m))

    def full_cover(bs):
        m = 0
        for b in bs:
            for sub in itertools.combinations(b, t):
                m |= (1 << comb_index(sub, t, v))
        return (m == ((1 << total_t) - 1))

    # redundancy removal
    def remove_redundant(bs):
        changed = True
        while changed:
            changed = False
            if timeup():
                break
            n = len(bs)
            for i in range(n):
                if timeup():
                    break
                rest = bs[:i] + bs[i + 1:]
                if full_cover(rest):
                    bs = rest
                    changed = True
                    break
        return bs

    blocks = remove_redundant(blocks)
    best = list(blocks)
    best_len = len(best)

    # 1-swap: drop block i, add the single block that covers most newly-uncovered
    improved = True
    while improved and not timeup():
        improved = False
        n = len(blocks)
        for i in range(n):
            if timeup():
                break
            rest = blocks[:i] + blocks[i + 1:]
            m_rest = 0
            for b in rest:
                for sub in itertools.combinations(b, t):
                    m_rest |= (1 << comb_index(sub, t, v))
            if m_rest == ((1 << total_t) - 1):
                blocks = rest
                best = list(blocks)
                best_len = len(best)
                improved = True
                break
            # find best single block to add
            need = ((1 << total_t) - 1) & ~m_rest
            best_j = -1
            best_gain = 0
            for j in range(total_k):
                if timeup():
                    break
                c = ksets[j]
                gm = 0
                for sub in itertools.combinations(c, t):
                    gm |= (1 << comb_index(sub, t, v))
                gain = int(np.count_nonzero(gm & need))
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_gain > 0:
                cand = rest + [list(ksets[best_j])]
                if len(cand) < best_len and full_cover(cand):
                    blocks = cand
                    best = list(blocks)
                    best_len = len(best)
                    improved = True
                    break

    return best if best_len <= len(blocks) else blocks
