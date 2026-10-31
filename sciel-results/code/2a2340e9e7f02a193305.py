import itertools
import time
import random
import numpy as np

def _elapsed(start):
    return time.time() - start

def _bitsets_of_blocks(v, k, t, order):
    n = int(math_comb(v, t))
    nbits = (n + 63) // 64
    m = len(order)
    bits = np.zeros((m, nbits), dtype=np.uint64)
    for bi, blk in enumerate(order):
        for sub in itertools.combinations(blk, t):
            idx = _comb_index(v, t, sub)
            bits[bi, idx >> 6] |= np.uint64(1) << (idx & 63)
    return bits

def _greedy(bits, order, budget):
    nbits = bits.shape[1]
    mask = np.ones(nbits, dtype=np.uint64)
    selected = []
    for step in range(len(order)):
        if _elapsed(budget) > 0:
            break
        gains = np.bitwise_and(bits, mask).sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        selected.append(best)
        mask &= ~bits[best]
        if not mask.any():
            break
    return selected

def _remove_redundant(bits, sel, budget):
    if not sel:
        return sel
    cover = np.zeros(bits.shape[1], dtype=np.int32)
    for bi in sel:
        cover += bits[bi].astype(np.int32)
    keep = [True] * len(sel)
    changed = True
    while changed:
        changed = False
        if _elapsed(budget) > 0:
            break
        for pos in range(len(sel)):
            if not keep[pos]:
                continue
            bi = sel[pos]
            if not np.any(cover[bits[bi] != 0] < 2):
                for j in np.nonzero(bits[bi])[0]:
                    cover[j] -= 1
                keep[pos] = False
                changed = True
    return [sel[i] for i in range(len(sel)) if keep[i]]

def _try_swaps(bits, sel, budget):
    if not sel:
        return sel
    cover = np.zeros(bits.shape[1], dtype=np.int32)
    for bi in sel:
        cover += bits[bi].astype(np.int32)
    in_sel = np.zeros(len(bits), dtype=bool)
    for bi in sel:
        in_sel[bi] = True
    sel_set = set(sel)
    improved = True
    while improved:
        improved = False
        if _elapsed(budget) > 0:
            break
        for pos in range(len(sel)):
            bi = sel[pos]
            if _elapsed(budget) > 0:
                break
            if np.any(cover[bits[bi] != 0] < 2):
                continue
            cand = np.where(~in_sel)[0]
            if len(cand) == 0:
                continue
            b = bits[cand]
            gains = (b & (cover < 2).view(np.uint64)).sum(axis=1)
            best = int(np.argmax(gains))
            if gains[best] == 0:
                continue
            newbi = int(cand[best])
            for j in np.nonzero(bits[bi])[0]:
                cover[j] -= 1
            for j in np.nonzero(bits[newbi])[0]:
                cover[j] += 1
            sel[pos] = newbi
            in_sel[bi] = False
            in_sel[newbi] = True
            sel_set.discard(bi)
            sel_set.add(newbi)
            improved = True
    return sel

def _try_two_for_one(bits, sel, budget):
    if len(sel) < 2:
        return sel
    cover = np.zeros(bits.shape[1], dtype=np.int32)
    for bi in sel:
        cover += bits[bi].astype(np.int32)
    sel_set = set(sel)
    improved = True
    while improved:
        improved = False
        if _elapsed(budget) > 0:
            break
        for pos in range(len(sel)):
            a = sel[pos]
            if _elapsed(budget) > 0:
                break
            if np.any(cover[bits[a] != 0] < 2):
                continue
            for q in range(len(sel)):
                if q == pos:
                    continue
                b = sel[q]
                if np.any(cover[bits[b] != 0] < 2):
                    continue
                union = (cover < 2) & ((bits[a] != 0) | (bits[b] != 0))
                if not union.any():
                    continue
                need = int(union.sum())
                cand = [c for c in range(len(bits)) if c not in sel_set]
                if not cand:
                    continue
                bmat = bits[cand]
                gains = (bmat & union.view(np.uint64)).sum(axis=1)
                best = int(np.argmax(gains))
                if gains[best] < need:
                    continue
                c = int(cand[best])
                for j in np.nonzero(bits[a])[0]:
                    cover[j] -= 1
                for j in np.nonzero(bits[b])[0]:
                    cover[j] -= 1
                for j in np.nonzero(bits[c])[0]:
                    cover[j] += 1
                new_sel = [x for i, x in enumerate(sel) if i != pos and i != q] + [c]
                sel = new_sel
                sel_set = set(sel)
                improved = True
                break
            if improved:
                break
    return sel

def _is_valid(bits, sel, total):
    if not sel:
        return total == 0
    covered = np.zeros(bits.shape[1], dtype=np.uint64)
    for bi in sel:
        covered |= bits[bi]
    return bool(np.all(covered != 0))

def math_comb(n, r):
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - r + i) // i
    return res

def _comb_index(n, k, combo):
    # rank of k-combination in colex (co-lexicographic) order
    rank = 0
    for j in range(k):
        rank += math_comb(combo[j], j + 1)
    return rank - 1

def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # hard safety margin (kill at 150s)
    total = math_comb(v, t)
    base_order = list(itertools.combinations(range(v), k))
    bits = _bitsets_of_blocks(v, k, t, base_order)

    def budget_left():
        return max(0.0, deadline - _elapsed(start) - 1.0)

    def full_polish(sel):
        sel = _remove_redundant(bits, sel, budget_left)
        sel = _try_swaps(bits, sel, budget_left)
        sel = _try_two_for_one(bits, sel, budget_left)
        sel = _remove_redundant(bits, sel, budget_left)
        sel = _try_swaps(bits, sel, budget_left)
        return sel

    # Base greedy in natural order
    base_sel = _greedy(bits, base_order, budget_left)
    base_sel = full_polish(base_sel)
    best_sel = list(base_sel)
    best_len = len(best_sel)
    if not _is_valid(bits, best_sel, total):
        # should not happen, but guard
        best_sel = list(range(len(bits)))

    # Multi-restart with random block order if base greedy was fast
    base_greedy_time = _elapsed(start)
    if base_greedy_time < 5.0 and budget_left() > 20.0:
        rng = random.Random(12345)
        attempts = 0
        max_attempts = 30
        while budget_left() > 10.0 and attempts < max_attempts:
            attempts += 1
            order = list(range(len(base_order)))
            rng.shuffle(order)
            sel = _greedy(bits, [base_order[i] for i in order], budget_left)
            if len(sel) >= best_len:
                continue
            sel = full_polish(sel)
            if len(sel) < best_len and _is_valid(bits, sel, total):
                best_sel = sel
                best_len = len(sel)

    # Final safety: ensure validity
    if not _is_valid(bits, best_sel, total):
        # Fallback: full greedy then polish
        sel = _greedy(bits, base_order, budget_left)
        sel = full_polish(sel)
        if _is_valid(bits, sel, total):
            best_sel = sel

    return [list(base_order[i]) for i in best_sel]
