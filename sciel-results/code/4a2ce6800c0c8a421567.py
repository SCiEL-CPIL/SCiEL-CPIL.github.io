# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

_DEADLINE = 110.0  # safety margin under the 150s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _all_covers(mask, n):
    """True if every bit of n is covered by mask (n-bit word)."""
    return (mask | n) == n


def construct(v, k, t):
    global _DEADLINE, _START
    _START = time.time()
    _DEADLINE = _START + 110.0

    # Precompute all t-subsets and their indices.
    triples = list(itertools.combinations(range(v), t))
    ntri = len(triples)
    idx = {tr: i for i, tr in enumerate(triples)}
    full = (1 << ntri) - 1

    # Precompute, for every point, the set of triples containing it.
    pt_mask = []
    for x in range(v):
        m = 0
        for tr in triples:
            if x in tr:
                m |= 1 << idx[tr]
        pt_mask.append(m)

    # Greedy construction.
    blocks = []
    covered = 0
    while covered < ntri:
        if _elapsed() > _DEADLINE - 8:
            break
        best_gain = 0
        best_ov = -1
        best_block = None
        for blk in itertools.combinations(range(v), k):
            m = 0
            for x in blk:
                m |= pt_mask[x]
            g = bin(m & ~covered).count("1")
            if g <= best_gain:
                continue
            ov = bin(m & covered).count("1")
            if g > best_gain or (g == best_gain and ov < best_ov):
                best_gain = g
                best_ov = ov
                best_block = blk
        if best_block is None:
            break
        blocks.append(best_block)
        covered += best_gain

    # Local-search reduction.
    masks = [0] * len(blocks)
    for i, b in enumerate(blocks):
        m = 0
        for x in b:
            m |= pt_mask[x]
        masks[i] = m

    n = len(blocks)
    cov_count = [0] * ntri
    for m in masks:
        mm = m
        while mm:
            lsb = mm & -mm
            cov_count[lsb.bit_length() - 1] += 1
            mm ^= lsb

    def _try_remove(i):
        nonlocal n, covered
        mm = masks[i]
        ok = True
        while mm:
            lsb = mm & -mm
            if cov_count[lsb.bit_length() - 1] <= 1:
                ok = False
                break
            mm ^= lsb
        if not ok:
            return False
        mm = masks[i]
        while mm:
            lsb = mm & -mm
            cov_count[lsb.bit_length() - 1] -= 1
            mm ^= lsb
        del masks[i]
        del blocks[i]
        n -= 1
        return True

    changed = True
    while changed and _elapsed() < _DEADLINE - 10:
        changed = False
        i = 0
        while i < n:
            if _try_remove(i):
                changed = True
            else:
                i += 1

    # 1-for-1 block replacement.
    while _elapsed() < _DEADLINE - 15:
        progress = False
        for i in range(n):
            if _elapsed() > _DEADLINE - 15:
                break
            mi = masks[i]
            while mi:
                lsb = mi & -mi
                j = lsb.bit_length() - 1
                mi ^= lsb
                if cov_count[j] == 1:
                    # triple j is unique to block i; find a candidate covering it.
                    cand = None
                    for cand_blk in itertools.combinations(range(v), k):
                        if cand_blk in blocks:
                            continue
                        m = 0
                        for x in cand_blk:
                            m |= pt_mask[x]
                        if not (m & lsb):
                            continue
                        net = bin(m & ~covered).count("1") - bin(m & covered).count("1")
                        if net > 0:
                            cand = (cand_blk, m, net)
                            break
                    if cand is None:
                        continue
                    cand_blk, m, net = cand
                    # remove block i
                    mm = mi = masks[i]
                    mm = masks[i]
                    while mm:
                        lsb2 = mm & -mm
                        cov_count[lsb2.bit_length() - 1] -= 1
                        mm ^= lsb2
                    # add candidate
                    mm = m
                    while mm:
                        lsb2 = mm & -mm
                        cov_count[lsb2.bit_length() - 1] += 1
                        mm ^= lsb2
                    blocks[i] = list(cand_blk)
                    masks[i] = m
                    progress = True
                    break
            if progress:
                break
        if not progress:
            break

    return [sorted(b) for b in blocks]
# EVOLVE-BLOCK-END
