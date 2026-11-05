# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _build_incidence(v, k, t):
    """Dense 0/1 incidence matrix: rows = points, cols = t-subsets (lex order)."""
    subs = list(itertools.combinations(range(v), t))
    m = len(subs)
    idx = np.empty(m, dtype=object)
    for i, c in enumerate(subs):
        idx[i] = c
    inc = np.zeros((v, m), dtype=np.int8)
    for p in range(v):
        others = [q for q in range(v) if q != p]
        for c in itertools.combinations(others, t - 1):
            s = sorted(c + (p,))
            j = int(np.searchsorted(idx, s, side='left'))
            inc[p, j] = 1
    first_elem = np.array([c[0] for c in subs], dtype=np.int64)
    return inc, idx, first_elem


def _block_mask(block, t, idx, m):
    mask = np.zeros(m, dtype=bool)
    for c in itertools.combinations(block, t):
        j = int(np.searchsorted(idx, c, side='left'))
        mask[j] = True
    return mask


def _greedy(inc, v, k, t, idx, first_elem, m, rng, deadline):
    cov = np.zeros(m, dtype=np.int32)
    blocks = []
    while True:
        if time.time() > deadline:
            break
        unc = (cov == 0)
        if not unc.any():
            break
        first = int(np.argmax(unc))
        block = [int(first_elem[first])]
        chosen = np.zeros(v, dtype=bool)
        chosen[block[0]] = True
        while len(block) < k:
            if time.time() > deadline:
                break
            cand = np.where(~chosen)[0]
            g = inc[cand].astype(np.int64) @ unc
            gmax = g.max()
            best = cand[g == gmax]
            if best.size > 1:
                p = int(best[rng.integers(0, best.size)])
            else:
                p = int(best[0])
            block.append(p)
            chosen[p] = True
        block.sort()
        blocks.append(tuple(block))
        mask = _block_mask(block, t, idx, m)
        cov[mask] += 1
    return blocks


def _remove_redundant(blocks, v, k, t, idx, m, deadline):
    cov = np.zeros(m, dtype=np.int32)
    masks = []
    for b in blocks:
        mask = _block_mask(b, t, idx, m)
        cov[mask] += 1
        masks.append(mask)
    keep = [True] * len(blocks)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if not keep[i]:
                continue
            if time.time() > deadline:
                break
            removable = True
            for c in itertools.combinations(blocks[i], t):
                j = int(np.searchsorted(idx, c, side='left'))
                if cov[j] < 2:
                    removable = False
                    break
            if removable:
                cov[masks[i]] -= 1
                keep[i] = False
                changed = True
    return [blocks[i] for i in range(len(blocks)) if keep[i]]


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0
    inc, idx, first_elem = _build_incidence(v, k, t)
    m = inc.shape[1]
    rng = np.random.default_rng(12345)
    best = None
    best_len = None
    while time.time() < deadline:
        blocks = _greedy(inc, v, k, t, idx, first_elem, m, rng, deadline)
        if not blocks:
            break
        blocks = _remove_redundant(blocks, v, k, t, idx, m, deadline)
        if best_len is None or len(blocks) < best_len:
            best_len = len(blocks)
            best = blocks
        if best_len <= 131:
            break
    if best is None:
        blocks = _greedy(inc, v, k, t, idx, first_elem, m, rng, deadline)
        best = blocks
    return [list(b) for b in best]
# EVOLVE-BLOCK-END
