"""Greedy covering design C(v,k,t) with vectorized gain and a deadline guard."""
import time
import itertools
import numpy as np

# Wall-clock safety margin (seconds) before the evaluator's hard kill.
_DEADLINE_MARGIN = 12.0
# If this many consecutive blocks add no new coverage, stop early.
_STAGNATION_LIMIT = 5


def _block_cover_mask(block, t, comb_table, index_table, v):
    """Boolean array: which t-subsets of {0..v-1} are contained in `block`."""
    n = len(block)
    if n < t:
        return np.zeros(len(comb_table), dtype=bool)
    idx = index_table[block]
    if t == 1:
        return idx
    if t == 2:
        return idx[:, None] | idx[None, :]
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    for i in range(1, n):
        mask[i] = idx[i]
        mask[:i] |= idx[i][:i]
    return comb_table[mask]


def _greedy_cover(v, k, t, deadline):
    """Greedy set cover. Returns (blocks, complete) where complete means every
    t-subset is covered."""
    comb_table = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    index_table = np.zeros((v, t), dtype=np.int64)
    for i in range(v):
        index_table[i] = np.array([j for j in range(t) if i in comb_table[:, j]], dtype=np.int64)
    uncovered = np.ones(len(comb_table), dtype=bool)
    blocks = []
    stagnant = 0
    while uncovered.any():
        if time.time() >= deadline:
            return blocks, False
        # Seed: the most-frequent point among the uncovered t-subsets.
        counts = np.bincount(np.where(uncovered, comb_table, -1).ravel(), minlength=v)
        block = [int(np.argmax(counts))]
        while len(block) < k:
            if time.time() >= deadline:
                return blocks, False
            idx = index_table[block]
            if t == 1:
                gains = idx.sum(axis=0)
            else:
                occ = idx[:, :t - 1]
                gains = np.zeros(v, dtype=np.int64)
                for j in range(t - 1):
                    gains[occ[:, j]] += 1
            cand = np.where(np.isin(np.arange(v), np.array(block))[0] >= 0)[0]
            in_block = np.zeros(v, dtype=bool)
            in_block[block] = True
            cand = np.where(~in_block)[0]
            best = int(cand[np.argmax(gains[cand])])
            block.append(best)
        block.sort()
        mask = _block_cover_mask(block, t, comb_table, index_table, v)
        newly = int(np.count_nonzero(mask & uncovered))
        blocks.append(block)
        uncovered &= ~mask
        if newly == 0:
            stagnant += 1
            if stagnant >= _STAGNATION_LIMIT:
                break
        else:
            stagnant = 0
    return blocks, bool(not uncovered.any())


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE_MARGIN
    blocks, complete = _greedy_cover(v, k, t, deadline)
    return blocks
