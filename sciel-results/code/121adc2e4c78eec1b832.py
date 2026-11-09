"""Greedy covering for C(19,7,4) using precomputed bitmasks + vectorized gain."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0
_SAFETY = 8.0          # seconds of margin before the 150s kill
_START = time.time()

# Precomputed structures (independent of the task parameters).
_SUB3 = []
_SUB3M = []
for v in range(19):
    for a in range(v):
        for b in range(a + 1, v):
            for c in range(b + 1, v):
                _SUB3.append((a, b, c))
                _SUB3M.append((1 << a) | (1 << b) | (1 << c))
_SUB3 = np.array(_SUB3, dtype=np.int64)          # (969, 3)
_SUB3M = np.array(_SUB3M, dtype=np.int64)        # (969,)
_BITS = np.array([1 << i for i in range(19)], dtype=np.int64)


def _gain_from_mask(uncovered, mask, k):
    """Vectorized gain of every candidate point for a partial block (as bitmask)."""
    partial = _SUB3[np.isin(_SUB3, mask)]
    if partial.shape[0] == 0:
        return np.zeros(k, dtype=np.int64)
    return np.sum((uncovered[:, None] & _BITS[None, :]) == uncovered, axis=0)


def _greedy(v, k, t, budget):
    allsubs = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    uncovered = np.ones(len(allsubs), dtype=bool)
    blocks = []
    while True:
        if time.time() > budget:
            break
        n = int(uncovered.sum())
        if n == 0:
            break
        first = allsubs[uncovered][0]
        block = set(int(x) for x in first)
        mask = 0
        for x in first:
            mask |= (1 << x)
        while len(block) < k:
            g = _gain_from_mask(uncovered, mask, k)
            best = int(np.argmax(g))
            if g[best] == 0:
                break
            block.add(best)
            mask |= (1 << best)
        block = sorted(block)
        blocks.append(block)
        for comb in itertools.combinations(block, t):
            idx = np.where((allsubs == np.array(comb)).all(axis=1))[0]
            uncovered[idx] = False
    return blocks


def _local_improve(blocks, v, k, t, budget):
    """Drop any block that is not needed (its t-subsets are all covered by others)."""
    allsubs = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n = len(allsubs)
    block_masks = []
    for b in blocks:
        m = np.zeros(n, dtype=bool)
        for comb in itertools.combinations(b, t):
            idx = np.where((allsubs == np.array(comb)).all(axis=1))[0]
            m[idx] = True
        block_masks.append(m)
    keep = list(range(len(blocks)))
    changed = True
    while changed:
        changed = False
        if time.time() > budget:
            break
        for i in range(len(keep)):
            if time.time() > budget:
                break
            idx = keep[i]
            others = np.zeros(n, dtype=bool)
            for j in keep:
                if j != idx:
                    others |= block_masks[j]
            if np.all(others[block_masks[idx]]):
                keep.remove(idx)
                changed = True
                break
    return [blocks[i] for i in keep]


def construct(v, k, t):
    global _START
    _START = time.time()
    budget = _START + _DEADLINE - _SAFETY
    blocks = _greedy(v, k, t, budget)
    if time.time() < budget:
        blocks = _local_improve(blocks, v, k, t, budget)
    return blocks
