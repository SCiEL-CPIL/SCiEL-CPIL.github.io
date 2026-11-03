"""Deadline-guarded randomized greedy covering with vectorized gains."""
import time
import random
import numpy as np
from itertools import combinations


def _key_of(combo, v):
    key = 0
    for x in combo:
        key = key * v + x
    return key


def _subsets_keys(block, t, v):
    return [_key_of(c, v) for c in combinations(block, t)]


def _build_block(v, k, t, start_block, unc, pos, contains, ext_map, rng):
    """Greedily extend start_block to size k maximizing uncovered gain."""
    block = list(start_block)
    while len(block) < k:
        # (t-1)-subset keys of current block
        keys = [_key_of(c, v) for c in combinations(block, t - 1)]
        if not keys:
            cand = [x for x in range(v) if x not in block]
            if not cand:
                return None
            block.append(cand[rng.randint(0, len(cand) - 1)])
            continue
        keyset = set(keys)
        best_x = -1
        best_g = -1
        for x in range(v):
            if x in block:
                continue
            g = 0
            for tkey in contains[x]:
                if not unc[tkey]:
                    continue
                combo = pos[tkey]
                # does combo minus x lie in block?
                ok = True
                for y in combo:
                    if y == x:
                        continue
                    if y not in block:
                        ok = False
                        break
                if ok:
                    g += 1
            if g > best_g:
                best_g = g
                best_x = x
        if best_x < 0:
            return None
        block.append(best_x)
    return block


def _greedy_once(v, k, t, m, pos, contains, rng, deadline):
    unc = np.ones(m, dtype=bool)
    blocks = []
    while True:
        if time.time() > deadline:
            return None
        if not unc.any():
            break
        idx = np.flatnonzero(unc)
        start_i = idx[rng.randint(0, len(idx) - 1)]
        start_block = [int(x) for x in pos[start_i]]
        block = _build_block(v, k, t, start_block, unc, pos, contains, None, rng)
        if block is None:
            return None
        block.sort()
        for key in _subsets_keys(block, t, v):
            unc[key] = False
        blocks.append(block)
    return blocks


def construct(v, k, t):
    start = time.time()
    deadline = start + 100.0
    rng = random.Random(20240517)

    all_t = list(combinations(range(v), t))
    m = len(all_t)
    if m == 0:
        return []
    arr = np.array(all_t, dtype=np.int64)

    # dense index per t-subset
    key = np.zeros(m, dtype=np.int64)
    for j in range(t):
        key = key * v + arr[:, j]
    order = np.argsort(key, kind="stable")
    pos = np.empty(m, dtype=np.int64)
    pos[order] = np.arange(m)

    contains = []
    for x in range(v):
        contains.append(np.flatnonzero((arr == x).any(axis=1)))

    best = None
    while time.time() < deadline:
        res = _greedy_once(v, k, t, m, pos, contains, rng, deadline)
        if res is not None and (best is None or len(res) < len(best)):
            best = res
            if len(res) <= 335:
                break
        if time.time() > deadline - 25.0:
            break

    if best is None:
        # deterministic fallback
        unc = np.ones(m, dtype=bool)
        blocks = []
        while True:
            if not unc.any():
                break
            idx = np.flatnonzero(unc)
            start_i = idx[0]
            start_block = [int(x) for x in pos[start_i]]
            block = _build_block(v, k, t, start_block, unc, pos, contains, None, rng)
            if block is None:
                # last-resort: pad arbitrarily
                block = list(start_block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
            block.sort()
            for key2 in _subsets_keys(block, t, v):
                unc[key2] = False
            blocks.append(block)
        return blocks

    return best
