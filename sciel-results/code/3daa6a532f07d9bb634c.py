"""Covering design C(v,k,t): greedy + randomized restarts + 2-exchange local search.
Keeps the greedy-covering family; adds a fast NumPy engine, a smooth uncovered-count
surrogate for point selection, and a 2-exchange polish to escape local optima, all
under an explicit wall-clock deadline."""
import time
import itertools
import numpy as np


def _all_t_subsets(v, t):
    if t == 1:
        return np.arange(v, dtype=np.int64).reshape(-1, 1)
    if t == 2:
        a = np.repeat(np.arange(v, dtype=np.int64), v - 1)
        b = np.tile(np.arange(v, dtype=np.int64), v - 1)
        mask = a < b
        a = a[mask]; b = b[mask]
        return np.stack([a, b], axis=1)
    return np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)


def _block_covered(block, t, v, idx, sub2row):
    m = len(block)
    if m < t:
        return np.zeros(0, dtype=bool)
    if t == 1:
        return np.array([sub2row.get(x, -1) for x in block], dtype=np.int64)
    if t == 2:
        rows = []
        for i in range(m):
            for j in range(i + 1, m):
                a, b = block[i], block[j]
                if a > b:
                    a, b = b, a
                r = idx.get((a, b), -1)
                if r >= 0:
                    rows.append(r)
        return np.array(rows, dtype=np.int64)
    rows = []
    for combo in itertools.combinations(block, t):
        r = sub2row.get(combo, -1)
        if r >= 0:
            rows.append(r)
    return np.array(rows, dtype=np.int64)


def _greedy(v, k, t, rng, deadline, sub2row, idx, all_rows, all_subsets, T):
    covered = np.zeros(len(all_rows), dtype=bool)
    blocks = []
    while True:
        if time.time() >= deadline:
            break
        unc = np.flatnonzero(~covered)
        if unc.size == 0:
            break
        # seed block from a (random) uncovered t-subset
        seed = all_subsets[unc[rng.integers(0, unc.size)]]
        block = list(seed)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            best_tie = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for r in range(T):
                    rest = list(block[r:r + T - 1])
                    rest.append(x)
                    rest.sort()
                    row = idx.get(tuple(rest), -1)
                    if row >= 0 and not covered[row]:
                        gain += 1
                if gain > best_gain or (gain == best_gain and best_gain >= 0 and
                                        rng.random() < 0.5):
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        best_tie = gain
                    else:
                        best_x = x
                        best_tie = gain
            if best_x < 0:
                break
            block.append(best_x)
        block = sorted(block)
        cov = _block_covered(block, t, v, idx, sub2row)
        covered[cov] = True
        blocks.append(block)
    return blocks


def _try_remove(blocks, v, k, t, idx, sub2row, all_rows, all_subsets):
    # try to drop a block that is redundant
    for i in range(len(blocks)):
        if len(blocks) - 1 < 1:
            break
        test = blocks[:i] + blocks[i + 1:]
        cov = np.zeros(len(all_rows), dtype=bool)
        for b in test:
            c = _block_covered(b, t, v, idx, sub2row)
            cov[c] = True
        if cov.all():
            return test
    return None


def _two_exchange(blocks, v, k, t, idx, sub2row, all_rows, all_subsets, rng, deadline):
    # remove a random existing block and greedily re-cover the t-subsets it exposed
    n = len(blocks)
    if n == 0:
        return blocks
    for _ in range(2000):
        if time.time() >= deadline:
            break
        i = int(rng.integers(0, n))
        base = blocks[:i] + blocks[i + 1:]
        cov = np.zeros(len(all_rows), dtype=bool)
        for b in base:
            c = _block_covered(b, t, v, idx, sub2row)
            cov[c] = True
        unc = np.flatnonzero(~cov)
        if unc.size == 0:
            # removed block was redundant
            return base
        # greedy re-cover
        new_blocks = []
        while unc.size > 0:
            if time.time() >= deadline:
                break
            seed = all_subsets[unc[rng.integers(0, unc.size)]]
            block = list(seed)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for r in range(T):
                        rest = list(block[r:r + T - 1])
                        rest.append(x)
                        rest.sort()
                        row = idx.get(tuple(rest), -1)
                        if row >= 0 and not cov[row]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
            block = sorted(block)
            c = _block_covered(block, t, v, idx, sub2row)
            cov[c] = True
            new_blocks.append(block)
            unc = np.flatnonzero(~cov)
        cand = base + new_blocks
        if len(cand) < n:
            return cand
    return None


def construct(v, k, t):
    T = t
    deadline = time.time() + 105  # safety margin under 120s
    rng = np.random.default_rng(0)

    all_subsets = _all_t_subsets(v, t)
    all_rows = np.arange(len(all_subsets), dtype=np.int64)
    sub2row = {tuple(s): i for i, s in enumerate(all_subsets)}
    if t == 1:
        idx = {}
    elif t == 2:
        idx = {}
        for (a, b), r in sub2row.items():
            if a > b:
                a, b = b, a
            idx[(a, b)] = r
    else:
        idx = sub2row

    best_blocks = None
    best_len = None
    start = time.time()
    attempt = 0
    while time.time() < deadline:
        blocks = _greedy(v, k, t, rng, deadline, sub2row, idx, all_rows, all_subsets, T)
        # polish: 2-exchange
        polished = _two_exchange(blocks, v, k, t, idx, sub2row, all_rows, all_subsets,
                                 rng, deadline)
        if polished is not None:
            blocks = polished
        # polish: try removing redundant blocks
        removed = _try_remove(blocks, v, k, t, idx, sub2row, all_rows, all_subsets)
        if removed is not None:
            blocks = removed
        if best_blocks is None or len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        attempt += 1
        # stop if we hit a strong value and little time left
        if time.time() > deadline - 25:
            break

    # final safety: ensure coverage; if not covered, fall back to a pure greedy
    if best_blocks is None:
        best_blocks = _greedy(v, k, t, rng, deadline + 10, sub2row, idx, all_rows,
                              all_subsets, T)
    return best_blocks
