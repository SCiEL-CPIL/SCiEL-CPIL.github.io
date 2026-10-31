"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Randomized Parallel Batch Construction (diverse candidate pools + fast
bitmask greedy over each pool, keep best) combined with aggressive bitmask
local search (removal + swap+remove) using numpy coverage counts.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 118.0  # safety margin under 150s wall-clock
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    full_mask = (1 << n_t) - 1
    return all_t, t_index, t_index, full_mask


def _coverage_mask(ks, t_index, t):
    mask = 0
    for ts in itertools.combinations(ks, t):
        mask |= (1 << t_index[ts])
    return mask


def _greedy_over_pool(pool, t_index, full_mask, t, budget):
    """Fast bitmask greedy over a candidate pool (list of k-subsets)."""
    n = len(pool)
    cov = [0] * n
    for i in range(n):
        cov[i] = _coverage_mask(pool[i], t_index, t)
    uncovered = full_mask
    blocks = []
    for _ in range(n):
        if _elapsed() > budget:
            break
        best_i = -1
        best_gain = 0
        for i in range(n):
            g = bin(cov[i] & uncovered).count('1')
            if g > best_gain:
                best_gain = g
                best_i = i
        if best_i < 0 or best_gain == 0:
            break
        blocks.append(best_i)
        uncovered &= ~cov[best_i]
        if uncovered == 0:
            break
    return blocks, cov, uncovered


def _local_search(blocks, cov, full_mask, rng, time_budget):
    """Removal + swap+remove local search using numpy coverage counts."""
    n_t = full_mask.bit_length()
    # Build numpy coverage matrix for the current blocks
    mat = np.zeros((len(blocks), n_t), dtype=np.uint8)
    for i, b in enumerate(blocks):
        m = cov[b]
        # expand bitmask to vector
        mm = m
        j = 0
        while mm:
            if mm & 1:
                mat[i, j] = 1
            mm >>= 1
            j += 1
    counts = mat.sum(axis=0).astype(np.int32)

    def counts_rebuild():
        mat[:] = 0
        for i, b in enumerate(blocks):
            m = cov[b]
            mm = m
            j = 0
            while mm:
                if mm & 1:
                    mat[i, j] = 1
                mm >>= 1
                j += 1
        counts[:] = mat.sum(axis=0)

    def try_remove():
        # Remove any block that covers no t-subset with count==1
        removed_any = False
        for i in range(len(blocks)):
            if _elapsed() > time_budget:
                break
            row = mat[i]
            # block removable if no position where row==1 and counts==1
            if not np.any((row == 1) & (counts == 1)):
                # remove
                counts -= row
                mat = np.delete(mat, i, axis=0)
                blocks.pop(i)
                removed_any = True
                # restart scan from 0 since counts changed
                return True, mat
        return removed_any, mat

    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        # Removal pass until stable
        while _elapsed() < time_budget:
            removed_any, mat = try_remove()
            if not removed_any:
                break
            improved = True
        # Swap pass
        for _ in range(40):
            if _elapsed() > time_budget:
                break
            if len(blocks) == 0:
                break
            i = rng.randrange(len(blocks))
            # pick a random new block index not currently present
            new_b = rng.randrange(len(cov))
            if new_b in blocks:
                continue
            old_b = blocks[i]
            # tentatively swap
            counts -= mat[i]
            # add coverage of new block
            nm = cov[new_b]
            mm = nm
            j = 0
            addvec = np.zeros(n_t, dtype=np.uint8)
            while mm:
                if mm & 1:
                    addvec[j] = 1
                mm >>= 1
                j += 1
            counts += addvec
            blocks[i] = new_b
            # try removals
            mat = np.zeros((len(blocks), n_t), dtype=np.uint8)
            for ii, b in enumerate(blocks):
                m = cov[b]
                mm2 = m
                jj = 0
                while mm2:
                    if mm2 & 1:
                        mat[ii, jj] = 1
                    mm2 >>= 1
                    jj += 1
            counts[:] = mat.sum(axis=0)
            removed = False
            while _elapsed() < time_budget:
                removed_any, mat = try_remove()
                if not removed_any:
                    break
                removed = True
                improved = True
            if not removed:
                # revert swap
                blocks[i] = old_b
                mat = np.zeros((len(blocks), n_t), dtype=np.uint8)
                for ii, b in enumerate(blocks):
                    m = cov[b]
                    mm2 = m
                    jj = 0
                    while mm2:
                        if mm2 & 1:
                            mat[ii, jj] = 1
                        mm2 >>= 1
                        jj += 1
                counts[:] = mat.sum(axis=0)
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(20240517)

    all_t, t_index, _, full_mask = _precompute(v, k, t)

    # Build a diverse candidate pool: random k-subsets (plus all if small)
    # For v=19,k=7 there are 50388 k-subsets; use a random sample pool.
    all_ks = list(itertools.combinations(range(v), k))
    n_ks = len(all_ks)
    pool_size = min(n_ks, 6000)
    pool = [all_ks[i] for i in rng.sample(range(n_ks), pool_size)]

    # Batch construction: run greedy over the pool, plus a few perturbed pools
    best_blocks = None
    best_cov = None
    n_batches = 4
    per_batch = (_DEADLINE - 30) / n_batches
    for b in range(n_batches):
        budget = _DEADLINE - 30 - b * 0  # use global; per-batch cap below
        if _elapsed() > _DEADLINE - 30:
            break
        blocks_idx, cov, uncovered = _greedy_over_pool(pool, t_index, full_mask, t, _DEADLINE - 30)
        # repair if not fully covered
        if uncovered != 0:
            # add greedy from full set to repair
            while uncovered != 0 and _elapsed() < _DEADLINE - 30:
                best_i = -1
                best_g = 0
                for i in range(n_ks):
                    g = bin(_coverage_mask(all_ks[i], t_index, t) & uncovered).count('1')
                    if g > best_g:
                        best_g = g
                        best_i = i
                if best_i < 0:
                    break
                blocks_idx.append(best_i)
                uncovered &= ~_coverage_mask(all_ks[best_i], t_index, t)
        if best_blocks is None or len(blocks_idx) < len(best_blocks):
            best_blocks = blocks_idx
            best_cov = cov
        # perturb pool for next batch: resample
        pool = [all_ks[i] for i in rng.sample(range(n_ks), pool_size)]

    if best_blocks is None:
        # fallback: full greedy
        blocks_idx, cov, uncovered = _greedy_over_pool(all_ks, t_index, full_mask, t, _DEADLINE - 30)
        best_blocks = blocks_idx
        best_cov = cov

    # Local search
    time_budget = _DEADLINE - 3
    best_blocks = _local_search(best_blocks, best_cov, full_mask, rng, time_budget)

    # Safety repair
    covered = 0
    for b in best_blocks:
        covered |= best_cov[b]
    if covered != full_mask:
        missing = full_mask & ~covered
        while missing != 0 and _elapsed() < _DEADLINE - 1:
            best_i = -1
            best_g = 0
            for i in range(n_ks):
                g = bin(_coverage_mask(all_ks[i], t_index, t) & missing).count('1')
                if g > best_g:
                    best_g = g
                    best_i = i
            if best_i < 0:
                break
            best_blocks.append(best_i)
            missing &= ~_coverage_mask(all_ks[best_i], t_index, t)

    # Convert to lists, dedup
    result = [list(all_ks[b]) for b in best_blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
