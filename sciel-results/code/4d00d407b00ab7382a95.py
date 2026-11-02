import itertools
import random
import time
import math
import sys
import numpy as np

_cache = {}

def _precompute(v, k, t):
    if (v, k, t) in _cache:
        return _cache[(v, k, t)]

    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    n_t = len(all_t_subs)
    t_sub_map = {sub: i for i, sub in enumerate(all_t_subs)}

    all_k_blocks = list(itertools.combinations(points, k))
    n_k = len(all_k_blocks)

    # For each block, the list of t-subset indices it covers (size C(k,t))
    block_to_subs = []
    for block in all_k_blocks:
        lst = [t_sub_map[ts] for ts in itertools.combinations(block, t)]
        block_to_subs.append(lst)

    # Bitset per block over the n_t t-subsets (uint64 words)
    n_words = (n_t + 63) // 64
    block_bits = np.zeros((n_k, n_words), dtype=np.uint64)
    for b in range(n_k):
        for s in block_to_subs[b]:
            block_bits[b, s >> 6] |= np.uint64(1) << (s & 63)

    _cache[(v, k, t)] = (all_k_blocks, block_to_subs, block_bits, n_t, n_k, n_words)
    return _cache[(v, k, t)]


def _greedy(all_k_blocks, block_to_subs, block_bits, n_t, n_k, n_words,
            rng, deadline, soft_beta):
    """Randomized greedy with a log-sum-exp smooth surrogate over gains."""
    # initial gain[b] = C(k,t) for all b
    gain = np.full(n_k, len(block_to_subs[0]), dtype=np.int32)
    covered = np.zeros(n_words, dtype=np.uint64)
    selected = []
    remaining = n_t
    # popcount table for 64-bit words (vectorized via numpy bit_count)

    while remaining > 0:
        if time.time() > deadline:
            break
        # smooth surrogate: soft-max over gains -> probability weights
        g = gain.astype(np.float64)
        gmax = g.max()
        # log-sum-exp style softmax with temperature 1/beta
        w = np.exp((g - gmax) * soft_beta)
        ssum = w.sum()
        if ssum <= 0:
            break
        p = w / ssum
        # sample top-p for diversity
        b_sel = rng.choice(n_k, p=p)
        selected.append(all_k_blocks[b_sel])
        # update coverage bitset
        bb = block_bits[b_sel]
        new = ~covered & bb
        # count newly covered
        remaining -= int(np.sum([bin(int(x)).count('1') for x in new]))
        covered |= bb
        # update gains: subtract newly-covered count for every block sharing them
        # For each newly covered t-subset s, decrement gain of blocks containing s.
        # Efficient: for each block b, gain[b] -= popcount(block_bits[b] & new)
        if np.any(new):
            # compute popcount of (block_bits & new) for all blocks
            inter = block_bits & new
            pc = np.zeros(n_k, dtype=np.int32)
            for widx in range(n_words):
                word = inter[:, widx]
                # popcount per row
                pc += np.array([bin(int(x)).count('1') for x in word], dtype=np.int32)
            gain -= pc
    return selected


def _vnd_1for1(all_k_blocks, block_to_subs, block_bits, n_t, n_k, n_words,
               selected, rng, deadline, max_iter=400):
    """Try to remove one selected block and replace with a better single block."""
    selected_set = set(map(tuple, selected))
    # coverage bitset of current selected
    covered = np.zeros(n_words, dtype=np.uint64)
    for blk in selected:
        covered |= block_bits[all_k_blocks.index(blk)] if False else 0
    # rebuild properly: map tuple->index
    blk_index = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    covered = np.zeros(n_words, dtype=np.uint64)
    for blk in selected:
        covered |= block_bits[blk_index[tuple(blk)]]
    cur_count = len(selected)
    improved = True
    it = 0
    while improved and it < max_iter and time.time() < deadline:
        improved = False
        it += 1
        # For each selected block, compute t-subsets uniquely covered by it
        # (i.e., covered by it and by no other selected block). If a block has
        # zero unique t-subsets, it can be dropped for free.
        n_sel = len(selected)
        for idx in range(n_sel):
            if time.time() > deadline:
                return selected, cur_count
            b_i = blk_index[tuple(selected[idx])]
            # unique = block_bits[b_i] & ~(OR of all other selected blocks)
            others = np.zeros(n_words, dtype=np.uint64)
            for j in range(n_sel):
                if j != idx:
                    others |= block_bits[blk_index[tuple(selected[j])]]
            unique = block_bits[b_i] & ~others
            uniq_cnt = int(np.sum([bin(int(x)).count('1') for x in unique]))
            if uniq_cnt == 0:
                # free removal
                del selected[idx]
                cur_count -= 1
                covered = np.zeros(n_words, dtype=np.uint64)
                for blk in selected:
                    covered |= block_bits[blk_index[tuple(blk)]]
                improved = True
                break
            else:
                # try to find a single block that covers all of `unique` and
                # reduces total count (i.e., covers unique and we drop b_i)
                # We need a block b such that unique ⊆ block_bits[b].
                # candidate must contain every t-subset in `unique`.
                # Find blocks that contain all bits of `unique`:
                #   (block_bits[b] & unique) == unique  for all words
                # Compute mask: for each word, candidate must have all unique bits.
                ok = np.ones(n_k, dtype=bool)
                for widx in range(n_words):
                    u = unique[widx]
                    if u:
                        # blocks whose word has all bits of u
                        ok &= (block_bits[:, widx] & u) == u
                # exclude currently selected blocks (we are replacing b_i)
                ok[b_i] = False
                cands = np.nonzero(ok)[0]
                if len(cands) > 0:
                    # pick one that maximizes additional coverage of currently-uncovered
                    # but since we keep count same (1 for 1), any works; pick best extra
                    # uncovered = ~covered
                    unc = ~covered
                    best_b = cands[0]
                    best_extra = -1
                    for cb in cands[:50]:
                        extra = int(np.sum([bin(int(x)).count('1') for x in (block_bits[cb] & unc)]))
                        if extra > best_extra:
                            best_extra = extra
                            best_b = cb
                    selected[idx] = all_k_blocks[best_b]
                    # recompute covered
                    covered = np.zeros(n_words, dtype=np.uint64)
                    for blk in selected:
                        covered |= block_bits[blk_index[tuple(blk)]]
                    improved = True
                    break
    return selected, cur_count


def _vnd_mforM(all_k_blocks, block_to_subs, block_bits, n_t, n_k, n_words,
               selected, rng, deadline, m=2):
    """m-for-m block swap: replace m selected blocks with m new blocks that
    cover the union of t-subsets uniquely covered by the removed ones,
    while improving total coverage or count."""
    blk_index = {tuple(b): i for i, b in enumerate(all_k_blocks)}
    n_sel = len(selected)
    if n_sel < m:
        return selected
    # coverage of current
    covered = np.zeros(n_words, dtype=np.uint64)
    for blk in selected:
        covered |= block_bits[blk_index[tuple(blk)]]
    improved = True
    it = 0
    while improved and time.time() < deadline and it < 300:
        improved = False
        it += 1
        # sample m indices
        idxs = rng.sample(range(n_sel), m)
        # union of bits of removed blocks
        union = np.zeros(n_words, dtype=np.uint64)
        for i in idxs:
            union |= block_bits[blk_index[tuple(selected[i])]]
        # t-subsets uniquely covered by these m (not by any other selected)
        others = np.zeros(n_words, dtype=np.uint64)
        for j in range(n_sel):
            if j not in idxs:
                others |= block_bits[blk_index[tuple(selected[j])]]
        unique = union & ~others
        uniq_cnt = int(np.sum([bin(int(x)).count('1') for x in unique]))
        if uniq_cnt == 0:
            # free removal of all m
            for i in sorted(idxs, reverse=True):
                del selected[i]
            improved = True
            continue
        # Find m blocks whose union covers `unique`.
        # Heuristic: greedily build a set of m blocks covering unique.
        target = unique.copy()
        chosen = []
        for _ in range(m):
            if not np.any(target):
                break
            # pick block maximizing coverage of target
            best_b = -1
            best_cov = 0
            # sample candidates for speed
            cand_iter = range(n_k)
            for b in cand_iter:
                cov = int(np.sum([bin(int(x)).count('1') for x in (block_bits[b] & target)]))
                if cov > best_cov:
                    best_cov = cov
                    best_b = b
                if time.time() > deadline:
                    break
            if best_b < 0 or best_cov == 0:
                break
            chosen.append(best_b)
            target &= ~block_bits[best_b]
            if not np.any(target):
                break
        if len(chosen) == m and not np.any(target):
            # feasible: replace
            for i, cb in zip(idxs, chosen):
                selected[i] = all_k_blocks[cb]
            improved = True
    return selected


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s

    all_k_blocks, block_to_subs, block_bits, n_t, n_k, n_words = _precompute(v, k, t)

    best_blocks = None
    best_count = float('inf')

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        rng = random.Random(1000 + attempts)
        # vary soft_beta for diversity
        soft_beta = 0.05 + 0.1 * (attempts % 5)

        # 1) Greedy construction
        blocks = _greedy(all_k_blocks, block_to_subs, block_bits, n_t, n_k,
                         n_words, rng, deadline, soft_beta)
        if not blocks:
            continue
        cur_count = len(blocks)

        # 2) VND 1-for-1
        if time.time() < deadline - 5:
            blocks, cur_count = _vnd_1for1(all_k_blocks, block_to_subs, block_bits,
                                           n_t, n_k, n_words, blocks, rng, deadline)

        # 3) VND 2-for-2
        if time.time() < deadline - 5:
            blocks = _vnd_mforM(all_k_blocks, block_to_subs, block_bits,
                                n_t, n_k, n_words, blocks, rng, deadline, m=2)
            cur_count = len(blocks)

        # 4) VND 3-for-3 (only if time permits)
        if time.time() < deadline - 3:
            blocks = _vnd_mforM(all_k_blocks, block_to_subs, block_bits,
                                n_t, n_k, n_words, blocks, rng, deadline, m=3)
            cur_count = len(blocks)

        # 5) Another 1-for-1 pass after swaps
        if time.time() < deadline - 2:
            blocks, cur_count = _vnd_1for1(all_k_blocks, block_to_subs, block_bits,
                                           n_t, n_k, n_words, blocks, rng, deadline)

        if cur_count < best_count:
            best_count = cur_count
            best_blocks = blocks

    if best_blocks is None:
        return []
    return best_blocks
