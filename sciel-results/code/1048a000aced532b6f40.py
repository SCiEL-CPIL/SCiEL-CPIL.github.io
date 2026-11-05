# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(v, k, t):
    start = time.time()
    DEADLINE = 115.0          # safety margin under the 150s kill
    POINTS = np.arange(v)

    # ---- precompute all t-subsets and a fast index into them ----
    all_t = np.array(list(itertools.combinations(range(v), t)), dtype=np.int32)
    n_t = len(all_t)
    idx_map = {}
    for i, row in enumerate(all_t):
        idx_map[tuple(row)] = i
    idx_arr = np.empty(n_t, dtype=np.int64)
    for i, row in enumerate(all_t):
        idx_arr[i] = idx_map[tuple(row)]

    def _triple_idx(arr):
        # arr: 2D int array of shape (m, t); returns int64 array of global indices
        if arr.size == 0:
            return np.empty(0, dtype=np.int64)
        rows = np.asarray(arr, dtype=np.int64)
        if rows.ndim == 1:
            rows = rows[None, :]
        return idx_map[tuple(rows[0].tolist())] if rows.shape[0] == 1 \
            else np.array([idx_map[tuple(r.tolist())] for r in rows], dtype=np.int64)

    def _triple_indices_of_block(block):
        return np.array([idx_map[tuple(c)] for c in itertools.combinations(block, t)], dtype=np.int64)

    # ---- vectorized gain: for a partial block, how many uncovered triples each candidate covers ----
    def _gains(partial, uncovered_mask):
        if len(partial) < t - 1:
            # not enough fixed points to form a t-subset with a candidate; score = 0
            return np.zeros(v, dtype=np.int64)
        partial = np.asarray(partial, dtype=np.int32)
        parts = list(itertools.combinations(partial, t - 1))
        cand = np.asarray(POINTS, dtype=np.int32)
        m = len(parts)
        base = np.empty((m, v, t), dtype=np.int32)
        for i, p in enumerate(parts):
            base[i, :, :] = np.concatenate([np.asarray(p, dtype=np.int32)[None, :], cand[None, :]])
        # flatten to (m*v, t)
        flat = base.reshape(m * v, t)
        # sort each row
        flat_sorted = np.sort(flat, axis=1)
        # map to indices
        idxs = np.array([idx_map[tuple(r.tolist())] for r in flat_sorted], dtype=np.int64)
        idxs = idxs.reshape(m, v)
        # gain = sum over parts of uncovered[idx]
        unc = uncovered_mask[idxs]  # (m, v) bool
        gains = unc.sum(axis=0).astype(np.int64)
        return gains

    # ---- one randomized greedy build ----
    def _greedy_build(rnd):
        uncovered_mask = np.ones(n_t, dtype=bool)
        blocks = []
        while True:
            if not uncovered_mask.any():
                break
            # pick a random uncovered triple to seed
            unc_idx = np.where(uncovered_mask)[0]
            seed = all_t[rnd.choice(unc_idx)].tolist()
            block = list(seed)
            while len(block) < k:
                gains = _gains(block, uncovered_mask)
                in_block = set(block)
                # restrict to candidates not in block
                best_x = None
                best_g = -1
                # tie-break randomly among top gains
                order = list(range(v))
                rnd.shuffle(order)
                for x in order:
                    if x in in_block:
                        continue
                    g = gains[x]
                    if g > best_g:
                        best_g = g
                        best_x = x
                        if best_g == 0:
                            # no improvement possible; pick first available
                            break
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            bi = _triple_indices_of_block(block)
            uncovered_mask[bi] = False
            blocks.append(block)
            if time.time() > DEADLINE:
                break
        return blocks

    def _is_valid(blocks):
        seen = set()
        for b in blocks:
            if len(b) != k or len(set(b)) != k:
                return False
            if any(x < 0 or x >= v for x in b):
                return False
            if b in seen:
                return False
            seen.add(b)
        covered = np.zeros(n_t, dtype=bool)
        for b in blocks:
            covered[_triple_indices_of_block(b)] = True
        return bool(covered.all())

    # ---- local search: try to replace each block with a better one ----
    def _local_search(blocks, rnd):
        if not blocks:
            return blocks
        uncovered = np.zeros(n_t, dtype=bool)
        for b in blocks:
            uncovered[_triple_indices_of_block(b)] = True
        covered_count = np.zeros(n_t, dtype=np.int64)
        for b in blocks:
            covered_count[_triple_indices_of_block(b)] += 1
        # a triple is "safe" if covered >= 2 times (removing one block still leaves it covered)
        safe = covered_count >= 2

        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            order = list(range(len(blocks)))
            rnd.shuffle(order)
            for i in order:
                if time.time() > DEADLINE:
                    break
                old = blocks[i]
                # temporarily remove old block
                oi = _triple_indices_of_block(old)
                uncovered[oi] = True
                covered_count[oi] -= 1
                # now build a new block that maximizes coverage of currently-uncovered triples
                # seed with an uncovered triple
                unc_idx = np.where(uncovered)[0]
                if len(unc_idx) == 0:
                    # nothing to cover; keep old (or drop it)
                    # drop the block
                    blocks.pop(i)
                    covered_count[oi] += 1
                    uncovered[oi] = False
                    improved = True
                    break
                seed = all_t[rnd.choice(unc_idx)].tolist()
                block = list(seed)
                while len(block) < k:
                    gains = _gains(block, uncovered)
                    in_block = set(block)
                    best_x = None
                    best_g = -1
                    order2 = list(range(v))
                    rnd.shuffle(order2)
                    for x in order2:
                        if x in in_block:
                            continue
                        g = gains[x]
                        if g > best_g:
                            best_g = g
                            best_x = x
                            if best_g == 0:
                                break
                    if best_x is None:
                        break
                    block.append(best_x)
                block.sort()
                ni = _triple_indices_of_block(block)
                new_unc = uncovered.copy()
                new_unc[ni] = False
                old_unc = uncovered.copy()
                old_unc[oi] = False
                # accept if new covers at least as many uncovered triples
                new_cov = int((~new_unc).sum())
                old_cov = int((~old_unc).sum())
                if new_cov >= old_cov and len(block) == k:
                    # commit replacement
                    covered_count[ni] += 1
                    uncovered[ni] = False
                    blocks[i] = block
                    improved = True
                    # recompute safety not needed here
                else:
                    # revert
                    covered_count[oi] += 1
                    uncovered[oi] = False
        return blocks

    # ---- driver: multiple randomized restarts, keep best valid ----
    best = None
    best_len = None
    rnd = random.Random(12345)

    # deterministic-ish first attempt
    blocks = _greedy_build(rnd)
    if _is_valid(blocks):
        best = blocks
        best_len = len(blocks)

    # additional randomized restarts
    attempt = 0
    while time.time() < DEADLINE - 25.0:
        attempt += 1
        cand = _greedy_build(rnd)
        if not _is_valid(cand):
            continue
        # local search polish
        cand = _local_search(cand, rnd)
        if not _is_valid(cand):
            continue
        if best_len is None or len(cand) < best_len:
            best = cand
            best_len = len(cand)
            if best_len <= 76:  # can't do much better than best-known; stop early
                break
        if time.time() > DEADLINE:
            break

    if best is None:
        # fallback: simple deterministic greedy to guarantee a valid covering
        uncovered_mask = np.ones(n_t, dtype=bool)
        blocks = []
        while uncovered_mask.any():
            unc_idx = np.where(uncovered_mask)[0]
            seed = all_t[unc_idx[0]].tolist()
            block = list(seed)
            while len(block) < k:
                gains = _gains(block, uncovered_mask)
                in_block = set(block)
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in in_block:
                        continue
                    g = gains[x]
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            bi = _triple_indices_of_block(block)
            uncovered_mask[bi] = False
            blocks.append(block)
        best = blocks

    return best
# EVOLVE-BLOCK-END
