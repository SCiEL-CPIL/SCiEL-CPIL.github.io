"""Covering design C(19,8,5) - vectorized full-block greedy + two-for-one local search.

Structural change (vs. parent): instead of building each block point-by-point, we
precompute every 8-block and the t-subsets it covers, then run a numpy-vectorized
greedy that repeatedly picks the block of maximum marginal gain (bincount gain
updates). This is far faster per step, so many randomized restarts fit in the
wall-clock budget.

Refinement machinery: after each greedy pass we (a) strip redundant blocks and
(b) run a 'two-for-one' local search that replaces two blocks by a single block
whenever one block covers all t-subsets that were uniquely covered by the pair.
Both phases strictly reduce the block count, which is the structural refinement
the parent lacked.
"""
import itertools
import random
import time
import numpy as np


def _C(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    res = 1
    for i in range(1, r + 1):
        res = res * (n - i + 1) // i
    return res


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before the 150s kill

    # ---- Fallback for non-target parameters ----
    if not (v == 19 and k == 8 and t == 5):
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            target = min(uncovered)
            tset = set(target)
            remaining = [x for x in range(v) if x not in tset]
            need = k - t
            best_block, best_gain = None, -1
            for comb in itertools.combinations(remaining, need):
                block = tuple(sorted(target + comb))
                gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                if gain > best_gain:
                    best_gain, best_block = gain, block
            if best_block is None:
                break
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            if time.time() > deadline:
                break
        return blocks

    # ---- Precompute full-block universe and coverage (structural core) ----
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)

    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    block_ts_lists = []
    ts_blocks_lists = [[] for _ in range(n_t)]
    for j, block in enumerate(all_blocks):
        covered = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_ts_lists.append(covered)
        for idx in covered:
            ts_blocks_lists[idx].append(j)

    def run_greedy(seed):
        """Vectorized full-block greedy with random tie-breaking."""
        rng = random.Random(seed)
        gain = np.full(n_blocks, _C(k, t), dtype=np.int32)
        is_uncovered = np.ones(n_t, dtype=bool)
        is_selected = np.zeros(n_blocks, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 8.0:
                return None
            max_gain = int(gain.max())
            if max_gain <= 0:
                break
            cands = np.where((gain == max_gain) & (~is_selected))[0]
            if cands.size == 0:
                break
            b = int(cands[rng.randrange(cands.size)])
            sol.append(list(all_blocks[b]))
            is_selected[b] = True
            gain[b] = -1
            newly = [idx for idx in block_ts_lists[b] if is_uncovered[idx]]
            if not newly:
                continue
            for idx in newly:
                is_uncovered[idx] = False
            dec = []
            for idx in newly:
                dec.extend(ts_blocks_lists[idx])
            if dec:
                gain -= np.bincount(dec, minlength=n_blocks)
                gain[is_selected] = -1000
        return sol

    def strip_redundant(blocks):
        """Remove any block whose every t-subset is also covered by another block."""
        if not blocks:
            return blocks
        block_ts = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            block_ts.append(block_ts_lists[idx] if idx >= 0 else [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)])
        cover = np.zeros(n_t, dtype=np.int32)
        for lst in block_ts:
            for x in lst:
                cover[x] += 1
        changed = True
        while changed:
            if time.time() > deadline - 4.0:
                return blocks
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for x in block_ts[i]:
                    if cover[x] == 1:
                        redundant = False
                        break
                if redundant:
                    for x in block_ts[i]:
                        cover[x] -= 1
                    blocks.pop(i)
                    block_ts.pop(i)
                    changed = True
                    break
        return blocks

    def two_for_one(blocks):
        """Replace two blocks by one whenever a single block covers all t-subsets
        that the pair uniquely covered. Strictly reduces the block count."""
        if len(blocks) < 2:
            return blocks
        block_ts = []
        for b in blocks:
            idx = block_to_idx.get(tuple(b), -1)
            block_ts.append(block_ts_lists[idx] if idx >= 0 else [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)])
        cover = np.zeros(n_t, dtype=np.int32)
        for lst in block_ts:
            for x in lst:
                cover[x] += 1
        current_set = {tuple(b) for b in blocks}
        improved = True
        while improved:
            if time.time() > deadline - 6.0:
                return blocks
            improved = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 6.0:
                    return blocks
                uniq_i = [x for x in block_ts[i] if cover[x] == 1]
                if len(uniq_i) > 12:
                    continue
                for j in range(i + 1, n):
                    uniq_j = [x for x in block_ts[j] if cover[x] == 1]
                    if len(uniq_j) > 12:
                        continue
                    union = set(uniq_i) | set(uniq_j)
                    if len(union) > 14:
                        continue
                    first = next(iter(union))
                    found = -1
                    for cand in ts_blocks_lists[first]:
                        cand_set = set(block_ts_lists[cand])
                        if all(x in cand_set for x in union):
                            found = cand
                            break
                    if found >= 0:
                        new_tuple = all_blocks[found]
                        if new_tuple in current_set:
                            continue  # would create a duplicate block
                        new_block = list(new_tuple)
                        blocks = blocks[:i] + blocks[i + 1:j] + blocks[j + 1:] + [new_block]
                        block_ts = []
                        for b in blocks:
                            idx = block_to_idx.get(tuple(b), -1)
                            block_ts.append(block_ts_lists[idx] if idx >= 0 else [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)])
                        cover = np.zeros(n_t, dtype=np.int32)
                        for lst in block_ts:
                            for x in lst:
                                cover[x] += 1
                        current_set = {tuple(b) for b in blocks}
                        improved = True
                        break
                if improved:
                    break
        return blocks

    best_blocks = None
    best_len = float('inf')

    # First deterministic pass as a safety net
    sol = run_greedy(0)
    if sol is not None:
        sol = strip_redundant(sol)
        sol = two_for_one(sol)
        best_blocks, best_len = sol, len(sol)

    # Many randomized restarts (the per-step speedup buys these)
    seed = 1
    while time.time() < deadline - 10.0:
        sol = run_greedy(seed)
        seed += 1
        if sol is None:
            continue
        sol = strip_redundant(sol)
        sol = two_for_one(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = sol

    if best_blocks is None:
        best_blocks = run_greedy(0) or []
    return [list(b) for b in best_blocks]
