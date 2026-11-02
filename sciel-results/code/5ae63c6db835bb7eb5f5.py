"""Covering design C(19,8,5) - Incidence-structured Greedy with Bitmask Exchange Search."""
import itertools
import time
import random
import sys
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 150s kill

    # --- Precomputation ---
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_to_ts[b] = list of t-subset indices covered by block b
    # ts_to_blocks[ts] = list of block indices covering t-subset ts
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, block in enumerate(all_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(b_idx)
        block_to_ts.append(ts_list)

    # Precompute bitmask for each block (for fast gap computation in exchange)
    block_masks = [0] * n_blocks
    for b_idx in range(n_blocks):
        mask = 0
        for ts_idx in block_to_ts[b_idx]:
            mask |= (1 << ts_idx)
        block_masks[b_idx] = mask

    # --- Greedy construction with numpy gain array ---
    def run_greedy(rng, deterministic=False):
        gain = np.full(n_blocks, len(block_to_ts[0]), dtype=np.int32)
        is_selected = np.zeros(n_blocks, dtype=bool)
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            # Find max gain among unselected
            if deterministic:
                # Find first unselected with max gain
                max_g = -1
                best_b = -1
                for b in range(n_blocks):
                    if not is_selected[b] and gain[b] > max_g:
                        max_g = gain[b]
                        best_b = b
                if best_b < 0 or max_g <= 0:
                    break
                b_idx = best_b
            else:
                # Find all with max gain among unselected
                unselected_mask = ~is_selected
                if not unselected_mask.any():
                    break
                max_g = gain[unselected_mask].max()
                if max_g <= 0:
                    break
                candidates = np.where(unselected_mask & (gain == max_g))[0]
                b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])

            selected.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Update gains for blocks sharing newly-covered t-subsets
            for ts_idx in block_to_ts[b_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    for nb in ts_to_blocks[ts_idx]:
                        if not is_selected[nb]:
                            gain[nb] -= 1

        return selected

    # --- Polish: remove redundant blocks ---
    def polish(block_indices):
        if not block_indices:
            return block_indices
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        i = len(block_indices) - 1
        while i >= 0:
            if time.time() > deadline - 1.0:
                break
            b = block_indices[i]
            redundant = True
            for ts_idx in block_to_ts[b]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in block_to_ts[b]:
                    cover_count[ts_idx] -= 1
                block_indices.pop(i)
            else:
                i -= 1
        return block_indices

    # --- Exchange local search using bitmask gap computation ---
    def exchange_search(block_indices, rng):
        """Multi-level exchange: 1-exchange, 2-exchange, 3-exchange."""
        if not block_indices:
            return block_indices

        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        selected_set = set(block_indices)

        # 1-exchange: remove one block, find a single replacement covering the gap
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            n_sel = len(block_indices)
            if n_sel < 2:
                break
            n_tries = min(300, n_sel)
            for _ in range(n_tries):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randrange(n_sel)
                b = block_indices[i]
                # Compute gap: t-subsets uniquely covered by b
                gap = 0
                for ts_idx in block_to_ts[b]:
                    if cover_count[ts_idx] == 1:
                        gap |= (1 << ts_idx)
                if gap == 0:
                    # Redundant, just remove
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    block_indices.pop(i)
                    selected_set.discard(b)
                    improved = True
                    break
                # Find candidate that covers gap using ts_to_blocks
                # Intersect ts_to_blocks for all gap t-subsets
                gap_bits = gap
                if gap_bits.bit_count() > 25:
                    continue
                # Collect gap t-subset indices
                gap_ts_list = []
                m = gap
                while m:
                    idx = (m & -m).bit_length() - 1
                    gap_ts_list.append(idx)
                    m &= m - 1
                if not gap_ts_list:
                    continue
                # Intersect candidates
                candidates = list(ts_to_blocks[gap_ts_list[0]])
                for ts_idx in gap_ts_list[1:]:
                    if not candidates:
                        break
                    candidates = [c for c in candidates if c in ts_to_blocks[ts_idx]]
                if not candidates:
                    continue
                # Pick a candidate not already selected
                for c in candidates:
                    if c == b:
                        continue
                    if c in selected_set:
                        continue
                    # Perform swap
                    for ts_idx in block_to_ts[b]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_to_ts[c]:
                        cover_count[ts_idx] += 1
                    block_indices[i] = c
                    selected_set.discard(b)
                    selected_set.add(c)
                    improved = True
                    break
                if improved:
                    break

        # 2-exchange: remove two blocks, replace with one
        if len(block_indices) >= 3:
            improved2 = True
            while improved2 and time.time() < deadline - 2.0:
                improved2 = False
                n_sel = len(block_indices)
                n_pairs = min(200, n_sel * (n_sel - 1) // 2)
                for _ in range(n_pairs):
                    if time.time() > deadline - 2.0:
                        break
                    i, j = rng.sample(range(n_sel), 2)
                    bi = block_indices[i]
                    bj = block_indices[j]
                    # Compute gap after removing both
                    gap = 0
                    for ts_idx in block_to_ts[bi]:
                        if cover_count[ts_idx] == 1:
                            gap |= (1 << ts_idx)
                    for ts_idx in block_to_ts[bj]:
                        if cover_count[ts_idx] == 1:
                            gap |= (1 << ts_idx)
                    # Pairwise intersection with count==2
                    inter = block_masks[bi] & block_masks[bj]
                    m = inter
                    while m:
                        idx = (m & -m).bit_length() - 1
                        if cover_count[idx] == 2:
                            gap |= (1 << idx)
                        m &= m - 1
                    if gap == 0:
                        for ts_idx in block_to_ts[bi]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bj]:
                            cover_count[ts_idx] -= 1
                        if j > i:
                            block_indices.pop(j)
                            block_indices.pop(i)
                        else:
                            block_indices.pop(i)
                            block_indices.pop(j)
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        improved2 = True
                        break
                    if gap.bit_count() > 30:
                        continue
                    # Find candidate covering gap
                    gap_ts_list = []
                    m = gap
                    while m:
                        idx = (m & -m).bit_length() - 1
                        gap_ts_list.append(idx)
                        m &= m - 1
                    if not gap_ts_list:
                        continue
                    candidates = list(ts_to_blocks[gap_ts_list[0]])
                    for ts_idx in gap_ts_list[1:]:
                        if not candidates:
                            break
                        candidates = [c for c in candidates if c in ts_to_blocks[ts_idx]]
                    for c in candidates:
                        if c == bi or c == bj:
                            continue
                        if c in selected_set:
                            continue
                        for ts_idx in block_to_ts[bi]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bj]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[c]:
                            cover_count[ts_idx] += 1
                        if j > i:
                            block_indices.pop(j)
                            block_indices.pop(i)
                        else:
                            block_indices.pop(i)
                            block_indices.pop(j)
                        block_indices.append(c)
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.add(c)
                        improved2 = True
                        break
                    if improved2:
                        break

        # 3-exchange: remove three blocks, replace with one
        if len(block_indices) >= 4:
            improved3 = True
            while improved3 and time.time() < deadline - 2.0:
                improved3 = False
                n_sel = len(block_indices)
                n_triples = min(150, n_sel * (n_sel - 1) * (n_sel - 2) // 6)
                for _ in range(n_triples):
                    if time.time() > deadline - 2.0:
                        break
                    i, j, l = rng.sample(range(n_sel), 3)
                    bi = block_indices[i]
                    bj = block_indices[j]
                    bl = block_indices[l]
                    gap = 0
                    for ts_idx in block_to_ts[bi]:
                        if cover_count[ts_idx] == 1:
                            gap |= (1 << ts_idx)
                    for ts_idx in block_to_ts[bj]:
                        if cover_count[ts_idx] == 1:
                            gap |= (1 << ts_idx)
                    for ts_idx in block_to_ts[bl]:
                        if cover_count[ts_idx] == 1:
                            gap |= (1 << ts_idx)
                    # Pairwise intersections with count==2
                    for pair in [(bi, bj), (bi, bl), (bj, bl)]:
                        inter = block_masks[pair[0]] & block_masks[pair[1]]
                        m = inter
                        while m:
                            idx = (m & -m).bit_length() - 1
                            if cover_count[idx] == 2:
                                gap |= (1 << idx)
                            m &= m - 1
                    # Triple intersection with count==3
                    inter_ijk = block_masks[bi] & block_masks[bj] & block_masks[bl]
                    m = inter_ijk
                    while m:
                        idx = (m & -m).bit_length() - 1
                        if cover_count[idx] == 3:
                            gap |= (1 << idx)
                        m &= m - 1
                    if gap == 0:
                        for ts_idx in block_to_ts[bi]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bj]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bl]:
                            cover_count[ts_idx] -= 1
                        idxs = sorted([i, j, l], reverse=True)
                        for idx in idxs:
                            block_indices.pop(idx)
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.discard(bl)
                        improved3 = True
                        break
                    if gap.bit_count() > 35:
                        continue
                    gap_ts_list = []
                    m = gap
                    while m:
                        idx = (m & -m).bit_length() - 1
                        gap_ts_list.append(idx)
                        m &= m - 1
                    if not gap_ts_list:
                        continue
                    candidates = list(ts_to_blocks[gap_ts_list[0]])
                    for ts_idx in gap_ts_list[1:]:
                        if not candidates:
                            break
                        candidates = [c for c in candidates if c in ts_to_blocks[ts_idx]]
                    for c in candidates:
                        if c == bi or c == bj or c == bl:
                            continue
                        if c in selected_set:
                            continue
                        for ts_idx in block_to_ts[bi]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bj]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[bl]:
                            cover_count[ts_idx] -= 1
                        for ts_idx in block_to_ts[c]:
                            cover_count[ts_idx] += 1
                        idxs = sorted([i, j, l], reverse=True)
                        for idx in idxs:
                            block_indices.pop(idx)
                        block_indices.append(c)
                        selected_set.discard(bi)
                        selected_set.discard(bj)
                        selected_set.discard(bl)
                        selected_set.add(c)
                        improved3 = True
                        break
                    if improved3:
                        break

        return block_indices

    # --- Main search ---
    best_blocks = None
    best_count = float('inf')
    rng = random.Random(42)

    # Deterministic greedy first
    sol = run_greedy(rng, deterministic=True)
    if sol is not None:
        sol = polish(sol)
        sol = exchange_search(sol, rng)
        sol = polish(sol)
        best_blocks = sol[:]
        best_count = len(sol)

    # Randomized greedy restarts
    max_restarts = 400
    for restart in range(max_restarts):
        if time.time() > deadline - 8.0:
            break

        sol = run_greedy(rng, deterministic=False)
        if sol is None:
            break

        sol = polish(sol)
        cur_len = len(sol)

        if cur_len < best_count:
            sol = exchange_search(sol, rng)
            sol = polish(sol)
            if len(sol) < best_count:
                best_count = len(sol)
                best_blocks = sol[:]
        elif cur_len <= best_count + 8:
            sol = exchange_search(sol, rng)
            sol = polish(sol)
            if len(sol) < best_count:
                best_count = len(sol)
                best_blocks = sol[:]

        if best_count <= 335:
            break

    # Final polish on best
    if best_blocks is not None:
        best_blocks = polish(best_blocks)
        best_blocks = exchange_search(best_blocks, rng)
        best_blocks = polish(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_blocks[b]) for b in best_blocks]
