import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 117.0

    def time_left():
        return deadline - time.time()

    # Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)

    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # Convert to numpy arrays
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]

    # Hardness: number of blocks covering each tset (lower = harder)
    tset_hardness = np.array([len(bl) for bl in tset_to_blocks], dtype=np.float64)

    def run_greedy(randomize_ties=False, seed=None, use_hardness=False):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)

        selected = []
        remaining = n_tsub

        while remaining > 0:
            if time_left() < 25.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break

            candidates = np.where(gain == max_gain)[0]

            if len(candidates) == 1:
                bi = int(candidates[0])
            elif randomize_ties:
                bi = int(candidates[rng.randint(0, len(candidates) - 1)])
            elif use_hardness:
                # Score by sum of 1/hardness over covered tsets
                best_score = -1
                best_bi = int(candidates[0])
                for ci in candidates:
                    bts = block_tsets_np[ci]
                    score = np.sum(1.0 / (tset_hardness[bts] + 1.0))
                    if score > best_score:
                        best_score = score
                        best_bi = int(ci)
                bi = best_bi
            else:
                bi = int(candidates[0])

            selected.append(bi)
            gain[bi] = 0

            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1

        return selected

    def reduce_solution(sel, max_passes=3):
        if not sel:
            return []
        if time_left() < 5.0:
            return sel

        for _ in range(max_passes):
            if time_left() < 5.0:
                break

            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in sel:
                cov[block_tsets_np[bi]] += 1

            new_sel = []
            for bi in sel:
                bts = block_tsets_np[bi]
                if np.any(cov[bts] == 1):
                    new_sel.append(bi)
                else:
                    cov[bts] -= 1

            if len(new_sel) == len(sel):
                break
            sel = new_sel

        return sel

    def local_search(selected, max_rounds=50):
        if not selected or time_left() < 15.0:
            return selected

        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            cov[block_tsets_np[bi]] += 1

        best_selected = list(selected)
        best_size = len(selected)

        for rnd in range(max_rounds):
            if time_left() < 15.0:
                break

            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)

            if not selected:
                break

            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1

            improved = False
            n_swaps = min(len(selected), 30)

            for _ in range(n_swaps):
                if time_left() < 15.0:
                    break
                if not selected:
                    break

                idx_remove = random.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]
                bts_remove = block_tsets_np[bi_remove]
                newly_uncovered = bts_remove[cov[bts_remove] == 1]

                if len(newly_uncovered) == 0:
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    improved = True
                    continue

                # Find blocks that cover all newly_uncovered tsets
                if len(newly_uncovered) > 3:
                    continue  # Too hard to find a single block

                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])

                cand_set.discard(bi_remove)
                cand_set -= set(selected)

                if not cand_set:
                    continue

                cand_list = list(cand_set)
                if len(cand_list) <= 5:
                    bi_add = random.choice(cand_list)
                else:
                    # Score by additional coverage
                    bts_remove_set = set(bts_remove.tolist())
                    scores = []
                    for bj in cand_list:
                        bts_bj = block_tsets_np[bj]
                        additional = np.sum(~np.isin(bts_bj, bts_remove_set))
                        scores.append((additional, bj))
                    scores.sort(key=lambda x: -x[0])
                    top_n = min(3, len(scores))
                    bi_add = random.choice([s[1] for s in scores[:top_n]])

                bts_add = block_tsets_np[bi_add]
                cov[bts_remove] -= 1
                cov[bts_add] += 1

                valid = np.all(cov[bts_remove] >= 1)
                if not valid:
                    cov[bts_add] -= 1
                    cov[bts_remove] += 1
                    continue

                selected[idx_remove] = bi_add
                improved = True

                selected = reduce_solution(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)

                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1

            if not improved:
                break

        return best_selected

    def simulated_annealing(selected, max_iters=200):
        if not selected or time_left() < 20.0:
            return selected

        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            cov[block_tsets_np[bi]] += 1

        best_selected = list(selected)
        best_size = len(selected)
        current_size = len(selected)

        T = 5.0
        cooling = 0.995

        for it in range(max_iters):
            if time_left() < 20.0:
                break

            # Try a random swap: remove one block, add another
            if current_size < 2:
                break

            idx_remove = random.randint(0, current_size - 1)
            bi_remove = selected[idx_remove]
            bts_remove = block_tsets_np[bi_remove]
            newly_uncovered = bts_remove[cov[bts_remove] == 1]

            if len(newly_uncovered) == 0:
                # Redundant block
                cov[bts_remove] -= 1
                selected.pop(idx_remove)
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_selected = list(selected)
                continue

            if len(newly_uncovered) > 2:
                # Hard to fix, try adding a block that covers some
                # Just skip this iteration
                continue

            # Find candidate blocks covering the newly uncovered tsets
            if len(newly_uncovered) == 1:
                ti = int(newly_uncovered[0])
                cand_list = [bj for bj in tset_to_blocks[ti] if bj != bi_remove and bj not in selected]
            else:
                ti0 = int(newly_uncovered[0])
                ti1 = int(newly_uncovered[1])
                cand_list = [bj for bj in tset_to_blocks[ti0]
                             if bj != bi_remove and bj not in selected and bj in tset_to_blocks[ti1]]

            if not cand_list:
                continue

            bi_add = random.choice(cand_list)
            bts_add = block_tsets_np[bi_add]

            # Apply swap
            cov[bts_remove] -= 1
            cov[bts_add] += 1

            # Check if valid
            valid = np.all(cov[bts_remove] >= 1)

            if valid:
                selected[idx_remove] = bi_add
                # Try to reduce
                selected = reduce_solution(selected, max_passes=1)
                current_size = len(selected)
                if current_size < best_size:
                    best_size = current_size
                    best_selected = list(selected)
                # Rebuild cov
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1
            else:
                # Revert
                cov[bts_add] -= 1
                cov[bts_remove] += 1

            T *= cooling

        return best_selected

    # Main search
    best_overall = None
    best_size_overall = float('inf')

    def try_solution(sel):
        nonlocal best_overall, best_size_overall
        if sel:
            sel = reduce_solution(sel)
            if len(sel) < best_size_overall:
                best_size_overall = len(sel)
                best_overall = sel

    # 1. Deterministic greedy
    if time_left() > 30.0:
        sel = run_greedy(randomize_ties=False)
        try_solution(sel)

    # 2. Hardness-aware greedy
    if time_left() > 30.0:
        sel = run_greedy(use_hardness=True)
        try_solution(sel)

    # 3. Local search on best so far
    if best_overall and time_left() > 25.0:
        sel = local_search(list(best_overall))
        try_solution(sel)

    # 4. Simulated annealing
    if best_overall and time_left() > 25.0:
        sel = simulated_annealing(list(best_overall))
        try_solution(sel)

    # 5. Randomized greedy restarts
    n_restarts = 0
    while time_left() > 15.0 and n_restarts < 6:
        n_restarts += 1
        seed = n_restarts * 104729
        sel = run_greedy(randomize_ties=True, seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        if len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
            # Local search on improved solution
            if time_left() > 20.0:
                sel2 = local_search(list(best_overall))
                try_solution(sel2)

    # 6. More hardness-aware variants
    for hseed in [7, 13, 42]:
        if time_left() < 15.0:
            break
        sel = run_greedy(randomize_ties=True, seed=hseed, use_hardness=True)
        if not sel:
            continue
        sel = reduce_solution(sel)
        if len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
            if time_left() > 20.0:
                sel2 = local_search(list(best_overall))
                try_solution(sel2)

    # 7. Final polish
    if best_overall and time_left() > 10.0:
        best_overall = local_search(list(best_overall), max_rounds=30)
        best_overall = reduce_solution(best_overall)

    if best_overall is None:
        best_overall = run_greedy(randomize_ties=False)
        best_overall = reduce_solution(best_overall)

    return [list(all_blocks[bi]) for bi in best_overall]
