import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s kill

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = [tset_index[ts] for ts in itertools.combinations(b, t)]
        block_tsets.append(ts_indices)

    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # 2. Greedy Set Cover with vectorized gain updates
    uncovered = np.ones(n_tsub, dtype=bool)
    gain = np.array([len(block_tsets[bi]) for bi in range(n_blocks)], dtype=np.int32)

    selected = []
    remaining = n_tsub

    while remaining > 0:
        if time.time() > deadline - 60.0:
            break
        bi = int(np.argmax(gain))
        if gain[bi] == 0:
            break
        selected.append(bi)
        gain[bi] = 0

        # Find newly covered tsets
        newly_covered_mask = np.array([uncovered[ti] for ti in block_tsets[bi]], dtype=bool)
        newly_covered = [block_tsets[bi][i] for i in range(len(block_tsets[bi])) if newly_covered_mask[i]]

        if newly_covered:
            remaining -= len(newly_covered)
            uncovered[newly_covered] = False
            # Vectorized gain decrement
            # For each newly covered tset, decrement gain of all blocks covering it
            # Build a list of all block indices to decrement
            blocks_to_decrement = []
            for ti in newly_covered:
                blocks_to_decrement.extend(tset_to_blocks[ti])
            if blocks_to_decrement:
                # Use numpy to decrement efficiently
                # We need to handle duplicates (same block may cover multiple newly covered tsets)
                # Use a counter approach
                arr = np.array(blocks_to_decrement, dtype=np.int32)
                # Count occurrences
                unique, counts = np.unique(arr, return_counts=True)
                gain[unique] -= counts

    # 3. Fast Reduction Pass
    def reduce_solution(sel):
        if not sel:
            return []
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1

        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)

        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    selected = reduce_solution(selected)

    # 4. Aggressive deletion polish
    def aggressive_polish(sel, time_limit):
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in sel:
                for ti in block_tsets[bi]:
                    cov[ti] += 1
            new_sel = []
            for bi in sel:
                essential = False
                for ti in block_tsets[bi]:
                    if cov[ti] == 1:
                        essential = True
                        break
                if not essential:
                    for ti in block_tsets[bi]:
                        cov[ti] -= 1
                    changed = True
                else:
                    new_sel.append(bi)
            sel = new_sel
        return sel

    selected = aggressive_polish(selected, deadline - 40.0)

    # 5. Simulated Annealing Local Search
    cov = np.zeros(n_tsub, dtype=np.int32)
    for bi in selected:
        for ti in block_tsets[bi]:
            cov[ti] += 1

    best_selected = list(selected)
    best_size = len(selected)

    # Precompute "uniqueness" for each block: number of t-sets it uniquely covers
    # This helps target removals at low-impact blocks

    T = 5.0
    cooling_rate = 0.997
    rng = random.Random(12345)

    while time.time() < deadline - 5.0 and T > 0.1:
        if not selected:
            break

        move_type = rng.random()

        if move_type < 0.5:
            # Kick: remove one block, greedily refill uncovered
            idx = rng.randint(0, len(selected) - 1)
            bi_kick = selected[idx]

            saved_selected = list(selected)
            saved_cov = cov.copy()

            # Remove
            selected.pop(idx)
            for ti in block_tsets[bi_kick]:
                cov[ti] -= 1

            uncovered_tsets = [ti for ti in block_tsets[bi_kick] if cov[ti] == 0]

            if not uncovered_tsets:
                # Valid removal! Polish and accept
                selected = aggressive_polish(selected, deadline - 5.0)
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                T *= cooling_rate
                continue

            # Greedily cover uncovered
            new_blocks = []
            uncovered_set = set(uncovered_tsets)

            while uncovered_set and time.time() < deadline - 5.0:
                best_bj = -1
                max_cover = 0
                candidates = set()
                for ti in uncovered_set:
                    candidates.update(tset_to_blocks[ti])

                for bj in candidates:
                    cnt = sum(1 for ti in block_tsets[bj] if ti in uncovered_set)
                    if cnt > max_cover:
                        max_cover = cnt
                        best_bj = bj

                if best_bj == -1:
                    break

                new_blocks.append(best_bj)
                for ti in block_tsets[best_bj]:
                    if ti in uncovered_set:
                        uncovered_set.discard(ti)

            for bj in new_blocks:
                selected.append(bj)
                for ti in block_tsets[bj]:
                    cov[ti] += 1

            selected = aggressive_polish(selected, deadline - 5.0)

            delta = len(selected) - len(saved_selected)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                cov = saved_cov

        else:
            # Swap: remove one block, add one different block
            if len(selected) < 2:
                T *= cooling_rate
                continue

            idx_remove = rng.randint(0, len(selected) - 1)
            bi_remove = selected[idx_remove]

            single_covered = [ti for ti in block_tsets[bi_remove] if cov[ti] == 1]

            if single_covered:
                candidates = set()
                for ti in single_covered:
                    candidates.update(tset_to_blocks[ti])
                candidates.discard(bi_remove)
                selected_set = set(selected)
                candidates -= selected_set
                if not candidates:
                    T *= cooling_rate
                    continue
                bi_add = rng.choice(list(candidates))
            else:
                bi_add = rng.randint(0, n_blocks - 1)
                if bi_add == bi_remove or bi_add in selected:
                    T *= cooling_rate
                    continue

            saved_selected = list(selected)
            saved_cov = cov.copy()

            # Perform swap
            selected[idx_remove] = bi_add
            for ti in block_tsets[bi_remove]:
                cov[ti] -= 1
            for ti in block_tsets[bi_add]:
                cov[ti] += 1

            # Check coverage validity
            valid = all(cov[ti] > 0 for ti in block_tsets[bi_remove])

            if not valid:
                selected = saved_selected
                cov = saved_cov
                T *= cooling_rate
                continue

            selected = aggressive_polish(selected, deadline - 5.0)

            delta = len(selected) - len(saved_selected)
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    for ti in block_tsets[bi]:
                        cov[ti] += 1
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                selected = saved_selected
                cov = saved_cov

        T *= cooling_rate

    # Final aggressive polish
    best_selected = aggressive_polish(best_selected, deadline - 2.0)

    return [list(all_blocks[bi]) for bi in best_selected]
