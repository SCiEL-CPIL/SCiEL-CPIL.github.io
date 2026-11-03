import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 143.0

    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []

        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts

        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])

        selected_mask = np.zeros(num_blocks, dtype=np.bool_)

        while uncovered_count > 0 and time.time() < deadline - 30:
            best_idx = -1
            best_gain = -1.0
            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i
            if best_idx == -1 or best_gain <= 0:
                break
            selected.append(best_idx)
            selected_mask[best_idx] = True
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w
        return selected

    def run_local_search(initial_indices, max_time_per_run, seed):
        rng = random.Random(seed)
        current = list(initial_indices)
        if not current:
            return []

        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1

        best_size = len(current)
        best_solution = list(current)

        local_start = time.time()
        no_improve = 0

        while (time.time() - local_start) < max_time_per_run:
            improved = False

            # Phase 1: Remove redundant blocks
            indices = list(range(len(current)))
            rng.shuffle(indices)
            for i in indices:
                if (time.time() - local_start) > max_time_per_run:
                    break
                if i >= len(current):
                    continue
                b_idx = current[i]
                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if cover_counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for ts_idx in block_to_tsets[b_idx]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break
            if improved:
                no_improve = 0
                continue

            # Phase 2: 1-for-1 Swaps
            swap_attempts = 0
            max_swap_attempts = 30
            while swap_attempts < max_swap_attempts:
                if (time.time() - local_start) > max_time_per_run:
                    break
                swap_attempts += 1
                if not current:
                    break
                i = rng.randint(0, len(current) - 1)
                b_out = current[i]
                unique_tsets = [ts for ts in block_to_tsets[b_out] if cover_counts[ts] == 1]
                if not unique_tsets:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break
                min_count_ts = unique_tsets[0]
                min_count = len(tset_to_blocks[min_count_ts])
                for ts in unique_tsets[1:]:
                    c = len(tset_to_blocks[ts])
                    if c < min_count:
                        min_count = c
                        min_count_ts = ts
                candidates = tset_to_blocks[min_count_ts]
                best_cand = -1
                best_score = -1.0
                for b_cand in candidates:
                    if b_cand == b_out:
                        continue
                    cand_tsets = set(block_to_tsets[b_cand])
                    valid = all(ts in cand_tsets for ts in unique_tsets)
                    if not valid:
                        continue
                    score = 0
                    for ts in block_to_tsets[b_cand]:
                        if cover_counts[ts] >= 2:
                            score += 1
                    score += rng.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_cand = b_cand
                if best_cand != -1:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_cand]:
                        cover_counts[ts_idx] += 1
                    current[i] = best_cand
                    improved = True
                    break
            if improved:
                no_improve = 0
                continue

            # Phase 3: 2-for-1 moves
            if len(current) > 2:
                pair_attempts = 0
                max_pair_attempts = 15
                while pair_attempts < max_pair_attempts:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    pair_attempts += 1
                    i1 = rng.randint(0, len(current) - 1)
                    i2 = rng.randint(0, len(current) - 1)
                    if i1 == i2:
                        continue
                    b_out1 = current[i1]
                    b_out2 = current[i2]
                    tsets_A = set(block_to_tsets[b_out1])
                    tsets_B = set(block_to_tsets[b_out2])
                    lost_tsets = set()
                    for ts in tsets_A:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                        elif cover_counts[ts] == 2 and ts in tsets_B:
                            lost_tsets.add(ts)
                    for ts in tsets_B:
                        if cover_counts[ts] == 1:
                            lost_tsets.add(ts)
                    if not lost_tsets:
                        continue
                    if len(lost_tsets) > 10:
                        continue
                    lost_list = list(lost_tsets)
                    candidates = set(tset_to_blocks[lost_list[0]])
                    candidates.discard(b_out1)
                    candidates.discard(b_out2)
                    for ts in lost_list[1:]:
                        if not candidates:
                            break
                        candidates.intersection_update(tset_to_blocks[ts])
                    if not candidates:
                        continue
                    best_c = -1
                    best_add = -1
                    for c in list(candidates)[:25]:
                        add = 0
                        for ts in block_to_tsets[c]:
                            if cover_counts[ts] == 2:
                                add += 1
                        if add > best_add:
                            best_add = add
                            best_c = c
                    if best_c != -1:
                        for ts_idx in block_to_tsets[b_out1]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b_out2]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[best_c]:
                            cover_counts[ts_idx] += 1
                        if i1 < i2:
                            current.pop(i2)
                            current.pop(i1)
                        else:
                            current.pop(i1)
                            current.pop(i2)
                        current.append(best_c)
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
            if improved:
                no_improve = 0
                continue

            # Phase 4: 3-for-2 moves
            if len(current) > 3:
                triple_attempts = 0
                max_triple_attempts = 8
                while triple_attempts < max_triple_attempts:
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    triple_attempts += 1
                    i1 = rng.randint(0, len(current) - 1)
                    i2 = rng.randint(0, len(current) - 1)
                    i3 = rng.randint(0, len(current) - 1)
                    if len({i1, i2, i3}) < 3:
                        continue
                    b1 = current[i1]
                    b2 = current[i2]
                    b3 = current[i3]
                    ts1 = set(block_to_tsets[b1])
                    ts2 = set(block_to_tsets[b2])
                    ts3 = set(block_to_tsets[b3])
                    lost = set()
                    all_ts = ts1 | ts2 | ts3
                    for ts in all_ts:
                        cnt = cover_counts[ts]
                        in_a = ts in ts1
                        in_b = ts in ts2
                        in_c = ts in ts3
                        in_count = in_a + in_b + in_c
                        if cnt - in_count < 1:
                            lost.add(ts)
                    if not lost:
                        continue
                    if len(lost) > 14:
                        continue
                    lost_list = list(lost)
                    # Find two blocks covering all lost
                    c1_candidates = set(tset_to_blocks[lost_list[0]])
                    for ts in lost_list[1:]:
                        if not c1_candidates:
                            break
                        c1_candidates.intersection_update(tset_to_blocks[ts])
                    if not c1_candidates:
                        continue
                    best_pair = None
                    best_pair_score = -1
                    for c1 in list(c1_candidates)[:15]:
                        if c1 in (b1, b2, b3):
                            continue
                        ts_c1 = set(block_to_tsets[c1])
                        remaining = lost - ts_c1
                        if not remaining:
                            c2_candidates = set(tset_to_blocks[lost_list[0]])
                            for ts in lost_list[1:]:
                                if not c2_candidates:
                                    break
                                c2_candidates.intersection_update(tset_to_blocks[ts])
                        else:
                            rem_list = list(remaining)
                            c2_candidates = set(tset_to_blocks[rem_list[0]])
                            for ts in rem_list[1:]:
                                if not c2_candidates:
                                    break
                                c2_candidates.intersection_update(tset_to_blocks[ts])
                        if not c2_candidates:
                            continue
                        for c2 in list(c2_candidates)[:15]:
                            if c2 in (b1, b2, b3) or c2 == c1:
                                continue
                            if c1 in (b1, b2, b3) or c2 in (b1, b2, b3):
                                continue
                            score = 0
                            for ts in block_to_tsets[c1]:
                                if cover_counts[ts] == 2:
                                    score += 1
                            for ts in block_to_tsets[c2]:
                                if cover_counts[ts] == 2:
                                    score += 1
                            score += rng.random() * 0.1
                            if score > best_pair_score:
                                best_pair_score = score
                                best_pair = (c1, c2)
                    if best_pair is not None:
                        c1, c2 = best_pair
                        for ts_idx in block_to_tsets[b1]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b2]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[b3]:
                            cover_counts[ts_idx] -= 1
                        for ts_idx in block_to_tsets[c1]:
                            cover_counts[ts_idx] += 1
                        for ts_idx in block_to_tsets[c2]:
                            cover_counts[ts_idx] += 1
                        # Remove i1, i2, i3 (sorted descending)
                        idxs = sorted([i1, i2, i3], reverse=True)
                        for idx in idxs:
                            current.pop(idx)
                        current.append(c1)
                        current.append(c2)
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                        break
            if improved:
                no_improve = 0
                continue

            no_improve += 1
            if no_improve > 100:
                break

        return best_solution

    best_global_solution = None
    best_global_size = float('inf')

    seeds = [42, 123, 777, 1234, 5678, 9999, 1111, 2222, 3333, 4444]
    for seed in seeds:
        if time.time() > deadline - 40:
            break
        t_start = time.time()
        init = greedy_cover(seed)
        if init:
            remaining = deadline - time.time() - 10
            ls_time = min(30.0, remaining)
            if ls_time > 8:
                sol = run_local_search(init, max_time_per_run=ls_time, seed=seed)
                if sol and len(sol) < best_global_size:
                    best_global_size = len(sol)
                    best_global_solution = sol

    # Final polish
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 3
        if remaining_time > 8:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time, seed=99999)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol

    if not best_global_solution:
        return []
    return get_solution(best_global_solution)


if __name__ == "__main__":
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
