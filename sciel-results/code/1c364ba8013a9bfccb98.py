import itertools
import time
import random
import heapq
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)

    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute for each block: list of subset indices it covers
    block_subsets = []
    # Inverse: for each subset, which blocks cover it?
    blocks_covering_subset = [[] for _ in range(N_subsets)]

    for i, b in enumerate(all_blocks):
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        for s in subs:
            blocks_covering_subset[s].append(i)

    # Convert blocks_covering_subset to numpy arrays for faster iteration
    blocks_covering_subset_np = [np.array(lst, dtype=np.int32) for lst in blocks_covering_subset]

    # Precompute bitmask for each block (for fast swap validation in local search)
    block_bitmasks = []
    for subs in block_subsets:
        mask = 0
        for s in subs:
            mask |= (1 << s)
        block_bitmasks.append(mask)

    def run_greedy(seed=0):
        """Priority-queue-based greedy set cover."""
        rng = random.Random(seed)
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.zeros(N_blocks, dtype=np.int32)
        for i, subs in enumerate(block_subsets):
            potential[i] = len(subs)

        # Max-heap: (-potential, block_idx)
        heap = [(-potential[i], i) for i in range(N_blocks)]
        heapq.heapify(heap)

        selected_blocks = []
        selected_set = set()
        in_heap = np.ones(N_blocks, dtype=bool)

        while np.any(cover_count == 0):
            if time.time() > deadline - 5.0:
                return None, None

            # Pop the best block from the heap
            while heap:
                neg_pot, b_idx = heapq.heappop(heap)
                if not in_heap[b_idx]:
                    continue
                if -neg_pot != potential[b_idx]:
                    # Stale entry, skip
                    continue
                break
            else:
                break

            selected_blocks.append(b_idx)
            selected_set.add(b_idx)
            in_heap[b_idx] = False

            # Update cover_count and potential
            for s in block_subsets[b_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    # This subset was newly covered; decrement potential of all blocks covering it
                    for b in blocks_covering_subset_np[s]:
                        if b not in selected_set:
                            potential[b] -= 1
                            heapq.heappush(heap, (-potential[b], b))

        return selected_blocks, cover_count

    def local_search(selected_blocks, cover_count):
        """Swap-based local search to reduce block count."""
        selected_blocks = list(selected_blocks)
        cover_count = cover_count.copy()
        selected_set = set(selected_blocks)

        best_len = len(selected_blocks)
        best_blocks = list(selected_blocks)

        steps = 0
        max_steps = 200000
        no_improve_count = 0
        max_no_improve = 8000

        while steps < max_steps:
            steps += 1
            if time.time() > deadline - 2.0:
                break
            if not selected_blocks:
                break

            b_out_idx = random.randint(0, len(selected_blocks) - 1)
            b_out = selected_blocks[b_out_idx]

            # Find critical subsets (covered only by b_out)
            critical_subsets = []
            for s in block_subsets[b_out]:
                if cover_count[s] == 1:
                    critical_subsets.append(s)

            if not critical_subsets:
                # Block is redundant; remove it
                for s in block_subsets[b_out]:
                    cover_count[s] -= 1
                selected_blocks.pop(b_out_idx)
                selected_set.remove(b_out)

                if len(selected_blocks) < best_len:
                    best_len = len(selected_blocks)
                    best_blocks = list(selected_blocks)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Need to find a swap: a block that covers all critical subsets
                critical_mask = 0
                for s in critical_subsets:
                    critical_mask |= (1 << s)

                first_crit = critical_subsets[0]
                candidates = blocks_covering_subset_np[first_crit]

                found_swap = False
                best_swap_score = -1
                best_swap_block = -1

                for b_in in candidates:
                    if b_in in selected_set:
                        continue

                    if (block_bitmasks[b_in] & critical_mask) == critical_mask:
                        # Valid swap: b_in covers all critical subsets
                        # Score by how many new subsets b_in covers (beyond critical)
                        score = 0
                        for s in block_subsets[b_in]:
                            if cover_count[s] == 0:
                                score += 1
                        if score > best_swap_score:
                            best_swap_score = score
                            best_swap_block = b_in

                if best_swap_block != -1:
                    b_in = best_swap_block
                    # Perform the swap
                    for s in block_subsets[b_out]:
                        cover_count[s] -= 1
                    selected_blocks.pop(b_out_idx)
                    selected_set.remove(b_out)

                    selected_blocks.append(b_in)
                    selected_set.add(b_in)
                    for s in block_subsets[b_in]:
                        cover_count[s] += 1

                    # Cascade: remove any now-redundant blocks
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(selected_blocks) - 1, -1, -1):
                            b_idx = selected_blocks[i]
                            is_redundant = True
                            for s in block_subsets[b_idx]:
                                if cover_count[s] == 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                for s in block_subsets[b_idx]:
                                    cover_count[s] -= 1
                                selected_blocks.pop(i)
                                selected_set.remove(b_idx)
                                changed = True
                                if len(selected_blocks) < best_len:
                                    best_len = len(selected_blocks)
                                    best_blocks = list(selected_blocks)
                                    no_improve_count = 0

                    if len(selected_blocks) >= best_len:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                else:
                    no_improve_count += 1

            if no_improve_count > max_no_improve:
                break

        return best_blocks, best_len

    # Main execution
    best_blocks_global = None
    best_len_global = float('inf')

    # 1. Deterministic greedy
    blocks, counts = run_greedy(0)
    if blocks:
        best_blocks_local, best_len_local = local_search(blocks, counts)
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local

    # 2. Randomized greedy restarts — adaptive count based on remaining time
    seed = 1
    while time.time() < deadline - 15.0:
        if seed > 50:  # cap on total restarts
            break
        blocks, counts = run_greedy(seed)
        if not blocks:
            seed += 1
            continue

        best_blocks_local, best_len_local = local_search(blocks, counts)

        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local

        seed += 1

    # 3. Final polish with remaining time
    if best_blocks_global and time.time() < deadline - 5.0:
        cover_count_final = np.zeros(N_subsets, dtype=np.int32)
        for b in best_blocks_global:
            for s in block_subsets[b]:
                cover_count_final[s] += 1

        best_blocks_final, best_len_final = local_search(list(best_blocks_global), cover_count_final)
        if best_len_final < best_len_global:
            best_len_global = best_len_final
            best_blocks_global = best_blocks_final

    if not best_blocks_global:
        blocks, _ = run_greedy(0)
        best_blocks_global = blocks

    return [list(all_blocks[i]) for i in best_blocks_global]
