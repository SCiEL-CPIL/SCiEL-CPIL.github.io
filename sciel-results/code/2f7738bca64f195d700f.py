import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit

    # Total number of t-subsets
    total_t_subsets = 1
    for i in range(t):
        total_t_subsets *= (v - i)
        total_t_subsets //= (i + 1)

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))

    # Map each t-subset to an index for efficient tracking
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute for each possible block (k-subset) which t-subsets it covers
    # This is C(v,k) blocks, each covering C(k,t) t-subsets
    # For v=19, k=7: C(19,7) = 50388 blocks, each with C(7,4)=35 t-subsets
    all_blocks_list = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks_list)}
    num_all_blocks = len(all_blocks_list)

    # For each block, precompute the list of t-subset indices it covers
    block_t_subsets_all = []
    for b in all_blocks_list:
        ts_list = []
        for ts in itertools.combinations(b, t):
            ts_list.append(t_subset_to_idx[ts])
        block_t_subsets_all.append(ts_list)

    # Greedy initialization
    def greedy_construct():
        coverage = [0] * total_t_subsets
        blocks = []
        block_ts_in_solution = []

        uncovered = [True] * total_t_subsets
        num_uncovered = total_t_subsets

        while num_uncovered > 0:
            if time.time() > deadline - 20:
                break

            # Pick a random uncovered t-subset to start
            idx = random.randrange(total_t_subsets)
            while not uncovered[idx]:
                idx = (idx + 1) % total_t_subsets

            seed_t = all_t_subsets[idx]
            block = list(seed_t)

            # Extend to k elements greedily
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        ts_idx = t_subset_to_idx[ts]
                        if uncovered[ts_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)

            block.sort()
            blocks.append(block)

            # Update coverage
            new_ts = []
            for ts in itertools.combinations(block, t):
                ts_idx = t_subset_to_idx[ts]
                new_ts.append(ts_idx)
                if coverage[ts_idx] == 0:
                    num_uncovered -= 1
                    uncovered[ts_idx] = False
                coverage[ts_idx] += 1

            block_ts_in_solution.append(new_ts)

        return blocks, coverage, block_ts_in_solution

    # Remove redundant blocks
    def remove_redundant(blocks, coverage, block_ts_in_solution):
        improved = True
        while improved:
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                redundant = True
                for ts_idx in block_ts_in_solution[i]:
                    if coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_ts_in_solution[i]:
                        coverage[ts_idx] -= 1
                    blocks.pop(i)
                    block_ts_in_solution.pop(i)
                    improved = True
        return blocks, coverage, block_ts_in_solution

    # Run greedy to get initial solution
    blocks, coverage, block_ts_in_solution = greedy_construct()
    blocks, coverage, block_ts_in_solution = remove_redundant(blocks, coverage, block_ts_in_solution)

    best_blocks = blocks[:]
    best_count = len(blocks)
    best_coverage = coverage[:]
    best_block_ts = block_ts_in_solution[:]

    # Simulated Annealing
    # State: set of blocks (represented as list of block indices into all_blocks_list)
    # Cost: number of uncovered t-subsets + penalty for block count
    # We want to minimize block count while keeping all t-subsets covered

    # Convert current solution to block indices
    current_block_indices = []
    for b in blocks:
        current_block_indices.append(block_to_idx[tuple(b)])

    def compute_uncovered(block_indices, coverage):
        """Count uncovered t-subsets given current coverage."""
        unc = 0
        for i in range(total_t_subsets):
            if coverage[i] == 0:
                unc += 1
        return unc

    # SA parameters
    initial_temp = 50.0
    cooling_rate = 0.9995
    min_temp = 0.1
    max_iterations = 500000

    current_block_indices = [block_to_idx[tuple(b)] for b in blocks]
    current_coverage = coverage[:]
    current_block_ts = [block_t_subsets_all[bi] for bi in current_block_indices]

    # Track best
    best_block_indices = current_block_indices[:]
    best_count = len(current_block_indices)

    temp = initial_temp
    iteration = 0

    while temp > min_temp and time.time() < deadline - 10:
        iteration += 1
        if iteration % 10000 == 0 and time.time() > deadline - 10:
            break

        # Choose a random move
        move_type = random.random()
        if move_type < 0.4:
            # Move: remove a random block
            if len(current_block_indices) <= 1:
                continue
            idx = random.randrange(len(current_block_indices))
            removed_block_idx = current_block_indices[idx]
            removed_ts = current_block_ts[idx]

            # Check if removal creates uncovered t-subsets
            new_uncovered = 0
            for ts_idx in removed_ts:
                if current_coverage[ts_idx] <= 1:
                    new_uncovered += 1

            # Apply removal
            for ts_idx in removed_ts:
                current_coverage[ts_idx] -= 1
            current_block_indices.pop(idx)
            current_block_ts.pop(idx)

            new_count = len(current_block_indices)
            # Cost change: delta_cost = new_uncovered * 1000 - 1 (block removed)
            delta_cost = new_uncovered * 1000 - 1

            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temp):
                if new_count < best_count and new_uncovered == 0:
                    best_count = new_count
                    best_block_indices = current_block_indices[:]
            else:
                # Revert
                current_block_indices.insert(idx, removed_block_idx)
                current_block_ts.insert(idx, removed_ts)
                for ts_idx in removed_ts:
                    current_coverage[ts_idx] += 1

        elif move_type < 0.8:
            # Move: add a random block
            if len(current_block_indices) > 300:  # Cap to avoid explosion
                continue
            new_block_idx = random.randrange(num_all_blocks)
            # Check if already in solution
            if new_block_idx in current_block_indices:
                continue

            new_ts = block_t_subsets_all[new_block_idx]
            # Count how many new t-subsets this covers
            new_covered = 0
            for ts_idx in new_ts:
                if current_coverage[ts_idx] == 0:
                    new_covered += 1

            # Apply addition
            for ts_idx in new_ts:
                current_coverage[ts_idx] += 1
            current_block_indices.append(new_block_idx)
            current_block_ts.append(new_ts)

            new_count = len(current_block_indices)
            delta_cost = 1 - new_covered * 1000

            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temp):
                pass  # Keep the move
            else:
                # Revert
                current_block_indices.pop()
                current_block_ts.pop()
                for ts_idx in new_ts:
                    current_coverage[ts_idx] -= 1

        else:
            # Move: swap a block with another
            if len(current_block_indices) <= 1:
                continue
            idx = random.randrange(len(current_block_indices))
            old_block_idx = current_block_indices[idx]
            old_ts = current_block_ts[idx]

            # Pick a random replacement block
            new_block_idx = random.randrange(num_all_blocks)
            if new_block_idx == old_block_idx:
                continue
            if new_block_idx in current_block_indices:
                continue

            new_ts = block_t_subsets_all[new_block_idx]

            # Compute delta: remove old, add new
            # First, what t-subsets become uncovered if we remove old?
            would_uncover = 0
            for ts_idx in old_ts:
                if current_coverage[ts_idx] <= 1:
                    would_uncover += 1

            # What t-subsets get newly covered by new?
            new_cover = 0
            for ts_idx in new_ts:
                # After removing old, coverage decreases by 1 for old's t-subsets
                # So effective coverage = current_coverage - (1 if in old_ts else 0)
                eff_cov = current_coverage[ts_idx]
                if ts_idx in old_ts:
                    eff_cov -= 1
                if eff_cov == 0:
                    new_cover += 1

            delta_cost = would_uncover * 1000 - new_cover * 1000

            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temp):
                # Apply swap
                for ts_idx in old_ts:
                    current_coverage[ts_idx] -= 1
                for ts_idx in new_ts:
                    current_coverage[ts_idx] += 1
                current_block_indices[idx] = new_block_idx
                current_block_ts[idx] = new_ts

                new_count = len(current_block_indices)
                # Check if all covered
                unc = 0
                for i in range(total_t_subsets):
                    if current_coverage[i] == 0:
                        unc += 1
                if unc == 0 and new_count < best_count:
                    best_count = new_count
                    best_block_indices = current_block_indices[:]
            # If rejected, do nothing (state unchanged)

        # Cool down
        temp *= cooling_rate

        # Periodically try to remove redundant blocks
        if iteration % 1000 == 0:
            # Try removing any redundant blocks
            for i in range(len(current_block_indices) - 1, -1, -1):
                redundant = True
                for ts_idx in current_block_ts[i]:
                    if current_coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in current_block_ts[i]:
                        current_coverage[ts_idx] -= 1
                    current_block_indices.pop(i)
                    current_block_ts.pop(i)
                    if len(current_block_indices) < best_count:
                        best_count = len(current_block_indices)
                        best_block_indices = current_block_indices[:]

    # Convert best_block_indices back to blocks
    result_blocks = [list(all_blocks_list[bi]) for bi in best_block_indices]
    result_blocks.sort()

    # Final validation and cleanup
    final_coverage = [0] * total_t_subsets
    final_block_ts = []
    for b in result_blocks:
        ts_list = []
        for ts in itertools.combinations(b, t):
            ts_idx = t_subset_to_idx[ts]
            ts_list.append(ts_idx)
            final_coverage[ts_idx] += 1
        final_block_ts.append(ts_list)

    # Remove any redundant blocks one more time
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(result_blocks) - 1, -1, -1):
            redundant = True
            for ts_idx in final_block_ts[i]:
                if final_coverage[ts_idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts_idx in final_block_ts[i]:
                    final_coverage[ts_idx] -= 1
                result_blocks.pop(i)
                final_block_ts.pop(i)
                improved = True

    return result_blocks
