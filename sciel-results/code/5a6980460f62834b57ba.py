import itertools
import random
import sys
import time

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using column generation with a
    pricing subproblem that maximizes uncovered coverage per unit weight.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    full_mask = (1 << n_t) - 1

    # Precompute all blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)

    def popcount(x):
        return x.bit_count()

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = 0
        uncovered = full_mask
        solution = []

        while uncovered:
            best_gain = 0
            candidates = []
            max_possible = popcount(uncovered)

            for i, mask in enumerate(block_masks):
                gain = popcount(mask & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                    if gain == max_possible:
                        break
                elif gain == best_gain and gain > 0:
                    candidates.append(i)

            if not candidates:
                break

            if len(candidates) == 1:
                best_block_idx = candidates[0]
            else:
                best_block_idx = random.choice(candidates)

            solution.append(all_blocks[best_block_idx])
            covered |= block_masks[best_block_idx]
            uncovered = full_mask ^ covered

        return solution

    def optimize_solution(sol):
        t_indices_cache = {}
        for block in sol:
            t_indices_cache[block] = [t_to_idx[ts] for ts in itertools.combinations(block, t)]

        counts = [0] * n_t
        for block in sol:
            for idx in t_indices_cache[block]:
                counts[idx] += 1

        improved = True
        while improved:
            improved = False
            n_sol = len(sol)
            for i in range(n_sol):
                block = sol[i]
                t_indices = t_indices_cache[block]
                is_redundant = True
                for idx in t_indices:
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for idx in t_indices:
                        counts[idx] -= 1
                    sol.pop(i)
                    improved = True
                    break
        return sol

    # Column generation framework
    def column_generation(seed=None):
        if seed is not None:
            random.seed(seed)

        # Start with a greedy solution
        sol = greedy_construct(seed)
        sol = optimize_solution(sol)

        # Track which blocks are in the current master
        master_blocks = set(sol)
        master_mask = 0
        for block in sol:
            idx = all_blocks.index(block)
            master_mask |= block_masks[idx]

        uncovered = full_mask ^ master_mask

        # Pricing subproblem: find new blocks that cover uncovered t-subsets
        # Iterate to add columns
        for _ in range(20):  # Max iterations
            if not uncovered:
                break

            # Find blocks that cover the most uncovered t-subsets
            best_gain = 0
            best_candidates = []

            for i, block in enumerate(all_blocks):
                if block in master_blocks:
                    continue
                gain = popcount(block_masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)

            if not best_candidates or best_gain == 0:
                break

            # Randomly select a new block to add
            new_idx = random.choice(best_candidates)
            new_block = all_blocks[new_idx]
            master_blocks.add(new_block)
            sol.append(new_block)
            master_mask |= block_masks[new_idx]
            uncovered = full_mask ^ master_mask

        # Optimize the resulting solution
        sol = optimize_solution(sol)
        return sol

    # Initial deterministic run
    best_sol = optimize_solution(greedy_construct(42))
    best_len = len(best_sol)

    # Column generation with different seeds
    start_time = time.time()
    seed = 1
    while time.time() - start_time < 95:
        seed += 1
        sol = column_generation(seed)
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol

    return best_sol
