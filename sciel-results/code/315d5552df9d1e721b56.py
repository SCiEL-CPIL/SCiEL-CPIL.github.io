"""Covering design C(v,k,t) with WalkSAT-style stochastic local search on Max-SAT encoding."""
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # Precompute all possible k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the set of t-subset indices it covers
    block_to_tsubs = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            indices.append(idx)
        block_to_tsubs.append(indices)

    # Reverse index: for each t-subset, list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for bi, indices in enumerate(block_to_tsubs):
        for idx in indices:
            tsub_to_blocks[idx].append(bi)

    def get_block_tsub_indices(block_tuple):
        """Return list of t-subset indices covered by a block (as tuple)."""
        indices = []
        for ts in itertools.combinations(block_tuple, t):
            idx = tsub_index[ts]
            indices.append(idx)
        return indices

    def validate_blocks(blocks):
        """Check if blocks form a valid covering. Returns True if valid."""
        if not blocks:
            return n_tsubs == 0
        cov = np.zeros(n_tsubs, dtype=np.int8)
        for block in blocks:
            for idx in get_block_tsub_indices(tuple(block)):
                cov[idx] += 1
        return cov.min() >= 1

    def greedy_cover(seed):
        """Fast greedy to get an initial feasible covering."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=np.int8)
        blocks_used = []
        used_block_indices = set()

        while True:
            if uncovered.sum() == 0:
                break
            if time.time() > deadline - 30:
                break

            uncovered_indices = np.where(uncovered == 1)[0]
            if len(uncovered_indices) == 0:
                break

            # Pick random uncovered t-subset, find best block for it
            ts_idx = rng.choice(uncovered_indices.tolist())
            covering_blocks = [bi for bi in tsub_to_blocks[ts_idx] if bi not in used_block_indices]
            if not covering_blocks:
                # Try all blocks as fallback
                covering_blocks = list(range(n_blocks))

            best_bi = -1
            best_gain = -1
            # Sample a few candidates
            sample = covering_blocks[:min(20, len(covering_blocks))]
            rng.shuffle(sample)
            for bi in sample:
                gain = sum(1 for idx in block_to_tsubs[bi] if uncovered[idx] == 1)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi

            if best_bi == -1:
                best_bi = covering_blocks[0]

            used_block_indices.add(best_bi)
            blocks_used.append(all_blocks[best_bi])
            for idx in block_to_tsubs[best_bi]:
                uncovered[idx] = 0

        return blocks_used

    def reduce_redundant(blocks):
        """Remove redundant blocks (those whose coverage is fully covered by others)."""
        if not blocks:
            return blocks
        blocks_list = list(blocks)
        block_tsubs_list = [get_block_tsub_indices(tuple(b)) for b in blocks_list]
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for indices in block_tsubs_list:
            for idx in indices:
                cov_count[idx] += 1

        improved = True
        while improved:
            improved = False
            for i in range(len(blocks_list)):
                redundant = True
                for idx in block_tsubs_list[i]:
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_tsubs_list[i]:
                        cov_count[idx] -= 1
                    blocks_list.pop(i)
                    block_tsubs_list.pop(i)
                    improved = True
        return blocks_list

    def walk_sat_optimize(blocks, seed, max_time=60):
        """
        WalkSAT-style local search.
        State: which blocks are selected (Boolean vector over n_blocks).
        Objective: minimize (penalty * unsatisfied + num_selected)
        where unsatisfied = number of t-subsets not covered by any selected block.
        """
        rng = random.Random(seed)
        end_time = time.time() + max_time

        # Initialize state from given blocks
        selected = np.zeros(n_blocks, dtype=np.int8)
        for block in blocks:
            bi = all_blocks.index(block) if block in all_blocks else -1
            if bi >= 0:
                selected[bi] = 1

        # If block not in all_blocks (shouldn't happen), just proceed
        # Build initial coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        for bi in range(n_blocks):
            if selected[bi] == 1:
                for idx in block_to_tsubs[bi]:
                    cov_count[idx] += 1

        unsatisfied = (cov_count == 0).sum()
        num_selected = selected.sum()

        penalty = 1000  # Heavy penalty for unsatisfied clauses
        current_score = penalty * unsatisfied + num_selected
        best_score = current_score
        best_selected = selected.copy()

        # Precompute for each block, the t-subsets it covers (for fast gain/loss)
        # Also precompute: for each t-subset, the list of blocks covering it

        # WalkSAT loop
        while time.time() < end_time:
            if unsatisfied == 0:
                # All satisfied, just try to remove blocks
                # Try to remove a random selected block
                selected_indices = np.where(selected == 1)[0]
                if len(selected_indices) == 0:
                    break

                for _ in range(50):  # Try 50 removals per iteration when satisfied
                    if time.time() > end_time:
                        break
                    if len(selected_indices) == 0:
                        break
                    bi = rng.choice(selected_indices.tolist())
                    if selected[bi] == 0:
                        continue

                    # Check if removing block bi leaves all covered
                    # A block can be removed iff for every t-subset it covers,
                    # there is another selected block covering it
                    can_remove = True
                    for idx in block_to_tsubs[bi]:
                        if cov_count[idx] <= 1:
                            can_remove = False
                            break

                    if can_remove:
                        selected[bi] = 0
                        for idx in block_to_tsubs[bi]:
                            cov_count[idx] -= 1
                        num_selected -= 1
                        current_score = penalty * unsatisfied + num_selected
                        if current_score < best_score:
                            best_score = current_score
                            best_selected = selected.copy()
                        # Update selected_indices
                        selected_indices = np.where(selected == 1)[0]
                    else:
                        # Can't remove, try flipping a different block
                        pass

                continue

            # Unsatisfied > 0: WalkSAT move
            # Pick a random unsatisfied t-subset
            unsatisfied_indices = np.where(cov_count == 0)[0]
            ts_idx = rng.choice(unsatisfied_indices.tolist())

            # Blocks that cover this t-subset
            candidate_blocks = tsub_to_blocks[ts_idx]

            # For each candidate, compute the delta if we add it
            # Adding block bi:
            #   - num_selected increases by 1 (if not already selected)
            #   - For each t-subset idx in block_to_tsubs[bi]:
            #     - if cov_count[idx] was 0, unsatisfied decreases by 1
            #     - if cov_count[idx] was > 0, no change in unsatisfied
            #   - If block was already selected, we're just flipping it off? No, we add.

            # Actually, let's think of this as: we want to flip a variable.
            # If unsatisfied > 0, we should prefer to ADD a block that covers an unsatisfied clause.
            # If we want to remove a block, we should only do so if it doesn't cause new unsatisfied.

            # Strategy: with some probability, add a block covering a random unsatisfied clause.
            # With some probability, try to remove a block that doesn't hurt.

            if rng.random() < 0.7:
                # Add a block
                # Pick from candidates that are not yet selected
                unselected_candidates = [bi for bi in candidate_blocks if selected[bi] == 0]
                if not unselected_candidates:
                    # All blocks covering this clause are already selected, so this clause should be covered
                    # This is a bug, but let's just pick a random unselected block
                    all_unselected = np.where(selected == 0)[0]
                    if len(all_unselected) == 0:
                        break
                    bi = rng.choice(all_unselected.tolist())
                else:
                    # Score each candidate by how many new t-subsets it covers
                    best_bi = -1
                    best_gain = -1
                    sample = unselected_candidates[:min(10, len(unselected_candidates))]
                    rng.shuffle(sample)
                    for bi in sample:
                        gain = 0
                        for idx in block_to_tsubs[bi]:
                            if cov_count[idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_bi = bi
                    bi = best_bi

                # Add block bi
                selected[bi] = 1
                num_selected += 1
                for idx in block_to_tsubs[bi]:
                    old_cov = cov_count[idx]
                    cov_count[idx] += 1
                    if old_cov == 0:
                        unsatisfied -= 1

                current_score = penalty * unsatisfied + num_selected
                if current_score < best_score:
                    best_score = current_score
                    best_selected = selected.copy()

            else:
                # Try to remove a block that doesn't increase unsatisfied
                selected_indices = np.where(selected == 1)[0]
                if len(selected_indices) == 0:
                    continue
                bi = rng.choice(selected_indices.tolist())

                # Check if removing is safe
                can_remove = True
                for idx in block_to_tsubs[bi]:
                    if cov_count[idx] <= 1:
                        can_remove = False
                        break

                if can_remove:
                    selected[bi] = 0
                    num_selected -= 1
                    for idx in block_to_tsubs[bi]:
                        cov_count[idx] -= 1
                    current_score = penalty * unsatisfied + num_selected
                    if current_score < best_score:
                        best_score = current_score
                        best_selected = selected.copy()
                else:
                    # Random restart-like: flip a random block
                    # Add a random unselected block
                    all_unselected = np.where(selected == 0)[0]
                    if len(all_unselected) > 0:
                        bi = rng.choice(all_unselected.tolist())
                        selected[bi] = 1
                        num_selected += 1
                        for idx in block_to_tsubs[bi]:
                            old_cov = cov_count[idx]
                            cov_count[idx] += 1
                            if old_cov == 0:
                                unsatisfied -= 1
                        current_score = penalty * unsatisfied + num_selected
                        if current_score < best_score:
                            best_score = current_score
                            best_selected = selected.copy()

        # Return the best state found
        best_blocks = []
        for bi in range(n_blocks):
            if best_selected[bi] == 1:
                best_blocks.append(all_blocks[bi])
        return best_blocks

    # Main search
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Get initial feasible coverings with greedy
    initial_blocks = None
    for seed in range(5):
        if time.time() > deadline - 40:
            break
        blocks = greedy_cover(seed)
        if validate_blocks(blocks):
            if initial_blocks is None or len(blocks) < len(initial_blocks):
                initial_blocks = blocks

    if initial_blocks is None:
        # Fallback: just return a greedy cover
        blocks = greedy_cover(0)
        return blocks

    # Reduce initial
    initial_blocks = reduce_redundant(initial_blocks)
    best_blocks = initial_blocks
    best_count = len(initial_blocks)

    # Phase 2: WalkSAT optimization with multiple seeds and time budgets
    remaining_time = deadline - time.time()
    n_walksat_runs = max(1, int(remaining_time / 15))  # ~15s per run

    for run_idx in range(n_walksat_runs):
        if time.time() > deadline - 10:
            break

        # Use a copy of the best blocks as starting point
        start_blocks = list(best_blocks)
        # Occasionally perturb the starting point
        if run_idx > 0 and rng.random() < 0.3 if 'rng' in dir() else random.random() < 0.3:
            # Remove a random block and see if still valid
            if len(start_blocks) > 1:
                idx = random.randrange(len(start_blocks))
                perturbed = start_blocks[:idx] + start_blocks[idx+1:]
                if validate_blocks(perturbed):
                    start_blocks = perturbed

        time_budget = min(15, deadline - time.time() - 5)
        if time_budget < 2:
            break

        optimized = walk_sat_optimize(start_blocks, seed=run_idx * 1000 + 42, max_time=time_budget)
        if optimized and validate_blocks(optimized):
            optimized = reduce_redundant(optimized)
            if len(optimized) < best_count:
                best_blocks = optimized
                best_count = len(optimized)

    # Final reduction
    if best_blocks is not None and time.time() < deadline - 2:
        best_blocks = reduce_redundant(best_blocks)

    if best_blocks is None:
        best_blocks = greedy_cover(0)

    return best_blocks
