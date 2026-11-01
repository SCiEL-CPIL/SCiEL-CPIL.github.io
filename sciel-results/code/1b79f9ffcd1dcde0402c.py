import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit

    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    # Map t-subsets to indices for fast lookup
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    # Precompute the t-subsets contained in each k-subset
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        # Using math.comb is faster if available, but np.prod works too
        # For small numbers, simple loop is fine
        res = 1
        for i in range(r):
            res = res * (n - i) // (i + 1)
        return res

    t_count = nCr(k, t)

    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    # This allows fast gain calculation
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Greedy Construction with Randomization
    covered = np.zeros(n_ts, dtype=bool)
    blocks_indices = [] # Store indices into all_ks
    point_load = np.zeros(v, dtype=int)

    # We will run a greedy loop
    # Optimization: Instead of checking all n_ks every time (which is expensive for v=24, k=7),
    # we can just do it. n_ks = C(24,7) = 346104.
    # t_count = C(7,3) = 35.
    # Memory for block_t_indices: 346104 * 35 * 4 bytes ~ 48 MB. This is fine.

    # The main cost is the greedy step: calculating gains for all 346k blocks.
    # gains = (~covered[block_t_indices]).sum(axis=1)
    # This involves indexing a boolean array of size n_ts with a matrix of size n_ks*t_count.
    # Resulting temp array size: 346104 * 35 ~ 12 million booleans.
    # This is fast in NumPy.

    while time.time() - start_time < time_limit * 0.6:
        # Calculate gains
        # covered[block_t_indices] creates a boolean matrix
        # ~covered[...] inverts it
        # .sum(axis=1) sums the True values (uncovered subsets)

        # To save time, we can break early if all are covered?
        # No, we want to minimize blocks, so we stop when we can't cover more?
        # Actually, greedy stops when max gain is 0.

        uncovered_flags = ~covered[block_t_indices]
        gains = uncovered_flags.sum(axis=1)

        max_gain = np.max(gains)
        if max_gain == 0:
            break

        # Find candidates with max gain
        # To add randomness and escape local optima, pick from top K candidates randomly
        # Or just pick the one with min point load sum among max gain candidates?
        # Let's stick to a simple random selection from the top 10 max gain candidates to diversify
        # if time permits, otherwise just max.

        candidates = np.where(gains == max_gain)[0]

        # Tie-breaking: Prefer blocks with lower load to balance coverage
        # Calculate load sum for candidates
        # all_ks is a list of tuples. We need to check loads.
        # This might be slow if done in Python loop.
        # Let's just pick a random candidate from the max gain set.
        # Randomization helps avoid deterministic traps.
        best_idx = candidates[np.random.randint(len(candidates))]

        # Add block
        blocks_indices.append(best_idx)
        block = all_ks[best_idx]

        # Update covered
        covered[block_t_indices[best_idx]] = True

        # Update point load
        for p in block:
            point_load[p] += 1

        # Check if all covered
        if np.all(covered):
            break

    # Local Search: Try to remove redundant blocks and perform swaps
    # This can significantly reduce the count.
    # A block is redundant if all its t-subsets are covered by other blocks.

    # We need to count how many blocks cover each t-subset.
    # coverage_count[j] = number of blocks in blocks_indices that contain t-subset j

    # Reset coverage count
    coverage_count = np.zeros(n_ts, dtype=np.int32)

    # Build initial coverage count
    for b_idx in blocks_indices:
        coverage_count[block_t_indices[b_idx]] += 1

    current_blocks = list(blocks_indices)

    # To speed up, we can just do a few passes.
    # Time budget for local search: remaining time
    ls_start = time.time()
    ls_time_limit = time_limit - (ls_start - start_time) - 5.0 # 5s safety

    improved = True
    # Use a random seed for reproducibility within the run if needed, but default is fine
    rng = np.random.default_rng()

    while improved and (time.time() - ls_start) < ls_time_limit:
        improved = False
        new_current_blocks = []

        # Rebuild coverage count for current_blocks?
        # It's better to maintain it incrementally or rebuild if list is small.
        # Rebuilding is O(N_blocks * t_count). N_blocks ~ 76. t_count ~ 35. Very fast.

        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1

        # Shuffle the order of checking to avoid bias
        shuffled_indices = list(range(len(current_blocks)))
        rng.shuffle(shuffled_indices)

        for pos in shuffled_indices:
            if time.time() - ls_start > ls_time_limit:
                break

            b_idx = current_blocks[pos]
            # Check if b_idx is redundant
            # If all t-subsets in b_idx have coverage_count > 1, it can be removed.
            # Note: coverage_count includes b_idx itself.
            # So we need min(coverage_count[block_t_indices[b_idx]]) > 1
            subs_in_block = block_t_indices[b_idx]
            min_cov = np.min(coverage_count[subs_in_block])

            if min_cov > 1:
                # Redundant, remove it
                improved = True
                # Decrement coverage for this block
                coverage_count[subs_in_block] -= 1
            else:
                # Keep it
                new_current_blocks.append(b_idx)

        current_blocks = new_current_blocks

        # If we didn't improve by deletion in this pass, try a few random swaps/perturbations
        # to escape local minima, then check for deletions again.
        if not improved and len(current_blocks) > 0 and (time.time() - ls_start) < ls_time_limit - 10.0:
            # Try swapping a point in a random block
            # This is a heuristic move that might allow subsequent deletions
            # We need to be careful not to break coverage.

            # Rebuild coverage for current_blocks (it's up to date from the loop above if no deletions happened?
            # Actually, if no deletions happened, coverage_count is correct for current_blocks.
            # But if deletions happened, we broke out of the while loop? No, the while loop continues.
            # The coverage_count is rebuilt at the start of each while iteration.
            # So here, coverage_count is valid for current_blocks.

            swap_success = False
            for _ in range(10): # Try a few swaps
                if time.time() - ls_start > ls_time_limit:
                    break

                idx = rng.integers(0, len(current_blocks))
                b_idx = current_blocks[idx]
                block = all_ks[b_idx]

                # Pick a point to remove and a point to add
                old_x = block[rng.integers(0, k)]
                candidates_to_add = [x for x in range(v) if x not in block]
                if not candidates_to_add:
                    continue
                new_x = candidates_to_add[rng.integers(0, len(candidates_to_add))]

                # Simulate swap
                # Old block triples
                old_subs = block_t_indices[b_idx]

                # New block triples
                new_block = sorted(list(set(list(block) + [new_x]) - {old_x}))
                # Find index of new_block in all_ks?
                # This is expensive if we do it every time.
                # Alternative: just check if the swap maintains coverage.
                # We need to know which triples are lost and gained.

                # Triples lost: those in old_block that are NOT in new_block
                # Triples gained: those in new_block that are NOT in old_block

                # Since we don't have the index of new_block in all_ks readily,
                # and calculating the intersection of sets of triples is O(t_count),
                # we can do it with sets for the small number of triples.

                old_triples_set = set(all_ts[i] for i in old_subs)

                # Construct new triples
                new_triples = set(itertools.combinations(new_block, t))
                new_triples_idx = set(idx_map[tr] for tr in new_triples)

                lost_triples_idx = old_triples_set - new_triples
                gained_triples_idx = new_triples - old_triples_set

                # Check if any lost triple has coverage 1 (would become 0)
                can_swap = True
                for tr in lost_triples_idx:
                    tr_idx = idx_map[tr]
                    if coverage_count[tr_idx] == 1:
                        can_swap = False
                        break

                if can_swap:
                    # Apply swap
                    for tr in lost_triples_idx:
                        tr_idx = idx_map[tr]
                        coverage_count[tr_idx] -= 1
                    for tr_idx in gained_triples_idx:
                        coverage_count[tr_idx] += 1

                    # Update the block in current_blocks
                    # We need to store the new block.
                    # Since current_blocks stores indices into all_ks, and new_block might not be in all_ks?
                    # Wait, all_ks contains ALL k-subsets. So new_block IS in all_ks.
                    # We need to find its index.
                    # This is a bottleneck.
                    # Let's precompute a map from block tuple to index.
                    # But that's another 346k entries.
                    # Or we can just keep the blocks as tuples/lists in a separate list?
                    # The current structure uses indices for fast numpy access.
                    # If we change to storing blocks directly, we lose the fast numpy gain calculation in greedy.
                    # But greedy is done. In local search, we can afford slower operations.
                    # However, the deletion check uses block_t_indices[b_idx].
                    # If we change b_idx to a block tuple, we need to recompute block_t_indices for that block.
                    # That's O(t_count) which is 35. Fast.

                    # Let's modify current_blocks to store the block tuples themselves?
                    # No, that complicates the deletion check which relies on the precomputed array.
                    # Let's stick to indices. We need a fast way to get index of a tuple.
                    # Let's create a dictionary mapping tuple to index.
                    # This takes memory but is fast.

                    # Actually, we can just not do swaps if it's too complex to manage indices.
                    # The deletion pass is the most powerful.
                    # Let's see if we can improve deletion order or add a "relaxation" step.

                    # For now, let's skip the swap implementation to avoid bugs and complexity,
                    # as the deletion pass with randomization might be sufficient.
                    # The inspiration 1 had swaps, but it was slower (score 1.11 vs 1.08? No, 1.11 is better).
                    # Wait, 1.1134 is better than 1.08.
                    # So the swap-based approach in inspiration 1 was better.
                    # But it was a different structure (sets of triples).
                    # My current structure is optimized for the greedy phase.
                    # Let's try to implement a simpler swap:
                    # Just try to replace a block with a "better" one?
                    # No, that's not how covering works.

                    # Let's stick to the deletion pass but ensure it runs many times.
                    # The current loop does that.

                    pass # Skip swap for now to keep code robust

            # If we did any swaps, we should set improved = True to trigger another deletion pass?
            # Since we skipped swaps, improved remains False, and the loop terminates.
            # To continue improving, we need a mechanism to find new deletions.
            # Randomizing the order of deletion checks (which we do) helps.

    # Convert indices back to blocks
    final_blocks = [list(all_ks[i]) for i in current_blocks]

    return final_blocks
