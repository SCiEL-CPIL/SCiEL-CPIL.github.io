import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their bitmask of covered t-subsets
    block_list = []
    block_masks = []

    # Precompute bit for each triple to speed up mask construction
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i

    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)

    n_blocks = len(block_list)
    full_mask = (1 << n_subsets) - 1

    if time.time() > deadline - 10.0:
        # Fallback: run a single fast greedy
        uncovered = full_mask
        selected = []
        used = set()
        while uncovered:
            best_gain = 0
            best_idx = -1
            for i in range(n_blocks):
                if i in used:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            selected.append(best_idx)
            used.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        return [list(block_list[i]) for i in selected]

    best_count = float('inf')
    best_blocks = None

    rng = random.Random(12345)

    def run_greedy_from(uncovered_mask, used_set, rng_seed=None):
        """Continue greedy from current state. Returns list of selected indices."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = rng
        selected = []
        uncovered = uncovered_mask
        used = set(used_set) if used_set else set()
        while uncovered:
            best_gain = 0
            best_candidates = []
            for i in range(n_blocks):
                if i in used:
                    continue
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            if not best_candidates:
                break
            chosen = local_rng.choice(best_candidates)
            selected.append(chosen)
            used.add(chosen)
            uncovered &= ~block_masks[chosen]
        return selected

    def local_search_add_drop(selected_indices, rng_obj, max_iters=500):
        """
        Local search: try adding one block and dropping up to two redundant blocks.
        Returns improved list or original if no improvement.
        """
        current = list(selected_indices)
        current_set = set(current)
        current_mask = 0
        for idx in current:
            current_mask |= block_masks[idx]

        # Verify it's a cover
        if current_mask != full_mask:
            # Not a valid cover, fill it greedily
            remaining = full_mask & ~current_mask
            fill = run_greedy_from(remaining, current_set, rng_obj.randint(0, 10**9))
            current = current + fill
            current_set = set(current)
            current_mask = full_mask

        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Compute coverage counts for each triple
            coverage = [0] * n_subsets
            for idx in current:
                block = block_list[idx]
                for sub in itertools.combinations(block, t):
                    coverage[subset_index[sub]] += 1

            # Try adding each unused block
            # We limit candidates to blocks that cover at least one triple with low coverage
            # to save time. Actually, for local search quality, we should try all,
            # but that's 346k * 70 blocks = 24M checks per iteration, too slow.
            # Instead, we sample or use a heuristic.
            # Better: Only consider blocks that are "close" to the current solution?
            # Or: Use a random sample of unused blocks.

            # Let's try a different approach:
            # 1. Pick a random unused block to add.
            # 2. After adding, try to remove redundant blocks.
            # 3. If the new set is smaller, accept.

            # To find good candidates for addition, we can look at blocks that
            # cover many triples that are currently covered by only 1 or 2 blocks.
            # But computing this for all blocks is expensive.

            # Alternative: Randomized local search.
            # Pick a random unused block. Add it. Then greedily remove redundant blocks.
            # If size decreases, keep. Else revert.

            # This is a "1-add, then cleanup" move.

            # Let's implement:
            # 1. Choose a random unused block index.
            # 2. Add it to current.
            # 3. Compute new coverage.
            # 4. Remove any block that is redundant (all its triples have coverage > 1 in the new set).
            # 5. If len(new_set) < len(current), accept.

            # To make this efficient, we don't recompute full coverage from scratch.
            # But for simplicity and given the small number of blocks (~70-80),
            # recomputing coverage for 70 blocks * 35 triples = 2450 ops is fast.

            # Let's try this random add-then-prune approach.

            # Pick a random unused block
            # We need a list of unused blocks.
            # Generating this list every iteration is O(n_blocks) = 346k.
            # 500 iters * 346k = 173M ops. Might be too slow in Python.
            # Let's precompute a list of all block indices and filter?
            # Or just pick a random index and check if it's used?
            # Probability of picking a used block is ~70/346k = 0.02%.
            # So we can just pick a random index in range(n_blocks) and check.

            new_candidate = rng_obj.randint(0, n_blocks - 1)
            if new_candidate in current_set:
                continue

            # Add it
            new_set = current_set | {new_candidate}
            new_list = list(new_set)

            # Compute coverage for new_set
            new_coverage = [0] * n_subsets
            for idx in new_list:
                block = block_list[idx]
                for sub in itertools.combinations(block, t):
                    new_coverage[subset_index[sub]] += 1

            # Remove redundant blocks
            # A block is redundant if all its triples have coverage >= 2 in new_set
            # (since we are checking if we can remove it while keeping coverage >= 1).
            # Wait, if we remove a block, the coverage of its triples decreases by 1.
            # So a block is safe to remove if for all its triples, coverage >= 2.
            # We want to remove as many as possible to minimize size.
            # Greedy removal:
            # Iterate through blocks, if removable, remove and update coverage.

            to_remove = []
            # We need to be careful: removing one block might make another removable.
            # So we iterate until no more removable blocks.

            changed = True
            while changed:
                changed = False
                for idx in list(new_list):
                    if idx not in new_set:
                        continue
                    block = block_list[idx]
                    removable = True
                    for sub in itertools.combinations(block, t):
                        if new_coverage[subset_index[sub]] <= 1:
                            removable = False
                            break
                    if removable:
                        new_set.discard(idx)
                        new_list.remove(idx)
                        for sub in itertools.combinations(block, t):
                            new_coverage[subset_index[sub]] -= 1
                        changed = True

            if len(new_list) < len(current):
                current = new_list
                current_set = new_set
                improved = True
            # If no improvement, we revert (current is unchanged)

        return current

    # Main loop: Multiple restarts with different strategies
    while time.time() < deadline - 2.0:
        strategy = rng.randint(0, 2)

        if strategy == 0:
            # Pure greedy with random tie-breaking
            selected = run_greedy_from(full_mask, set(), rng.randint(0, 10**9))
        elif strategy == 1:
            # Start with a random block, then greedy
            first_block = rng.randint(0, n_blocks - 1)
            initial_mask = block_masks[first_block]
            initial_set = {first_block}
            remaining = full_mask & ~initial_mask
            selected = [first_block] + run_greedy_from(remaining, initial_set, rng.randint(0, 10**9))
        else:
            # Start with 2 random blocks, then greedy
            b1 = rng.randint(0, n_blocks - 1)
            b2 = rng.randint(0, n_blocks - 1)
            if b1 == b2:
                b2 = (b2 + 1) % n_blocks
            initial_mask = block_masks[b1] | block_masks[b2]
            initial_set = {b1, b2}
            remaining = full_mask & ~initial_mask
            selected = [b1, b2] + run_greedy_from(remaining, initial_set, rng.randint(0, 10**9))

        # Apply local search
        if time.time() < deadline - 5.0:
            selected = local_search_add_drop(selected, rng, max_iters=100)

        # Update best
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [block_list[i] for i in selected]

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return []
