import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)

    # Precompute: for each block, the list of t-subset indices it covers
    block_cover = []
    # For each t-subset, list of block indices covering it
    t_to_blocks = [[] for _ in range(N_t)]

    for bi, b in enumerate(all_blocks):
        covered = []
        for ts in itertools.combinations(b, t):
            ti = t_to_idx[ts]
            covered.append(ti)
        block_cover.append(covered)
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Bitmask for each block
    block_masks = [0] * N_b
    for bi, covered in enumerate(block_cover):
        mask = 0
        for ti in covered:
            mask |= (1 << ti)
        block_masks[bi] = mask

    full_mask = (1 << N_t) - 1

    def is_valid(blocks):
        if not blocks:
            return N_t == 0
        union_mask = 0
        for bi in blocks:
            union_mask |= block_masks[bi]
        return (union_mask & full_mask) == full_mask

    def prune_redundant(blocks):
        if not blocks:
            return []
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        current_blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for bi in current_blocks:
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(bi)
                else:
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    changed = True
            current_blocks = new_blocks
        return current_blocks

    def dual_weighted_construction(rng, time_limit):
        """
        Dual-weighted block selection:
        - Maintain dual prices y[j] for each t-subset j.
        - At each step, select the block B maximizing sum_{j in B, j uncovered} y[j].
        - After selecting B, increase y[j] for all j in B (and for uncovered j not in B).
        - This mimics the primal-dual / Hungarian assignment heuristic.
        """
        y = np.ones(N_t, dtype=np.float64)  # initial dual prices
        uncovered_mask = full_mask
        selected = []
        uncovered_count = N_t

        # Precompute: for each block, the array of t-indices it covers (as numpy for speed)
        block_cover_arr = [np.array(bc, dtype=np.int32) for bc in block_cover]

        while uncovered_count > 0:
            if time.time() > time_limit:
                break

            # Find the block that maximizes sum of y over its uncovered t-subsets
            best_block = -1
            best_score = -1.0

            # To speed up, we can sample a subset of blocks if N_b is large
            # But for 75582 blocks, let's try all. Each block has 56 t-subsets.
            # 75582 * 56 = ~4.2M operations per step. If we do ~300 steps, that's 1.26B.
            # Too slow in pure Python. Let's use numpy vectorization.

            # For each block, compute the sum of y over its uncovered t-subsets.
            # We need to mask y with the uncovered status.
            # Let's create a numpy array of y and for each block, sum the relevant entries.

            # Optimization: precompute block_cover as a 2D numpy array?
            # N_b x 56 array. 75582 * 56 * 4 bytes = ~17 MB. Fine.
            # But we need to handle the "uncovered" mask.

            # Alternative: iterate over uncovered t-subsets and for each, look at blocks covering it.
            # But that's also expensive.

            # Let's try a hybrid: if uncovered_count is small, iterate over uncovered t's.
            # Otherwise, iterate over blocks.

            if uncovered_count < 500:
                # Iterate over uncovered t-subsets
                uncovered_list = []
                for ti in range(N_t):
                    if (uncovered_mask >> ti) & 1:
                        uncovered_list.append(ti)

                # Score for each block: sum of y[ti] for ti in block_cover[block] & uncovered
                # Initialize scores
                scores = np.zeros(N_b, dtype=np.float64)
                for ti in uncovered_list:
                    for bi in t_to_blocks[ti]:
                        scores[bi] += y[ti]

                best_block = int(np.argmax(scores))
                best_score = scores[best_block]
            else:
                # Iterate over blocks using numpy
                # For each block, sum y over its t-subsets that are uncovered
                # We'll do this in a loop but with numpy for the sum
                scores = np.zeros(N_b, dtype=np.float64)
                for bi in range(N_b):
                    # Get the t-subsets covered by this block
                    bc = block_cover_arr[bi]
                    # Check which are uncovered
                    # We can use the bitmask: for each ti in bc, check if bit is set
                    # This is slow. Let's use a different approach.
                    # Precompute a numpy array of uncovered status
                    # Actually, let's just sum y[bc] and then subtract covered ones?
                    # No, we need sum over uncovered only.

                    # Let's just do a Python loop for the 56 elements
                    s = 0.0
                    for ti in bc:
                        if (uncovered_mask >> ti) & 1:
                            s += y[ti]
                    scores[bi] = s

                best_block = int(np.argmax(scores))
                best_score = scores[best_block]

            if best_block == -1 or best_score <= 0:
                # No good block found, break
                break

            # Select this block
            selected.append(best_block)

            # Update uncovered_mask
            uncovered_mask &= ~block_masks[best_block]

            # Update uncovered_count
            # Count bits in uncovered_mask
            uncovered_count = uncovered_mask.bit_count()

            # Update dual prices:
            # Increase y for all t-subsets in the selected block
            # Also, increase y for uncovered t-subsets not in the block (to make them more "urgent")
            # Standard primal-dual: increase y for all uncovered elements at rate 1.
            # Here, we'll increase y for all uncovered t-subsets by a small amount,
            # and for those in the selected block, by a larger amount.

            # Actually, in primal-dual, when a block becomes tight, we fix y for its elements.
            # Let's do: for all uncovered t-subsets, increase y by 1.0.
            # For t-subsets in the selected block that were uncovered, increase by an extra 0.5.

            for ti in range(N_t):
                if (uncovered_mask >> ti) & 1:
                    y[ti] += 1.0

            for ti in block_cover_arr[best_block]:
                # If this ti was just covered (now in covered_mask), we still increase y a bit
                # to reflect that this block was chosen for it
                y[ti] += 0.5

            # Normalize y to prevent overflow
            y_max = np.max(y)
            if y_max > 1000:
                y /= y_max

        return selected

    def repair(blocks, rng, time_limit):
        """
        Repair: add blocks to cover any remaining uncovered t-subsets.
        """
        if not blocks:
            return blocks

        # Compute uncovered
        union_mask = 0
        for bi in blocks:
            union_mask |= block_masks[bi]
        uncovered_mask = full_mask & ~union_mask

        if uncovered_mask == 0:
            return blocks

        blocks = list(blocks)
        uncovered_count = uncovered_mask.bit_count()

        while uncovered_count > 0:
            if time.time() > time_limit:
                break

            # Find an uncovered t-subset
            ti_target = -1
            for ti in range(N_t):
                if (uncovered_mask >> ti) & 1:
                    ti_target = ti
                    break

            if ti_target == -1:
                break

            # Find the best block covering ti_target
            best_block = -1
            best_gain = -1
            for bi in t_to_blocks[ti_target]:
                if bi in blocks:
                    continue
                inter = block_masks[bi] & uncovered_mask
                gain = inter.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block != -1:
                blocks.append(best_block)
                uncovered_mask &= ~block_masks[best_block]
                uncovered_count = uncovered_mask.bit_count()
            else:
                break

        return blocks

    def local_search(blocks, rng, time_limit):
        """
        Local search: try to remove redundant blocks and swap blocks to reduce count.
        """
        if not blocks:
            return blocks

        blocks = list(blocks)
        best_blocks = list(blocks)
        best_len = len(blocks)

        # Compute counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1

        while time.time() < time_limit:
            improved = False

            # Try to remove a redundant block
            for i in range(len(blocks)):
                bi = blocks[i]
                # Check if any t-subset is uniquely covered by bi
                is_redundant = True
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove bi
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    blocks.pop(i)
                    best_blocks = list(blocks)
                    best_len = len(blocks)
                    improved = True
                    break

            if improved:
                continue

            # Try a swap: remove one block, add another that covers its unique parts
            if len(blocks) > 1:
                i = rng.randrange(len(blocks))
                bi_out = blocks[i]
                unique_ts = [ti for ti in block_cover[bi_out] if counts[ti] == 1]

                if unique_ts:
                    # Find a block that covers all unique_ts
                    candidates = set(t_to_blocks[unique_ts[0]])
                    for ti in unique_ts[1:]:
                        candidates &= set(t_to_blocks[ti])
                    candidates.discard(bi_out)
                    candidates.difference_update(set(blocks))

                    if candidates:
                        bi_in = rng.choice(list(candidates))
                        # Apply swap
                        for ti in block_cover[bi_out]:
                            counts[ti] -= 1
                        for ti in block_cover[bi_in]:
                            counts[ti] += 1
                        blocks[i] = bi_in
                        # Check if we can now remove some blocks
                        # (this is handled in the next iteration)
                        improved = True

            if not improved:
                # No improvement possible, try random perturbation
                if len(blocks) > 0:
                    i = rng.randrange(len(blocks))
                    bi_out = blocks[i]
                    # Remove it temporarily
                    for ti in block_cover[bi_out]:
                        counts[ti] -= 1
                    blocks.pop(i)

                    # Try to repair
                    temp_blocks = repair(blocks, rng, time_limit)
                    if len(temp_blocks) < best_len:
                        blocks = temp_blocks
                        best_blocks = list(blocks)
                        best_len = len(blocks)
                        # Recompute counts
                        counts = np.zeros(N_t, dtype=np.int32)
                        for bi in blocks:
                            for ti in block_cover[bi]:
                                counts[ti] += 1
                    else:
                        # Restore
                        blocks.insert(i, bi_out)
                        for ti in block_cover[bi_out]:
                            counts[ti] += 1

        return best_blocks

    # Main execution
    best_solution = None
    best_len = float('inf')

    # Phase 1: Dual-weighted construction with multiple restarts
    num_starts = 3
    for i in range(num_starts):
        if time.time() > deadline - 80.0:
            break
        rng = random.Random(42 + i * 1000)
        time_left = deadline - time.time() - 10.0
        if time_left < 5.0:
            break

        blocks = dual_weighted_construction(rng, deadline - 10.0)

        # Repair
        blocks = repair(blocks, rng, deadline - 5.0)

        # Prune
        blocks = prune_redundant(blocks)

        if is_valid(blocks) and len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks

    # Phase 2: Local search on the best solution
    if best_solution is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            rng = random.Random(999)
            optimized = local_search(best_solution, rng, deadline - 5.0)
            if is_valid(optimized) and len(optimized) < best_len:
                best_len = len(optimized)
                best_solution = optimized

    # Fallback: simple greedy if nothing worked
    if best_solution is None:
        rng = random.Random(1)
        uncovered_mask = full_mask
        selected = []
        while uncovered_mask != 0 and time.time() < deadline - 2.0:
            # Find an uncovered t-subset
            ti_target = -1
            for ti in range(N_t):
                if (uncovered_mask >> ti) & 1:
                    ti_target = ti
                    break
            if ti_target == -1:
                break

            best_block = -1
            best_gain = -1
            for bi in t_to_blocks[ti_target]:
                inter = block_masks[bi] & uncovered_mask
                gain = inter.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi

            if best_block != -1:
                selected.append(best_block)
                uncovered_mask &= ~block_masks[best_block]

        best_solution = prune_redundant(selected)

    return [list(all_blocks[bi]) for bi in best_solution]
