"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def construct(v, k, t):
    rng = random.Random(7)
    start_time = time.time()
    time_limit = 117.0  # safety margin under 150s

    # Precompute all t-subsets and index them
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}

    # For each point, the list of t-subset indices containing it
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(idx)

    # Precompute for each block (tuple of k points) its covered subset indices
    def block_mask(block):
        covered = []
        for c in itertools.combinations(block, t):
            covered.append(subset_to_idx[frozenset(c)])
        return covered

    # ---- Greedy initial covering ----
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()

    while uncovered:
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_local = set(block)

        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_local]
            rng.shuffle(cands)
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx[fs]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_local.add(best_x)

        block.sort()
        bt = tuple(block)
        if bt not in block_set:
            block_set.add(bt)
            blocks.append(bt)
            for c in itertools.combinations(block, t):
                uncovered.discard(subset_to_idx[frozenset(c)])

    # ---- State: counts array, block masks ----
    counts = [0] * n_subsets
    block_masks = []
    for b in blocks:
        m = block_mask(b)
        block_masks.append(m)
        for idx in m:
            counts[idx] += 1

    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    no_improve = 0
    max_no_improve = 400

    def try_remove_block():
        """Try to remove a redundant block. Returns True if removed."""
        nonlocal blocks, block_masks, counts, best_blocks, best_count, no_improve
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_masks[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_masks[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_masks.pop(i)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                no_improve = 0
                return True
        return False

    def try_swap_block():
        """Try to swap one block for a better one. Returns True if swapped."""
        nonlocal blocks, block_masks, counts, no_improve
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_mask = block_masks[i]

        # Find critical subsets (count == 1) in this block
        critical = [idx for idx in old_mask if counts[idx] == 1]
        if not critical:
            return False

        # Collect points that must be in any replacement
        points_needed = set()
        for idx in critical:
            points_needed.update(all_subsets[idx])
        if len(points_needed) > k:
            return False

        # Build a new block containing points_needed, greedily adding points
        # that maximize coverage of low-count subsets
        new_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)

        for x in cands:
            if len(new_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_set, t - 1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx[fs]
                if counts[idx] == 0:
                    gain += 10
                elif counts[idx] == 1:
                    gain += 1
            new_set.add(x)

        if len(new_set) != k:
            return False

        new_block = tuple(sorted(new_set))
        if new_block == old_block:
            return False

        new_mask = block_mask(new_block)

        # Validate: no subset loses its only cover
        new_set_mask = set(new_mask)
        for idx in old_mask:
            if idx not in new_set_mask and counts[idx] <= 1:
                return False

        # Apply swap
        for idx in old_mask:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1
        blocks[i] = new_block
        block_masks[i] = new_mask
        no_improve = 0
        return True

    def try_two_for_one():
        """Replace two blocks with one that covers their critical union.
        Returns True if successful."""
        nonlocal blocks, block_masks, counts, best_blocks, best_count, no_improve
        if len(blocks) < 2:
            return False
        i, j = rng.sample(range(len(blocks)), 2)
        b1 = blocks[i]
        b2 = blocks[j]
        m1 = block_masks[i]
        m2 = block_masks[j]

        # Critical subsets for each block
        crit1 = [idx for idx in m1 if counts[idx] == 1]
        crit2 = [idx for idx in m2 if counts[idx] == 1]

        # Points that must be in the new block
        points_needed = set()
        for idx in crit1:
            points_needed.update(all_subsets[idx])
        for idx in crit2:
            points_needed.update(all_subsets[idx])

        if len(points_needed) > k:
            return False

        # Greedily fill remaining slots to maximize coverage
        new_set = set(points_needed)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)

        for x in cands:
            if len(new_set) >= k:
                break
            gain = 0
            for rest in itertools.combinations(new_set, t - 1):
                fs = frozenset(list(rest) + [x])
                idx = subset_to_idx[fs]
                if counts[idx] == 0:
                    gain += 10
                elif counts[idx] == 1:
                    gain += 1
            new_set.add(x)

        if len(new_set) != k:
            return False

        new_block = tuple(sorted(new_set))
        if new_block in (b1, b2):
            return False

        new_mask = block_mask(new_block)
        new_mask_set = set(new_mask)

        # Validate: every critical subset of b1 and b2 is covered by new_block
        for idx in crit1:
            if idx not in new_mask_set:
                return False
        for idx in crit2:
            if idx not in new_mask_set:
                return False

        # Also validate: no other subset loses its only cover
        # Remove b1 and b2, add new_block; check all counts stay >= 1
        for idx in m1:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1

        # Check validity
        valid = True
        for idx in range(n_subsets):
            if counts[idx] < 0:
                valid = False
                break
        # Actually we need counts >= 0, but more importantly no subset goes to 0
        # that was >= 1 before. Let's check differently:
        # Revert and check properly
        for idx in new_mask:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] += 1
        for idx in m1:
            counts[idx] += 1

        # Proper check: simulate and verify no count drops below 0
        sim_counts = [0] * n_subsets  # too expensive; instead check critical
        # A subset is safe if it's covered by new_block OR has count >= 2 in (m1+m2)
        # For subsets in m1 but not new_block: need count in m2 >= 1 (i.e., in m2)
        # For subsets in m2 but not new_block: need count in m1 >= 1
        m1_set = set(m1)
        m2_set = set(m2)
        for idx in m1_set:
            if idx not in new_mask_set:
                # This subset was only in b1 (count==1 in m1 means it's critical)
                # If it's not in new_block, it must be in m2
                if idx not in m2_set:
                    valid = False
                    break
        if valid:
            for idx in m2_set:
                if idx not in new_mask_set:
                    if idx not in m1_set:
                        valid = False
                        break

        if not valid:
            return False

        # Apply the 2-for-1
        for idx in m1:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1

        # Remove blocks i and j (higher index first), add new block
        hi = max(i, j)
        lo = min(i, j)
        blocks.pop(hi)
        block_masks.pop(hi)
        blocks.pop(lo)
        block_masks.pop(lo)
        blocks.append(new_block)
        block_masks.append(new_mask)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        no_improve = 0
        return True

    def perturb():
        """Perturb: remove a random block and greedily re-cover.
        Returns True if valid covering restored."""
        nonlocal blocks, block_masks, counts, no_improve
        if len(blocks) <= 1:
            return False
        old_blocks = list(blocks)
        old_masks = list(block_masks)
        old_counts = list(counts)

        i = rng.randrange(len(blocks))
        b = blocks[i]
        m = block_masks[i]
        for idx in m:
            counts[idx] -= 1
        blocks.pop(i)
        block_masks.pop(i)

        # Find uncovered subsets
        unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        while unc_list:
            seed_idx = rng.choice(unc_list)
            seed = all_subsets[seed_idx]
            new_b = list(seed)
            rng.shuffle(new_b)
            new_local = set(new_b)

            while len(new_b) < k:
                best_x = None
                best_gain = -1
                cands = [x for x in range(v) if x not in new_local]
                rng.shuffle(cands)
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(new_b, t - 1):
                        fs = frozenset(list(rest) + [x])
                        idx = subset_to_idx[fs]
                        if counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_b.append(best_x)
                new_local.add(best_x)

            new_b.sort()
            new_bt = tuple(new_b)
            if new_bt not in blocks:
                new_m = block_mask(new_bt)
                for idx in new_m:
                    counts[idx] += 1
                blocks.append(new_bt)
                block_masks.append(new_m)
                # Recompute uncovered
                unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        if not unc_list:
            # Valid covering restored
            no_improve = 0
            return True
        else:
            # Revert
            blocks = old_blocks
            block_masks = old_masks
            counts = old_counts
            return False

    # Main search loop
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break

        # Try removals first (cheap and always beneficial)
        removed = try_remove_block()
        if removed:
            continue

        # Try swaps
        swapped = try_swap_block()
        if swapped:
            continue

        # Try 2-for-1 replacements (key move for escaping local minima)
        two_for_one = try_two_for_one()
        if two_for_one:
            continue

        # If stuck, perturb
        no_improve += 1
        if no_improve > max_no_improve:
            perturb()
            no_improve = 0
        else:
            # Small random perturbation: swap a block with a random one
            if blocks and rng.random() < 0.3:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_mask = block_masks[i]
                # Generate a random block
                new_block = tuple(sorted(rng.sample(range(v), k)))
                if new_block != old_block:
                    new_mask = block_mask(new_block)
                    # Check validity
                    valid = True
                    new_mask_set = set(new_mask)
                    for idx in old_mask:
                        if idx not in new_mask_set and counts[idx] <= 1:
                            valid = False
                            break
                    if valid:
                        for idx in old_mask:
                            counts[idx] -= 1
                        for idx in new_mask:
                            counts[idx] += 1
                        blocks[i] = new_block
                        block_masks[i] = new_mask
                    else:
                        no_improve += 1

    # Final cleanup: try removals one more time
    while True:
        removed = try_remove_block()
        if not removed:
            break

    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
