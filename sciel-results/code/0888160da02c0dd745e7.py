"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _block_mask_list(block, subset_to_idx):
    """Return the list of t-subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, 3):
        covered.append(subset_to_idx[frozenset(c)])
    return covered


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 145.0  # safety margin under 150s

    def time_left():
        return time_limit - (time.time() - start_time)

    # Precompute all t-subsets and index them
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}

    # For each point, the set of t-subset indices containing it (for fast gain)
    point_to_subsets = [set() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].add(idx)

    # Precompute all C(v,k) blocks and their masks for fast lookup
    # For v=24, k=7: C(24,7) = 346104 blocks, each with C(7,3)=35 subsets
    # This is feasible to precompute
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks_all = len(all_blocks)
    block_masks_all = [None] * n_blocks_all
    for i, b in enumerate(all_blocks):
        block_masks_all[i] = _block_mask_list(b, subset_to_idx)

    # For fast lookup: block tuple -> index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}

    def block_mask(block):
        return block_masks_all[block_to_idx[tuple(block)]]

    def block_mask_by_tuple(bt):
        return block_masks_all[block_to_idx[bt]]

    # ---- Greedy initial covering with scarcity weighting ----
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()
    # Track scarcity: for each point, how many uncovered subsets contain it
    point_scarcity = [0] * v
    for idx in uncovered:
        for p in all_subsets[idx]:
            point_scarcity[p] += 1

    while uncovered:
        # Pick a random uncovered subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_local = set(block)

        while len(block) < k:
            best_x = None
            best_score = -1
            cands = [x for x in range(v) if x not in block_local]
            for x in cands:
                # Score: prioritize points that cover many uncovered subsets
                # AND have high scarcity (appear in many uncovered subsets)
                gain = len(point_to_subsets[x] & uncovered)
                # Add scarcity weight to break ties and guide toward hard points
                score = gain + 0.1 * point_scarcity[x]
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_local.add(best_x)

            # Update scarcity for points in the new block
            for p in block:
                if p == best_x:
                    continue
                # Don't update here, we'll do it after adding the block

        block.sort()
        bt = tuple(block)
        if bt not in block_set:
            block_set.add(bt)
            blocks.append(bt)
            # Update uncovered and scarcity
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                if idx in uncovered:
                    uncovered.discard(idx)
                    for p in all_subsets[idx]:
                        point_scarcity[p] -= 1

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
    max_no_improve = 300
    T = 50.0
    T_min = 0.1
    cooling = 0.9998

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
        nonlocal blocks, block_masks, counts, no_improve, T
        if not blocks:
            return False
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        old_mask = block_masks[i]

        critical = [idx for idx in old_mask if counts[idx] == 1]
        if not critical:
            return False

        points_needed = set()
        for idx in critical:
            points_needed.update(all_subsets[idx])
        if len(points_needed) > k:
            return False

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
        new_set_mask = set(new_mask)
        for idx in old_mask:
            if idx not in new_set_mask and counts[idx] <= 1:
                return False

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

        crit1 = [idx for idx in m1 if counts[idx] == 1]
        crit2 = [idx for idx in m2 if counts[idx] == 1]

        points_needed = set()
        for idx in crit1:
            points_needed.update(all_subsets[idx])
        for idx in crit2:
            points_needed.update(all_subsets[idx])

        if len(points_needed) > k:
            return False

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

        for idx in crit1:
            if idx not in new_mask_set:
                return False
        for idx in crit2:
            if idx not in new_mask_set:
                return False

        m1_set = set(m1)
        m2_set = set(m2)
        valid = True
        for idx in m1_set:
            if idx not in new_mask_set:
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

        for idx in m1:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] -= 1
        for idx in new_mask:
            counts[idx] += 1

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

    def try_three_for_two():
        """Replace three blocks with two. Returns True if successful."""
        nonlocal blocks, block_masks, counts, best_blocks, best_count, no_improve
        if len(blocks) < 3:
            return False
        i, j, l = rng.sample(range(len(blocks)), 3)
        b1, b2, b3 = blocks[i], blocks[j], blocks[l]
        m1, m2, m3 = block_masks[i], block_masks[j], block_masks[l]

        crit1 = [idx for idx in m1 if counts[idx] == 1]
        crit2 = [idx for idx in m2 if counts[idx] == 1]
        crit3 = [idx for idx in m3 if counts[idx] == 1]

        # Try to cover crit1+crit2 with one block and crit3 with another
        # Or try various partitions
        # Simple approach: try to merge all critical points into 2 blocks
        all_crit = crit1 + crit2 + crit3
        if not all_crit:
            return False

        all_pts = set()
        for idx in all_crit:
            all_pts.update(all_subsets[idx])
        if len(all_pts) > 2 * k:
            return False

        # Try to split all_pts into two sets of size <= k
        # Use a simple heuristic: greedy split
        pts_list = list(all_pts)
        rng.shuffle(pts_list)
        set1 = set()
        set2 = set()
        for p in pts_list:
            if len(set1) < k:
                set1.add(p)
            else:
                set2.add(p)

        # Try to complete each set to size k
        for s in (set1, set2):
            if len(s) > k:
                return False
            cands = [x for x in range(v) if x not in s]
            rng.shuffle(cands)
            for x in cands:
                if len(s) >= k:
                    break
                s.add(x)

        if len(set1) != k or len(set2) != k:
            return False

        new_block1 = tuple(sorted(set1))
        new_block2 = tuple(sorted(set2))

        # Check they are distinct and not equal to existing
        if new_block1 == new_block2:
            return False
        if new_block1 in (b1, b2, b3) and new_block2 in (b1, b2, b3):
            return False

        new_mask1 = block_mask(new_block1)
        new_mask2 = block_mask(new_block2)
        new_mask1_set = set(new_mask1)
        new_mask2_set = set(new_mask2)

        # Check all critical subsets are covered
        for idx in all_crit:
            if idx not in new_mask1_set and idx not in new_mask2_set:
                return False

        # Check no uncovered subset is lost
        m1_set = set(m1)
        m2_set = set(m2)
        m3_set = set(m3)
        valid = True
        for idx in m1_set:
            if idx not in new_mask1_set and idx not in new_mask2_set:
                if idx not in m2_set and idx not in m3_set:
                    valid = False
                    break
        if valid:
            for idx in m2_set:
                if idx not in new_mask1_set and idx not in new_mask2_set:
                    if idx not in m1_set and idx not in m3_set:
                        valid = False
                        break
        if valid:
            for idx in m3_set:
                if idx not in new_mask1_set and idx not in new_mask2_set:
                    if idx not in m1_set and idx not in m2_set:
                        valid = False
                        break

        if not valid:
            return False

        for idx in m1:
            counts[idx] -= 1
        for idx in m2:
            counts[idx] -= 1
        for idx in m3:
            counts[idx] -= 1
        for idx in new_mask1:
            counts[idx] += 1
        for idx in new_mask2:
            counts[idx] += 1

        # Remove three, add two
        indices = sorted([i, j, l], reverse=True)
        for idx in indices:
            blocks.pop(idx)
            block_masks.pop(idx)
        blocks.append(new_block1)
        block_masks.append(new_mask1)
        blocks.append(new_block2)
        block_masks.append(new_mask2)

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
                    gain = len(point_to_subsets[x] & set(unc_list))
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
                unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        if not unc_list:
            no_improve = 0
            return True
        else:
            blocks = old_blocks
            block_masks = old_masks
            counts = old_counts
            return False

    def perturb_two_blocks():
        """Perturb: remove two random blocks and greedily re-cover."""
        nonlocal blocks, block_masks, counts, no_improve
        if len(blocks) <= 2:
            return False
        old_blocks = list(blocks)
        old_masks = list(block_masks)
        old_counts = list(counts)

        i, j = rng.sample(range(len(blocks)), 2)
        for idx in block_masks[i]:
            counts[idx] -= 1
        for idx in block_masks[j]:
            counts[idx] -= 1
        hi = max(i, j)
        lo = min(i, j)
        blocks.pop(hi)
        block_masks.pop(hi)
        blocks.pop(lo)
        block_masks.pop(lo)

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
                    gain = len(point_to_subsets[x] & set(unc_list))
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
                unc_list = [idx for idx in range(n_subsets) if counts[idx] == 0]

        if not unc_list:
            no_improve = 0
            return True
        else:
            blocks = old_blocks
            block_masks = old_masks
            counts = old_counts
            return False

    # ---- Main search loop ----
    while time_left() > 3.0:
        # Try removals first (cheap and always beneficial)
        if try_remove_block():
            continue

        # Try swaps
        if try_swap_block():
            T = max(T * cooling, T_min)
            continue

        # Try 2-for-1 replacements
        if try_two_for_one():
            T = max(T * cooling, T_min)
            continue

        # Try 3-for-2 replacements (new move)
        if rng.random() < 0.3 and try_three_for_two():
            T = max(T * cooling, T_min)
            continue

        # If stuck, perturb
        no_improve += 1
        if no_improve > max_no_improve:
            if rng.random() < 0.5:
                perturb()
            else:
                perturb_two_blocks()
            no_improve = 0
        else:
            # Small random perturbation: swap a block with a random one
            if blocks and rng.random() < 0.2:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_mask = block_masks[i]
                new_block = tuple(sorted(rng.sample(range(v), k)))
                if new_block != old_block:
                    new_mask = block_mask(new_block)
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

    # ---- Final high-precision polish pass ----
    # 1) Repeated exact redundancy sweep to a fixed point
    for _ in range(300):
        if time_left() < 1.0:
            break
        removed = True
        while removed:
            removed = try_remove_block()

    # 2) Exhaustive 2-for-1 scan
    for _ in range(100):
        if time_left() < 1.0:
            break
        progress = False
        n = len(blocks)
        for i in range(n):
            if time_left() < 1.0:
                break
            m1 = block_masks[i]
            crit1 = [idx for idx in m1 if counts[idx] == 1]
            if not crit1:
                continue
            pts1 = set()
            for idx in crit1:
                pts1.update(all_subsets[idx])
            if len(pts1) > k:
                continue
            for j in range(i + 1, n):
                m2 = block_masks[j]
                crit2 = [idx for idx in m2 if counts[idx] == 1]
                pts = set(pts1)
                for idx in crit2:
                    pts.update(all_subsets[idx])
                if len(pts) > k:
                    continue
                new_set = set(pts)
                cands = [x for x in range(v) if x not in new_set]
                cands_sorted = sorted(cands, key=lambda x: -len(point_to_subsets[x]))
                for x in cands_sorted:
                    if len(new_set) >= k:
                        break
                    new_set.add(x)
                if len(new_set) != k:
                    continue
                new_block = tuple(sorted(new_set))
                if new_block in (blocks[i], blocks[j]):
                    continue
                new_mask = block_mask(new_block)
                new_mask_set = set(new_mask)
                ok = True
                for idx in crit1:
                    if idx not in new_mask_set:
                        ok = False
                        break
                if ok:
                    for idx in crit2:
                        if idx not in new_mask_set:
                            ok = False
                            break
                if not ok:
                    continue
                m1_set = set(m1)
                m2_set = set(m2)
                for idx in m1_set:
                    if idx not in new_mask_set and idx not in m2_set:
                        ok = False
                        break
                if ok:
                    for idx in m2_set:
                        if idx not in new_mask_set and idx not in m1_set:
                            ok = False
                            break
                if not ok:
                    continue
                for idx in m1:
                    counts[idx] -= 1
                for idx in m2:
                    counts[idx] -= 1
                for idx in new_mask:
                    counts[idx] += 1
                blocks.pop(j)
                block_masks.pop(j)
                blocks.pop(i)
                block_masks.pop(i)
                blocks.append(new_block)
                block_masks.append(new_mask)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                progress = True
                break
            if progress:
                break
        if not progress:
            break

    # 3) Exhaustive 3-for-2 scan
    for _ in range(50):
        if time_left() < 1.0:
            break
        progress = False
        n = len(blocks)
        for i in range(n):
            if time_left() < 1.0:
                break
            m1 = block_masks[i]
            crit1 = [idx for idx in m1 if counts[idx] == 1]
            if not crit1:
                continue
            pts1 = set()
            for idx in crit1:
                pts1.update(all_subsets[idx])
            if len(pts1) > k:
                continue
            for j in range(i + 1, n):
                if time_left() < 1.0:
                    break
                m2 = block_masks[j]
                crit2 = [idx for idx in m2 if counts[idx] == 1]
                pts2 = set(pts1)
                for idx in crit2:
                    pts2.update(all_subsets[idx])
                if len(pts2) > 2 * k:
                    continue
                for l in range(j + 1, n):
                    if time_left() < 1.0:
                        break
                    m3 = block_masks[l]
                    crit3 = [idx for idx in m3 if counts[idx] == 1]
                    pts3 = set(pts2)
                    for idx in crit3:
                        pts3.update(all_subsets[idx])
                    if len(pts3) > 2 * k:
                        continue

                    # Try to split pts3 into two sets of size k
                    pts_list = list(pts3)
                    rng.shuffle(pts_list)
                    set1 = set()
                    set2 = set()
                    for p in pts_list:
                        if len(set1) < k:
                            set1.add(p)
                        else:
                            set2.add(p)

                    for s in (set1, set2):
                        if len(s) > k:
                            break
                        cands = [x for x in range(v) if x not in s]
                        rng.shuffle(cands)
                        for x in cands:
                            if len(s) >= k:
                                break
                            s.add(x)

                    if len(set1) != k or len(set2) != k:
                        continue

                    new_block1 = tuple(sorted(set1))
                    new_block2 = tuple(sorted(set2))
                    if new_block1 == new_block2:
                        continue
                    if new_block1 in (blocks[i], blocks[j], blocks[l]) and new_block2 in (blocks[i], blocks[j], blocks[l]):
                        continue

                    new_mask1 = block_mask(new_block1)
                    new_mask2 = block_mask(new_block2)
                    new_mask1_set = set(new_mask1)
                    new_mask2_set = set(new_mask2)

                    all_crit = crit1 + crit2 + crit3
                    ok = True
                    for idx in all_crit:
                        if idx not in new_mask1_set and idx not in new_mask2_set:
                            ok = False
                            break
                    if not ok:
                        continue

                    m1_set = set(m1)
                    m2_set = set(m2)
                    m3_set = set(m3)
                    for idx in m1_set:
                        if idx not in new_mask1_set and idx not in new_mask2_set:
                            if idx not in m2_set and idx not in m3_set:
                                ok = False
                                break
                    if ok:
                        for idx in m2_set:
                            if idx not in new_mask1_set and idx not in new_mask2_set:
                                if idx not in m1_set and idx not in m3_set:
                                    ok = False
                                    break
                    if ok:
                        for idx in m3_set:
                            if idx not in new_mask1_set and idx not in new_mask2_set:
                                if idx not in m1_set and idx not in m2_set:
                                    ok = False
                                    break
                    if not ok:
                        continue

                    for idx in m1:
                        counts[idx] -= 1
                    for idx in m2:
                        counts[idx] -= 1
                    for idx in m3:
                        counts[idx] -= 1
                    for idx in new_mask1:
                        counts[idx] += 1
                    for idx in new_mask2:
                        counts[idx] += 1

                    indices = sorted([i, j, l], reverse=True)
                    for idx in indices:
                        blocks.pop(idx)
                        block_masks.pop(idx)
                    blocks.append(new_block1)
                    block_masks.append(new_mask1)
                    blocks.append(new_block2)
                    block_masks.append(new_mask2)

                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    progress = True
                    break
                if progress:
                    break
            if progress:
                break
        if not progress:
            break

    # Final cleanup: try removals one more time
    while True:
        if not try_remove_block():
            break

    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
