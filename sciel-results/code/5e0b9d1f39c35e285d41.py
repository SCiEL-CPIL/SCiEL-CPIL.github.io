import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(t_subsets)}

    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # Build bitmasks for each block and reverse mapping
    block_masks = [0] * n_k
    t_to_k = [[] for _ in range(n_t)]
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_to_idx[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        block_masks[i] = mask

    full_mask = (1 << n_t) - 1

    def greedy_cover(seed=None):
        """Greedy set cover with incremental gain updates."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        used_set = set()
        gain = [m.bit_count() for m in block_masks]

        while uncovered:
            if time.time() > deadline:
                break

            # Find max gain
            best_gain = 0
            candidates = []
            for i in range(n_k):
                if i in used_set:
                    continue
                g = gain[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain and g > 0:
                    candidates.append(i)

            if best_gain == 0:
                break

            # Random tie-breaking
            if len(candidates) > 1 and seed is not None:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            used_set.add(best_idx)

            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]

            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used_set:
                        gain[blk] -= 1

        counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        return selected, counts

    def saturate_greedy(seed=None):
        """Saturate-then-greedy: first cover hardest t-subsets, then greedy."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        used_set = set()
        gain = [m.bit_count() for m in block_masks]

        # Phase 1: Saturate - repeatedly pick block covering the least-covered t-subset
        # This forces coverage of "hard" t-subsets early
        while uncovered:
            if time.time() > deadline:
                break

            # Find the uncovered t-subset with fewest covering blocks
            # (i.e., most "constrained")
            # For efficiency, find an uncovered t-subset and pick best block for it
            # Sample a few uncovered t-subsets
            nc = uncovered
            best_block_for_hard = -1
            best_gain = 0
            hard_idx = -1

            # Find a random uncovered t-subset
            # Use bit tricks to find one
            sample_count = 0
            candidate_hard = -1
            m = uncovered
            while m and sample_count < 20:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                # Count how many blocks cover this t-subset (and are unused)
                cnt = 0
                for blk in t_to_k[idx]:
                    if blk not in used_set:
                        cnt += 1
                        if cnt >= 5:
                            break
                if cnt < 5:
                    candidate_hard = idx
                    break
                sample_count += 1
                m ^= lsb

            if candidate_hard != -1:
                # Pick the best block covering this hard t-subset
                for blk in t_to_k[candidate_hard]:
                    if blk in used_set:
                        continue
                    g = gain[blk]
                    if g > best_gain:
                        best_gain = g
                        best_block_for_hard = blk

                if best_block_for_hard != -1:
                    selected.append(best_block_for_hard)
                    used_set.add(best_block_for_hard)
                    newly = uncovered & block_masks[best_block_for_hard]
                    uncovered &= ~block_masks[best_block_for_hard]
                    nc2 = newly
                    while nc2:
                        lsb2 = nc2 & -nc2
                        j2 = lsb2.bit_length() - 1
                        nc2 ^= lsb2
                        for blk2 in t_to_k[j2]:
                            if blk2 not in used_set:
                                gain[blk2] -= 1
                    continue

            # Phase 2: Standard greedy
            best_gain = 0
            candidates = []
            for i in range(n_k):
                if i in used_set:
                    continue
                g = gain[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain and g > 0:
                    candidates.append(i)

            if best_gain == 0:
                break

            if len(candidates) > 1 and seed is not None:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]

            selected.append(best_idx)
            used_set.add(best_idx)

            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]

            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used_set:
                        gain[blk] -= 1

        counts = np.zeros(n_t, dtype=np.int32)
        for b in selected:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= l1b

        return selected, counts

    def remove_redundant(blocks, counts):
        """Remove blocks that don't uniquely cover any t-subset."""
        if time.time() > deadline:
            return blocks, counts

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []

            for b in blocks:
                m = block_masks[b]
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b)

            if to_remove:
                remove_set = set(to_remove)
                blocks = [b for b in blocks if b not in remove_set]
                for b in to_remove:
                    m = block_masks[b]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                changed = True

        return blocks, counts

    def swap_improve(blocks, counts):
        """Try 1-for-1 swaps to reduce block count."""
        if time.time() > deadline:
            return blocks, counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            for i in range(len(blocks)):
                if time.time() > deadline:
                    break

                b = blocks[i]
                mask_remove = block_masks[b]

                # Temporarily remove
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb

                # Find uncovered
                uncovered = 0
                for j in range(n_t):
                    if counts[j] == 0:
                        uncovered |= (1 << j)

                if uncovered == 0:
                    blocks = [x for j, x in enumerate(blocks) if j != i]
                    improved = True
                    break

                # Try to find a single block that covers all uncovered
                best_new = -1
                best_gain = 0
                n_uncov = uncovered.bit_count()
                for new_b in range(n_k):
                    if new_b == b or new_b in blocks:
                        continue
                    g = (uncovered & block_masks[new_b]).bit_count()
                    if g > best_gain:
                        best_gain = g
                        best_new = new_b
                        if g == n_uncov:
                            break

                # Restore
                m = mask_remove
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb

                if best_new != -1 and best_gain == n_uncov:
                    # Swap
                    m = mask_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb

                    blocks[i] = best_new
                    m = block_masks[best_new]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    improved = True
                    break

        return blocks, counts

    def two_opt(blocks, counts):
        """Try 2-for-1 swaps: remove 2 blocks, add 1."""
        if time.time() > deadline:
            return blocks, counts

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False

            n = len(blocks)
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break

                    b1, b2 = blocks[i], blocks[j]
                    mask1, mask2 = block_masks[b1], block_masks[b2]

                    # Temporarily remove both
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb

                    # Find uncovered
                    uncovered = 0
                    for t_idx in range(n_t):
                        if counts[t_idx] == 0:
                            uncovered |= (1 << t_idx)

                    if uncovered == 0:
                        # Both were redundant
                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        counts2 = np.zeros(n_t, dtype=np.int32)
                        for b in new_blocks:
                            m = block_masks[b]
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                counts2[idx] += 1
                                m ^= lsb
                        blocks = new_blocks
                        counts = counts2
                        improved = True
                        break

                    # Try to find one block covering all uncovered
                    n_uncov = uncovered.bit_count()
                    best_new = -1
                    best_gain = 0
                    for new_b in range(n_k):
                        if new_b == b1 or new_b == b2 or new_b in blocks:
                            continue
                        g = (uncovered & block_masks[new_b]).bit_count()
                        if g > best_gain:
                            best_gain = g
                            best_new = new_b
                            if g == n_uncov:
                                break

                    # Restore
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb

                    if best_new != -1 and best_gain == n_uncov:
                        # Do the swap
                        m = mask1
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        m = mask2
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb

                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        new_blocks.append(best_new)
                        m = block_masks[best_new]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                        blocks = new_blocks
                        improved = True
                        break

                if improved:
                    break

        return blocks, counts

    def cover_holes(blocks, holes):
        """Greedily cover remaining holes."""
        new_blocks = []
        uncovered = holes
        used = set(blocks)
        gain = [(uncovered & block_masks[i]).bit_count() for i in range(n_k)]

        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k):
                if i in used:
                    continue
                if gain[i] > best_gain:
                    best_gain = gain[i]
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            used.add(best_idx)
            newly = uncovered & block_masks[best_idx]
            uncovered &= ~block_masks[best_idx]
            nc = newly
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                nc ^= lsb
                for blk in t_to_k[j]:
                    if blk not in used:
                        gain[blk] -= 1
        return new_blocks

    def rebuild_counts(blocks):
        counts = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            m = block_masks[b]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    def coverage_mask(blocks):
        mask = 0
        for b in blocks:
            mask |= block_masks[b]
        return mask

    # === Main search ===
    best_blocks = None
    best_counts = None
    best_count = float('inf')

    # Deterministic greedy
    blocks, counts = greedy_cover(seed=None)
    blocks, counts = remove_redundant(blocks, counts)
    blocks, counts = swap_improve(blocks, counts)
    blocks, counts = two_opt(blocks, counts)
    best_blocks = blocks
    best_counts = counts
    best_count = len(blocks)

    # Saturate-greedy
    if time.time() < deadline - 30:
        blocks, counts = saturate_greedy(seed=123)
        blocks, counts = remove_redundant(blocks, counts)
        blocks, counts = swap_improve(blocks, counts)
        blocks, counts = two_opt(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # Randomized restarts
    n_restarts = 12
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        seed = 100 + r * 7919
        blocks, counts = greedy_cover(seed=seed)
        blocks, counts = remove_redundant(blocks, counts)
        blocks, counts = swap_improve(blocks, counts)
        blocks, counts = two_opt(blocks, counts)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_counts = counts
            best_count = len(blocks)

    # Perturbation-based local search
    attempts = 0
    while time.time() < deadline - 8:
        attempts += 1
        num_remove = min(len(best_blocks), random.randint(2, 6))
        if num_remove == 0:
            break

        to_remove = random.sample(best_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed = [b for b in best_blocks if b not in remove_set]

        covered = coverage_mask(perturbed)
        holes = full_mask & ~covered

        added = cover_holes(perturbed, holes)
        new_blocks = perturbed + added

        if len(new_blocks) < best_count:
            new_counts = rebuild_counts(new_blocks)
            new_blocks, new_counts = remove_redundant(new_blocks, new_counts)
            new_blocks, new_counts = swap_improve(new_blocks, new_counts)
            new_blocks, new_counts = two_opt(new_blocks, new_counts)
            if len(new_blocks) < best_count:
                best_blocks = new_blocks
                best_counts = new_counts
                best_count = len(new_blocks)

    # Final polish
    if time.time() < deadline - 3:
        best_blocks, best_counts = remove_redundant(best_blocks, best_counts)
        best_blocks, best_counts = swap_improve(best_blocks, best_counts)
        best_blocks, best_counts = two_opt(best_blocks, best_counts)

    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    return final_blocks
