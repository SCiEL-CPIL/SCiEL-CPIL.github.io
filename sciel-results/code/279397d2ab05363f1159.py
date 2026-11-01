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

    def greedy_cover(seed=None, sample_size=None):
        """Greedy set cover with optional candidate sampling for speed."""
        if seed is not None:
            random.seed(seed)

        uncovered = full_mask
        selected = []
        used_set = set()
        gain = [m.bit_count() for m in block_masks]
        
        if sample_size is None:
            sample_size = min(n_k, 200)

        while uncovered:
            if time.time() > deadline:
                break

            # Find max gain
            best_gain = 0
            best_idx = -1
            candidates = []
            
            # Use sampling if n_k is large and we have a sample size
            if sample_size < n_k and seed is not None:
                sample = random.sample(range(n_k), sample_size)
                for i in sample:
                    if i in used_set:
                        continue
                    g = gain[i]
                    if g > best_gain:
                        best_gain = g
                        best_idx = i
            else:
                for i in range(n_k):
                    if i in used_set:
                        continue
                    g = gain[i]
                    if g > best_gain:
                        best_gain = g
                        best_idx = i
                    elif g == best_gain and g > 0:
                        candidates.append(i)
                if best_idx == -1 and candidates:
                    best_idx = random.choice(candidates) if seed is not None else candidates[0]

            if best_gain == 0:
                break

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
        """Try 2-for-1 swaps: remove 2 blocks, add 1 covering all uniquely-covered t-sets."""
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
                for j in range(i + 1, n):
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

                    # Find uncovered (t-sets now covered 0 times)
                    uncovered = 0
                    for t_idx in range(n_t):
                        if counts[t_idx] == 0:
                            uncovered |= (1 << t_idx)

                    if uncovered == 0:
                        # Both blocks were fully redundant
                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        blocks = new_blocks
                        improved = True
                        break

                    # Try to find one block covering all uncovered t-sets
                    n_uncov = uncovered.bit_count()
                    best_new = -1
                    best_gain = 0
                    block_set = set(blocks)
                    for new_b in range(n_k):
                        if new_b == b1 or new_b == b2 or new_b in block_set:
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
                        # Do the 2-for-1 merge: remove b1,b2, add best_new
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

    def merge_pass(blocks, counts, max_rounds=3):
        """Fast 2-for-1 merge: repeatedly find a pair whose uniquely-covered t-sets
        are all covered by a single other block, and merge them."""
        if time.time() > deadline:
            return blocks, counts
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            n = len(blocks)
            if n < 2:
                break
            block_set = set(blocks)
            merged = False
            for i in range(n):
                if time.time() > deadline:
                    return blocks, counts
                for j in range(i + 1, n):
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
                    uncovered = 0
                    for t_idx in range(n_t):
                        if counts[t_idx] == 0:
                            uncovered |= (1 << t_idx)
                    if uncovered == 0:
                        # both redundant
                        new_blocks = [x for p, x in enumerate(blocks) if p != i and p != j]
                        # restore not needed since we drop them
                        blocks = new_blocks
                        merged = True
                        break
                    n_uncov = uncovered.bit_count()
                    best_new = -1
                    best_gain = 0
                    for new_b in range(n_k):
                        if new_b == b1 or new_b == b2 or new_b in block_set:
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
                        merged = True
                        break
                if merged:
                    break
            if not merged:
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
    blocks, counts = swap_improve(blocks, counts)
    blocks, counts = merge_pass(blocks, counts, max_rounds=4)
    best_blocks = blocks
    best_counts = counts
    best_count = len(blocks)

    # Randomized restarts with sampling
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        seed = 100 + r * 7919
        # Use sampling to speed up greedy
        blocks, counts = greedy_cover(seed=seed, sample_size=150)
        blocks, counts = remove_redundant(blocks, counts)
        blocks, counts = swap_improve(blocks, counts)
        blocks, counts = two_opt(blocks, counts)
        blocks, counts = swap_improve(blocks, counts)
        blocks, counts = merge_pass(blocks, counts, max_rounds=3)
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
            new_blocks, new_counts = merge_pass(new_blocks, new_counts, max_rounds=3)
            if len(new_blocks) < best_count:
                best_blocks = new_blocks
                best_counts = new_counts
                best_count = len(new_blocks)

    # Final polish
    if time.time() < deadline - 3:
        best_blocks, best_counts = remove_redundant(best_blocks, best_counts)
        best_blocks, best_counts = swap_improve(best_blocks, best_counts)
        best_blocks, best_counts = two_opt(best_blocks, best_counts)
        best_blocks, best_counts = merge_pass(best_blocks, best_counts, max_rounds=4)

    final_blocks = [list(all_k_blocks[i]) for i in best_blocks]
    return final_blocks
