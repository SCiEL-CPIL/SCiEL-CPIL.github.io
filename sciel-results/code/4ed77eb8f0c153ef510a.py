import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(all_points, t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute coverage masks for all k-subsets
    all_ksubs = list(itertools.combinations(all_points, k))
    n_ksubs = len(all_ksubs)

    # For each k-subset, compute bitmask of t-subsets it covers
    # C(7,3) = 35 t-subsets per block
    masks = [0] * n_ksubs
    for i, ksub in enumerate(all_ksubs):
        m = 0
        for tsub in itertools.combinations(ksub, t):
            idx = tsub_to_idx[tsub]
            m |= (1 << idx)
        masks[i] = m

    # Also store the actual blocks for quick access
    # all_ksubs[i] is the i-th k-subset

    best_blocks = None
    best_count = float('inf')

    def greedy_cover(seed):
        """Randomized greedy set cover using precomputed masks."""
        rng = random.Random(seed)
        uncovered = (1 << n_tsubs) - 1
        selected = []  # list of k-subset indices
        used = [False] * n_ksubs

        while uncovered:
            best_gain = 0
            best_idx = -1
            best_mask = 0

            # Scan all candidates
            for i in range(n_ksubs):
                if used[i]:
                    continue
                gain_mask = uncovered & masks[i]
                if gain_mask == 0:
                    continue
                gain = gain_mask.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    best_mask = gain_mask
                elif gain == best_gain and gain > 0:
                    # Randomized tie-breaking
                    if rng.random() < 0.15:
                        best_idx = i
                        best_mask = gain_mask

            if best_idx == -1:
                break

            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~best_mask

            # Time check
            if time.time() > deadline - 15:
                break

        if uncovered != 0:
            return None  # Incomplete cover

        return selected

    def blocks_from_indices(indices):
        """Convert k-subset indices to list of blocks."""
        return [list(all_ksubs[i]) for i in indices]

    def coverage_counts(blocks):
        """Compute how many times each t-subset is covered."""
        counts = [0] * n_tsubs
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                counts[tsub_to_idx[tsub]] += 1
        return counts

    def try_remove_redundant(blocks, seed):
        """Remove redundant blocks (those whose t-subsets are all covered by others)."""
        rng = random.Random(seed)
        blocks = list(blocks)
        n = len(blocks)

        # Precompute coverage masks for selected blocks
        block_masks = []
        for block in blocks:
            m = 0
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                m |= (1 << idx)
            block_masks.append(m)

        changed = True
        while changed:
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                # Check if block i is redundant
                # For each t-subset in block i, is it covered by some other block?
                redundant = True
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    covered = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if (block_masks[j] >> idx) & 1:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        return blocks

    def try_swap_improve(blocks, seed):
        """Try to replace one block with another to reduce total count or maintain count with better coverage."""
        rng = random.Random(seed)
        blocks = list(blocks)
        n = len(blocks)

        # Precompute coverage masks
        block_masks = []
        for block in blocks:
            m = 0
            for tsub in itertools.combinations(block, t):
                idx = tsub_to_idx[tsub]
                m |= (1 << idx)
            block_masks.append(m)

        # Compute coverage count for each t-subset
        cov_count = [0] * n_tsubs
        for bm in block_masks:
            b = bm
            while b:
                lb = b & (-b)
                idx = lb.bit_length() - 1
                cov_count[idx] += 1
                b ^= lb

        # Try swaps: for each block i, try replacing with a random other block
        # that covers all t-subsets that i uniquely covers
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks

                # Find t-subsets uniquely covered by block i
                unique_tsubs = []
                for tsub in itertools.combinations(blocks[i], t):
                    idx = tsub_to_idx[tsub]
                    if cov_count[idx] == 1:
                        unique_tsubs.append(idx)

                if not unique_tsubs:
                    # Block i is redundant, remove it
                    blocks.pop(i)
                    block_masks.pop(i)
                    # Recompute cov_count
                    cov_count = [0] * n_tsubs
                    for bm in block_masks:
                        b = bm
                        while b:
                            lb = b & (-b)
                            idx = lb.bit_length() - 1
                            cov_count[idx] += 1
                            b ^= lb
                    improved = True
                    break

                # Try to find a replacement block that covers all unique_tsubs
                # and doesn't add too many new t-subsets
                # Randomly sample some candidate blocks
                candidates = rng.sample(range(n_ksubs), min(200, n_ksubs))
                best_candidate = -1
                best_extra = float('inf')

                for ci in candidates:
                    cm = masks[ci]
                    # Check if cm covers all unique_tsubs
                    ok = True
                    for ut in unique_tsubs:
                        if not (cm >> ut) & 1:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Count extra t-subsets covered (not in current coverage)
                    # This is approximate; we just want a valid replacement
                    best_candidate = ci
                    break

                if best_candidate >= 0:
                    # Replace block i with best_candidate
                    blocks[i] = list(all_ksubs[best_candidate])
                    block_masks[i] = masks[best_candidate]
                    # Recompute cov_count
                    cov_count = [0] * n_tsubs
                    for bm in block_masks:
                        b = bm
                        while b:
                            lb = b & (-b)
                            idx = lb.bit_length() - 1
                            cov_count[idx] += 1
                            b ^= lb
                    improved = True
                    break

        return blocks

    # Main loop: try multiple seeds
    seed = 0
    while time.time() < deadline - 20:
        selected = greedy_cover(seed)
        if selected is not None:
            blocks = blocks_from_indices(selected)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)

            # Try to improve with local search
            if time.time() < deadline - 10:
                improved = try_remove_redundant(blocks, seed)
                if len(improved) < best_count:
                    best_count = len(improved)
                    best_blocks = list(improved)

                if time.time() < deadline - 5:
                    improved2 = try_swap_improve(improved, seed + 1000)
                    if len(improved2) < best_count:
                        best_count = len(improved2)
                        best_blocks = list(improved2)

        seed += 1

    if best_blocks is None:
        # Fallback: simple greedy
        selected = greedy_cover(0)
        if selected is not None:
            best_blocks = blocks_from_indices(selected)
        else:
            # Emergency: construct a trivially valid (but large) cover
            best_blocks = [list(all_ksubs[i]) for i in range(n_ksubs)]

    return best_blocks
