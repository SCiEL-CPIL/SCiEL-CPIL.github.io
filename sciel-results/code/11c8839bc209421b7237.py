import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Generate all t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)

    # Generate all k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)

    # Build bitmasks: each block is a Python int with bits set for t-subsets it covers
    # For fast coverage operations
    block_masks = []
    for block in k_subs:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx.get(t_sub)
            if j is not None:
                mask |= (1 << j)
        block_masks.append(mask)

    # Full coverage mask
    full_mask = (1 << n_t) - 1

    # Precompute popcount helper
    def popcount(x):
        return bin(x).count('1')

    def greedy_cover(seed_blocks=None, max_iter=10000, rng=None, use_random_tiebreak=False):
        """Greedy set cover using bitmasks for fast operations."""
        if rng is None:
            rng = random

        uncovered = full_mask
        selected_blocks = []

        # If we have seed blocks, start with them
        if seed_blocks:
            for idx in seed_blocks:
                uncovered &= ~block_masks[idx]
                selected_blocks.append(idx)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if uncovered == 0:
                break

            # Find the block that covers the most uncovered t-subsets
            # We need to compute popcount(block_masks[i] & uncovered) for all i
            # This is expensive, so we'll use a smarter approach

            # First, try a quick heuristic: find blocks that cover many uncovered
            # Use sampling for speed if n_k is large
            best_idx = -1
            best_gain = 0

            # Check all blocks but with early termination
            # For speed, we can check a subset first
            if n_k > 5000:
                # Check a random sample first
                sample_size = min(200, n_k)
                sample = rng.sample(range(n_k), sample_size)
                for idx in sample:
                    gain = popcount(block_masks[idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx

                # Then check remaining blocks only if potential improvement
                for idx in range(n_k):
                    if idx in sample:
                        continue
                    # Quick upper bound: total bits in block
                    total_bits = popcount(block_masks[idx])
                    if total_bits <= best_gain:
                        continue
                    gain = popcount(block_masks[idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
            else:
                for idx in range(n_k):
                    gain = popcount(block_masks[idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx

            if best_gain == 0:
                break

            selected_blocks.append(best_idx)
            uncovered &= ~block_masks[best_idx]

        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove redundant blocks using bitmask operations."""
        if not selected_blocks:
            return selected_blocks

        n_sel = len(selected_blocks)
        # Compute coverage: for each t-subset, how many blocks cover it
        # Using bitmask: count how many selected block masks have bit j set
        cover_count = [0] * n_t

        for idx in selected_blocks:
            mask = block_masks[idx]
            # Count bits set
            m = mask
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                cover_count[j] += 1
                m ^= lsb

        active = [True] * n_sel
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False

            redundant = []
            for i in range(n_sel):
                if not active[i]:
                    continue
                mask = block_masks[selected_blocks[i]]
                # Check if any t-subset is covered only by this block
                is_essential = False
                m = mask
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                if not is_essential:
                    redundant.append(i)

            if redundant:
                changed = True
                for i in redundant:
                    mask = block_masks[selected_blocks[i]]
                    m = mask
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        cover_count[j] -= 1
                        m ^= lsb
                    active[i] = False

        return [selected_blocks[i] for i in range(n_sel) if active[i]]

    def local_improve(selected_blocks, max_rounds=5):
        """Improve by trying to remove blocks and add replacements."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            # Compute cover count
            cover_count = [0] * n_t
            for idx in best:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    cover_count[j] += 1
                    m ^= lsb

            # Try removing each block
            for i in range(n_best):
                if time.time() > deadline - 2:
                    break

                removed_idx = best[i]
                mask = block_masks[removed_idx]

                # Find t-subsets that would become uncovered
                uncovered = []
                m = mask
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    if cover_count[j] == 1:
                        uncovered.append(j)
                    m ^= lsb

                if not uncovered:
                    # Block is redundant, remove it
                    m = mask
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        cover_count[j] -= 1
                        m ^= lsb
                    best.pop(i)
                    improved = True
                    break

                # Find best candidate covering all uncovered
                best_candidate = -1
                best_score = 0
                uncovered_set = set(uncovered)

                for idx in range(n_k):
                    if idx in best:
                        continue
                    mask_c = block_masks[idx]
                    # Count how many uncovered are covered
                    score = 0
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        if j in uncovered_set:
                            score += 1
                        m ^= lsb

                    if score > best_score:
                        best_score = score
                        best_candidate = idx

                if best_score >= len(uncovered):
                    # Replace
                    m = mask
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        cover_count[j] -= 1
                        m ^= lsb
                    best[i] = best_candidate
                    mask_c = block_masks[best_candidate]
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        j = lsb.bit_length() - 1
                        cover_count[j] += 1
                        m ^= lsb
                    improved = True
                    break

            if not improved:
                break

        return best

    def local_improve2(selected_blocks, max_rounds=3):
        """1-for-2 swaps: remove 2 blocks, add 1."""
        if not selected_blocks:
            return selected_blocks

        best = list(selected_blocks)
        best_size = len(best)

        for round_num in range(max_rounds):
            if time.time() > deadline - 2:
                break
            if len(best) == 0:
                break

            improved = False
            n_best = len(best)

            if n_best < 2:
                break

            # Compute cover count
            cover_count = [0] * n_t
            for idx in best:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    cover_count[j] += 1
                    m ^= lsb

            # Try pairs
            max_pairs = min(150, n_best * (n_best - 1) // 2)
            pairs = [(i, j) for i in range(n_best) for j in range(i + 1, n_best)]
            if len(pairs) > max_pairs:
                pairs = random.sample(pairs, max_pairs)

            for i, j in pairs:
                if time.time() > deadline - 2:
                    break

                idx_i = best[i]
                idx_j = best[j]
                mask_i = block_masks[idx_i]
                mask_j = block_masks[idx_j]

                # Find uncovered if we remove both
                cc = cover_count[:]
                m = mask_i
                while m:
                    lsb = m & (-m)
                    pos = lsb.bit_length() - 1
                    cc[pos] -= 1
                    m ^= lsb
                m = mask_j
                while m:
                    lsb = m & (-m)
                    pos = lsb.bit_length() - 1
                    cc[pos] -= 1
                    m ^= lsb

                # Uncovered: positions where cc == 0 and were in mask_i or mask_j
                union_mask = mask_i | mask_j
                uncovered = []
                m = union_mask
                while m:
                    lsb = m & (-m)
                    pos = lsb.bit_length() - 1
                    if cc[pos] == 0:
                        uncovered.append(pos)
                    m ^= lsb

                if not uncovered:
                    # Both redundant
                    m = mask_i
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lsb
                    m = mask_j
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lsb
                    if j > i:
                        best.pop(j)
                        best.pop(i)
                    else:
                        best.pop(i)
                        best.pop(j)
                    improved = True
                    break

                # Find best candidate
                best_candidate = -1
                best_score = 0
                uncovered_set = set(uncovered)

                for idx in range(n_k):
                    if idx in best:
                        continue
                    mask_c = block_masks[idx]
                    score = 0
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        if pos in uncovered_set:
                            score += 1
                        m ^= lsb
                    if score > best_score:
                        best_score = score
                        best_candidate = idx

                if best_score >= len(uncovered):
                    # Replace both with one
                    m = mask_i
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lsb
                    m = mask_j
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        cover_count[pos] -= 1
                        m ^= lsb
                    new_best = []
                    for m_idx, idx in enumerate(best):
                        if m_idx != i and m_idx != j:
                            new_best.append(idx)
                    new_best.append(best_candidate)
                    mask_c = block_masks[best_candidate]
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        pos = lsb.bit_length() - 1
                        cover_count[pos] += 1
                        m ^= lsb
                    if len(new_best) < best_size:
                        best = new_best
                        best_size = len(best)
                        improved = True
                        break

            if not improved:
                break

        return best

    # Main search
    best_result = None
    best_size = float('inf')

    # Standard greedy
    result = greedy_cover()
    result = remove_redundant(result)
    if len(result) < best_size:
        best_result = result
        best_size = len(result)

    # Randomized restarts
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 45:
            break
        seed_size = min(5, n_k)
        seed_blocks = random.sample(range(n_k), seed_size)
        result = greedy_cover(seed_blocks=seed_blocks)
        result = remove_redundant(result)
        if len(result) < best_size:
            best_result = result
            best_size = len(result)

    # Local improvement
    if best_result and time.time() < deadline - 25:
        for _ in range(3):
            if time.time() > deadline - 25:
                break
            improved = local_improve(best_result, max_rounds=3)
            improved = remove_redundant(improved)
            if len(improved) < best_size:
                best_result = improved
                best_size = len(improved)
            else:
                break

    # Stronger local search
    if best_result and time.time() < deadline - 12:
        for _ in range(2):
            if time.time() > deadline - 12:
                break
            improved = local_improve2(best_result, max_rounds=2)
            improved = remove_redundant(improved)
            if len(improved) < best_size:
                best_result = improved
                best_size = len(improved)
            else:
                break

    # Final redundancy removal
    if best_result:
        best_result = remove_redundant(best_result)

    if best_result is None:
        best_result = []

    final_blocks = [list(k_subs[i]) for i in best_result]
    return final_blocks
