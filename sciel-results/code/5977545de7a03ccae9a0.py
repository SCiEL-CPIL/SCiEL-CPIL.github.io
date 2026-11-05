import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # Precompute all k-blocks and their t-subset coverage
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]

    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)

    # Build bitset for each block: a Python int where bit i is 1 if block covers subset i
    n_words = (n_subsets + 63) // 64
    block_bitsets = [0] * n_blocks
    for b_idx in range(n_blocks):
        bits = 0
        for u in block_to_subsets[b_idx]:
            bits |= (1 << u)
        block_bitsets[b_idx] = bits

    # For fast "is subset u in block b" check, precompute a set for each block
    block_subset_sets = [set(block_to_subsets[b]) for b in range(n_blocks)]

    def greedy_cover_ratio(seed_val, use_ratio=True):
        """Run greedy set cover with ratio-based scoring. Returns list of block indices."""
        rng = random.Random(seed_val if seed_val is not None else 0)
        uncovered_mask = (1 << n_subsets) - 1  # all subsets uncovered
        chosen = []
        used = set()

        # Precompute gain for each block (number of subsets it covers)
        block_sizes = [len(block_to_subsets[b]) for b in range(n_blocks)]

        while True:
            if time.time() > deadline - 1.0:
                break
            if uncovered_mask == 0:
                break

            best_b = -1
            best_score = -1.0

            if use_ratio:
                # Ratio: new_gain / block_size (penalizes large blocks with low marginal gain)
                for b in range(n_blocks):
                    if b in used:
                        continue
                    new_gain = bin(block_bitsets[b] & uncovered_mask).count('1')
                    if new_gain <= 0:
                        continue
                    score = new_gain / block_sizes[b]
                    if score > best_score:
                        best_score = score
                        best_b = b
            else:
                # Absolute gain
                for b in range(n_blocks):
                    if b in used:
                        continue
                    new_gain = bin(block_bitsets[b] & uncovered_mask).count('1')
                    if new_gain > best_score:
                        best_score = new_gain
                        best_b = b

            if best_b < 0:
                break

            # Randomized perturbation
            if seed_val is not None and rng.random() < 0.15:
                # Pick a random block with decent gain
                good_candidates = []
                for b in range(n_blocks):
                    if b in used:
                        continue
                    new_gain = bin(block_bitsets[b] & uncovered_mask).count('1')
                    if new_gain > best_score * 0.5:
                        good_candidates.append(b)
                if good_candidates:
                    best_b = rng.choice(good_candidates)

            chosen.append(best_b)
            used.add(best_b)
            uncovered_mask &= ~block_bitsets[best_b]

        return chosen

    def greedy_cover_fast(seed_val):
        """Fast greedy using numpy for gain tracking. Returns list of block indices."""
        rng = random.Random(seed_val if seed_val is not None else 0)
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        gains = np.array([len(block_to_subsets[b]) for b in range(n_blocks)], dtype=np.int32)
        chosen = []
        used = np.zeros(n_blocks, dtype=np.bool_)

        while True:
            if time.time() > deadline - 1.0:
                break
            if not uncovered.any():
                break

            max_gain = gains.max()
            if max_gain <= 0:
                break

            if seed_val is not None and rng.random() < 0.25:
                threshold = max_gain * 0.7
                candidates = np.where((gains >= threshold) & (~used))[0]
                if len(candidates) > 0:
                    b_star = int(rng.choice(candidates))
                else:
                    candidates = np.where(~used)[0]
                    b_star = int(candidates[np.argmax(gains[candidates])])
            else:
                unused = np.where(~used)[0]
                b_star = int(unused[np.argmax(gains[unused])])

            chosen.append(b_star)
            used[b_star] = True

            for u in block_to_subsets[b_star]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1

            if not uncovered.any():
                break

        return chosen

    def remove_redundant(block_indices):
        """Remove blocks that are not needed for coverage using bitsets. Returns reduced list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        remaining = list(block_indices)
        # Build union bitset of all blocks
        union_bits = 0
        for b in remaining:
            union_bits |= block_bitsets[b]

        # For each block, check if it's redundant: block_bits & (union_bits - block_bits) covers all its bits
        # i.e., block_bits is a subset of (union_bits - block_bits) is wrong.
        # A block is redundant if all its subsets are covered by OTHER blocks.
        # other_bits = union_bits & ~block_bitsets[b]
        # redundant if (block_bitsets[b] & ~other_bits) == 0, i.e., block_bitsets[b] & ~other_bits == 0
        # which means block_bitsets[b] | other_bits == other_bits, i.e., block_bitsets[b] & ~other_bits == 0

        changed = True
        while changed:
            changed = False
            for i in range(len(remaining)):
                b = remaining[i]
                # Recompute union without block b
                # Efficient: check if block_b's bits are all in the union of others
                # union_others = union_bits ^ block_bitsets[b] only works if b appears once
                # Safer: check each subset of b is covered by another block
                # Using bitsets: other_cover = union_bits & ~block_bitsets[b]
                # But this is wrong if another block also covers some of b's subsets.
                # Correct check: for each subset u in block b, is there another block in remaining that covers u?
                # Equivalent: (block_bitsets[b] & ~union_others) == 0 where union_others = OR of all other blocks
                # But computing union_others each time is expensive.
                # Alternative: count coverage
                # For speed with bitsets: 
                # A block b is redundant iff for every u in block_to_subsets[b], there exists b' != b in remaining with u in block_to_subsets[b']
                # Using bitsets: let other_union = union of bitsets of all blocks in remaining except b
                # b is redundant iff (block_bitsets[b] & ~other_union) == 0

                # Compute other_union efficiently:
                # Since we maintain union_bits, other_union = union_bits if b's bits are all in others
                # Actually: other_union = union_bits & ~block_bitsets[b] is NOT correct in general
                # because union_bits includes b's contribution.
                # We need: other_union = OR of block_bitsets[b'] for b' in remaining, b' != b
                # This is expensive to recompute. Instead, use a different approach:
                # Count how many blocks cover each subset.

                # Fast approach with bitsets:
                # other_bits = 0
                # for j in range(len(remaining)):
                #     if j != i:
                #         other_bits |= block_bitsets[remaining[j]]
                # if (block_bitsets[b] & ~other_bits) == 0:
                #     # redundant
                #     remaining.pop(i)
                #     changed = True
                #     break

                # Even faster: precompute pairwise? No, too many.
                # Use the count approach but with early exit:
                removable = True
                for u in block_to_subsets[b]:
                    count = 0
                    for j in range(len(remaining)):
                        if j != i and u in block_subset_sets[remaining[j]]:
                            count += 1
                            if count >= 1:
                                break
                    if count == 0:
                        removable = False
                        break
                if removable:
                    remaining.pop(i)
                    changed = True
                    break

        return remaining

    def local_search(block_indices):
        """Try to improve by removing redundant blocks and swapping. Returns improved list."""
        if len(block_indices) <= 1:
            return list(block_indices)

        remaining = list(block_indices)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for i in range(len(remaining)):
                if time.time() > deadline - 3.0:
                    break
                b_remove = remaining[i]
                # Check if b_remove can be removed
                can_remove = True
                for u in block_to_subsets[b_remove]:
                    count = 0
                    for j in range(len(remaining)):
                        if j != i and u in block_subset_sets[remaining[j]]:
                            count += 1
                            if count >= 1:
                                break
                    if count == 0:
                        can_remove = False
                        break
                if can_remove:
                    remaining.pop(i)
                    improved = True
                    break
                else:
                    # Try swapping
                    critical = []
                    for u in block_to_subsets[b_remove]:
                        count = 0
                        for j in range(len(remaining)):
                            if j != i and u in block_subset_sets[remaining[j]]:
                                count += 1
                                if count >= 1:
                                    break
                        if count == 0:
                            critical.append(u)
                    if not critical:
                        continue
                    # Find a block that covers all critical subsets
                    best_candidate = -1
                    best_extra = -1
                    for b_add in subset_to_blocks[critical[0]]:
                        if b_add == b_remove:
                            continue
                        if b_add in remaining:
                            continue
                        covers_all = all(cu in block_subset_sets[b_add] for cu in critical)
                        if not covers_all:
                            continue
                        # Count extra coverage
                        extra = 0
                        for u in block_to_subsets[b_add]:
                            # Check if u is covered by remaining (excluding b_remove)
                            covered = False
                            for j in range(len(remaining)):
                                if j != i and u in block_subset_sets[remaining[j]]:
                                    covered = True
                                    break
                            if not covered:
                                extra += 1
                        if extra > best_extra:
                            best_extra = extra
                            best_candidate = b_add
                    if best_candidate >= 0:
                        remaining[i] = best_candidate
                        improved = True
                        break

        return remaining

    # Run multiple greedy restarts with different seeds
    best_blocks = None
    best_count = float('inf')

    # First, fast greedy (absolute gain)
    chosen = greedy_cover_fast(0)
    chosen = remove_redundant(chosen)
    chosen = local_search(chosen)
    best_blocks = chosen
    best_count = len(chosen)

    # Ratio-based greedy
    if time.time() < deadline - 15.0:
        chosen = greedy_cover_ratio(0, use_ratio=True)
        chosen = remove_redundant(chosen)
        chosen = local_search(chosen)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    # More randomized restarts
    for seed_val in range(1, 30):
        if time.time() > deadline - 8.0:
            break
        use_ratio = (seed_val % 2 == 0)
        if use_ratio:
            chosen = greedy_cover_ratio(seed_val, use_ratio=True)
        else:
            chosen = greedy_cover_fast(seed_val)
        if len(chosen) >= best_count + 2:
            continue
        chosen = remove_redundant(chosen)
        chosen = local_search(chosen)
        if len(chosen) < best_count:
            best_count = len(chosen)
            best_blocks = chosen

    return [list(all_blocks[b]) for b in best_blocks]
