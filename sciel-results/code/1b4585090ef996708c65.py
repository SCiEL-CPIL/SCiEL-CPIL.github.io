"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)

    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # Coverage: numpy array of uint64 words
    n_words = (n_subsets + 63) // 64
    covered = np.zeros(n_words, dtype=np.uint64)
    uncovered_count = n_subsets

    # Precompute extend_map: for each (t-1)-subset, list of t-subset indices containing it
    from collections import defaultdict
    extend_map = defaultdict(list)
    for i, s in enumerate(all_t_subsets):
        for j in range(t):
            key = s[:j] + s[j+1:]
            extend_map[key].append(i)

    blocks = []

    # Greedy construction
    while uncovered_count > 0:
        if time.time() > deadline - 20:
            break

        # Find an uncovered subset to seed a new block
        seed_idx = _find_first_uncovered(covered, n_subsets)
        if seed_idx is None or seed_idx >= n_subsets:
            break

        seed = list(all_t_subsets[seed_idx])
        block = seed[:]

        # Build block greedily
        while len(block) < k:
            if time.time() > deadline - 15:
                break

            block_set = set(block)
            best_point = None
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                # Gain = number of newly covered t-subsets
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    key = tuple(sorted(rest))
                    if key in extend_map:
                        for idx in extend_map[key]:
                            w = idx >> 6
                            b = 1 << (idx & 63)
                            if not (covered[w] & b):
                                gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is None or best_gain <= 0:
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break

            if best_point is None:
                break
            block.append(best_point)

        if len(block) < k:
            block_set = set(block)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)

        block.sort()
        blocks.append(block)

        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            w = idx >> 6
            b = 1 << (idx & 63)
            if not (covered[w] & b):
                covered[w] |= b
                uncovered_count -= 1

        if uncovered_count == 0:
            break

    # Emergency: if not fully covered, add blocks until covered or time runs out
    while uncovered_count > 0 and time.time() < deadline - 5:
        seed_idx = _find_first_uncovered(covered, n_subsets)
        if seed_idx is None or seed_idx >= n_subsets:
            break

        seed = list(all_t_subsets[seed_idx])
        block = seed[:]
        while len(block) < k:
            for x in range(v):
                if x not in block:
                    block.append(x)
                    break
        block.sort()
        blocks.append(block)

        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            w = idx >> 6
            b = 1 << (idx & 63)
            if not (covered[w] & b):
                covered[w] |= b
                uncovered_count -= 1

    # Local search: multiple passes of block removal
    if time.time() < deadline - 8:
        blocks = _local_search(blocks, v, k, t, deadline)

    return blocks


def _find_first_uncovered(covered, n_subsets):
    """Find the first uncovered t-subset index from the covered bitmask array."""
    n_words = len(covered)
    for w in range(n_words):
        word_val = covered[w]
        if word_val != 0xFFFFFFFFFFFFFFFF:
            inv = (~word_val) & 0xFFFFFFFFFFFFFFFF
            if inv:
                bit_pos = (inv & (-inv)).bit_length() - 1
                idx = (w << 6) + bit_pos
                if idx < n_subsets:
                    return idx
        else:
            idx = w << 6
            if idx < n_subsets:
                return idx
    return None


def _local_search(blocks, v, k, t, deadline):
    """Reduce block count by iteratively removing redundant blocks."""
    if not blocks:
        return blocks

    # Precompute all t-subsets for indexing
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(subset_to_idx)

    # Precompute list of t-subset indices for each block
    block_subsets = []
    for block in blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subsets.append(subs)

    # Count coverage
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for subs in block_subsets:
        for idx in subs:
            cover_count[idx] += 1

    # Multiple passes of removal
    for _pass in range(10):
        if time.time() > deadline - 3:
            break

        removed = set()
        for i in range(len(blocks)):
            if i in removed:
                continue

            # Check if block i is the only cover for any subset
            is_critical = False
            for idx in block_subsets[i]:
                if cover_count[idx] == 1:
                    is_critical = True
                    break

            if not is_critical:
                removed.add(i)
                for idx in block_subsets[i]:
                    cover_count[idx] -= 1

        if not removed:
            break

        # Rebuild lists
        new_blocks = []
        new_block_subsets = []
        for i in range(len(blocks)):
            if i not in removed:
                new_blocks.append(blocks[i])
                new_block_subsets.append(block_subsets[i])

        blocks = new_blocks
        block_subsets = new_block_subsets

    # Swap-based local search: try replacing pairs of blocks with a single better block
    if time.time() < deadline - 5:
        blocks = _pair_swap_optimize(blocks, v, k, t, deadline)

    return blocks


def _pair_swap_optimize(blocks, v, k, t, deadline):
    """Try to reduce blocks by swapping pairs: if two blocks can be replaced by one better block."""
    if len(blocks) < 2:
        return blocks

    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_subsets = len(subset_to_idx)

    # Build coverage count
    block_subsets = []
    for block in blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_subsets.append(subs)

    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for subs in block_subsets:
        for idx in subs:
            cover_count[idx] += 1

    n_blocks = len(blocks)
    max_iterations = 200
    for _iter in range(max_iterations):
        if time.time() > deadline - 3:
            break

        improved = False
        for i in range(n_blocks):
            if time.time() > deadline - 3:
                break
            for j in range(i + 1, n_blocks):
                if time.time() > deadline - 3:
                    break

                # Try to find a single block that covers all subsets only covered by blocks i and j
                # Find subsets that would become uncovered if we remove both blocks
                critical_subsets = []
                for idx in block_subsets[i]:
                    if cover_count[idx] <= 2:
                        critical_subsets.append(idx)
                for idx in block_subsets[j]:
                    if cover_count[idx] == 1:
                        critical_subsets.append(idx)
                    elif cover_count[idx] == 2 and idx not in critical_subsets:
                        # Check if it's also in block i
                        pass

                if not critical_subsets:
                    # Both blocks are fully redundant together - remove both
                    # Actually we need to check if there exists a single block covering all their unique subsets
                    pass

                # Find the union of points that must be in the new block
                # The new block must cover all t-subsets that are uniquely covered by i or j
                must_include = set()
                for idx in critical_subsets:
                    for p in all_t_subsets[idx]:
                        must_include.add(p)

                if len(must_include) <= k:
                    # Check if there's a k-subset containing must_include that covers all critical subsets
                    remaining_points = [p for p in range(v) if p not in must_include]
                    need_extra = k - len(must_include)

                    if need_extra >= 0:
                        # Check if must_include itself (padded) covers all critical subsets
                        candidate = list(must_include)
                        for p in remaining_points:
                            if len(candidate) >= k:
                                break
                            candidate.append(p)
                        candidate.sort()

                        # Verify this candidate covers all critical subsets
                        candidate_set = set(candidate)
                        valid = True
                        for idx in critical_subsets:
                            if not set(all_t_subsets[idx]).issubset(candidate_set):
                                valid = False
                                break

                        if valid:
                            # Remove blocks i and j, add candidate
                            # Update cover counts
                            for idx in block_subsets[i]:
                                cover_count[idx] -= 1
                            for idx in block_subsets[j]:
                                cover_count[idx] -= 1
                            for sub in itertools.combinations(candidate, t):
                                idx = subset_to_idx[sub]
                                cover_count[idx] += 1

                            # Verify no subset has negative or zero count
                            ok = True
                            for idx in block_subsets[i] + block_subsets[j]:
                                if cover_count[idx] < 0:
                                    ok = False
                                    break

                            if ok:
                                new_blocks = [blocks[m] for m in range(n_blocks) if m != i and m != j]
                                new_blocks.append(candidate)
                                new_block_subsets = [block_subsets[m] for m in range(n_blocks) if m != i and m != j]
                                new_block_subsets.append([subset_to_idx[sub] for sub in itertools.combinations(candidate, t)])

                                blocks = new_blocks
                                block_subsets = new_block_subsets
                                n_blocks = len(blocks)
                                improved = True
                                break

            if improved:
                break

        if not improved:
            break

    return blocks
# EVOLVE-BLOCK-END
