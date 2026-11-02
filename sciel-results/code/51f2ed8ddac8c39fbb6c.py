"""Optimized covering design search for C(24,7,3).

Strategy:
1. Fast greedy construction with vectorized gain computation.
2. Aggressive reduction (remove redundant blocks iteratively).
3. Local search: block swap, replacement, and perturbation to reduce block count.
"""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}

    # For each point, precompute mapping from (t-1)-subset to list of t-subset indices
    # This allows fast gain computation
    point_tmap = [dict() for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for i in range(t):
            p = subset[i]
            rest = frozenset(subset[:i] + subset[i+1:])
            if rest not in point_tmap[p]:
                point_tmap[p][rest] = []
            point_tmap[p][rest].append(idx)

    def greedy_cover(seed, rng=None):
        """Greedy covering with randomization."""
        if rng is None:
            rng = random.Random(seed)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_set = set(range(n_t_subsets))
        blocks = []

        while uncovered_set:
            if time.time() > deadline - 15:
                return None

            # Pick a random uncovered t-subset
            start_idx = rng.choice(list(uncovered_set))
            uncovered_set.discard(start_idx)
            uncovered[start_idx] = False

            block = list(all_t_subsets[start_idx])
            rng.shuffle(block)
            block_set = set(block)

            # Add points to the block greedily
            while len(block) < k:
                if time.time() > deadline - 15:
                    return None

                best_x = -1
                best_gain = -1

                # Precompute (t-1)-subsets of current block
                block_t1_subsets = list(itertools.combinations(block, t - 1))
                block_t1_frozensets = [frozenset(s) for s in block_t1_subsets]

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for fs in block_t1_frozensets:
                        idxs = point_tmap[x].get(fs)
                        if idxs:
                            for idx in idxs:
                                if uncovered[idx]:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        if rng.random() < 0.3:
                            best_x = x

                if best_x < 0:
                    return None

                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_set.discard(idx)

        return blocks

    def reduce_blocks(blocks):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return []

        blocks = list(blocks)
        improved = True

        while improved:
            if time.time() > deadline - 8:
                break
            improved = False

            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] += 1

            # Try removing each block
            for bi in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break

                block = blocks[bi]
                can_remove = True
                affected_indices = []
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] -= 1
                    affected_indices.append(idx)
                    if coverage_count[idx] == 0:
                        can_remove = False
                        break

                if can_remove:
                    blocks.pop(bi)
                    improved = True
                    break
                else:
                    for a_idx in affected_indices:
                        coverage_count[a_idx] += 1

        return blocks

    def local_search(blocks):
        """Try to reduce block count via swap: replace two blocks with one better block."""
        if not blocks:
            return blocks

        blocks = list(blocks)
        improved = True
        max_iterations = 50

        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved = False

            n = len(blocks)
            if n < 2:
                break

            # Compute coverage counts
            coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
            for block in blocks:
                for subset in itertools.combinations(block, t):
                    idx = t_subset_to_idx[subset]
                    coverage_count[idx] += 1

            # For each pair of blocks, try to find a single block that covers all
            # t-subsets uniquely covered by these two blocks
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break

                    # Get t-subsets uniquely covered by block i or block j
                    unique_subsets = set()
                    for subset in itertools.combinations(blocks[i], t):
                        idx = t_subset_to_idx[subset]
                        if coverage_count[idx] == 1:
                            unique_subsets.add(idx)
                    for subset in itertools.combinations(blocks[j], t):
                        idx = t_subset_to_idx[subset]
                        if coverage_count[idx] == 1:
                            unique_subsets.add(idx)

                    if not unique_subsets:
                        # Both blocks are redundant? Remove one
                        blocks.pop(j)
                        improved = True
                        break
                        continue

                    # Try to find a k-subset that covers all unique_subsets
                    # Start with the union of points from both blocks
                    union_points = set(blocks[i]) | set(blocks[j])
                    if len(union_points) <= k:
                        candidate = sorted(union_points)
                        if len(candidate) == k:
                            # Check if this candidate covers all unique_subsets
                            cand_set = set(candidate)
                            ok = True
                            for idx in unique_subsets:
                                sub = all_t_subsets[idx]
                                if not all(p in cand_set for p in sub):
                                    ok = False
                                    break
                            if ok:
                                # Verify no other block is affected negatively
                                # Remove both i and j, add candidate
                                new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                                # Check full coverage
                                cov = np.zeros(n_t_subsets, dtype=np.int32)
                                for b in new_blocks:
                                    for s in itertools.combinations(b, t):
                                        cov[t_subset_to_idx[s]] += 1
                                if np.all(cov > 0):
                                    blocks = new_blocks
                                    blocks.append(candidate)
                                    improved = True
                                    break
                                    continue

                    # Try adding extra points to union to cover missing subsets
                    if len(union_points) < k:
                        # Need to add (k - len(union_points)) more points
                        # Greedily pick points that cover most unique subsets
                        remaining_points = [p for p in range(v) if p not in union_points]
                        candidate = list(union_points)
                        cand_set = set(candidate)
                        remaining_needed = set(unique_subsets)

                        for _ in range(k - len(union_points)):
                            best_p = -1
                            best_gain = -1
                            for p in remaining_points:
                                if p in cand_set:
                                    continue
                                gain = 0
                                new_cand_set = cand_set | {p}
                                for idx in remaining_needed:
                                    sub = all_t_subsets[idx]
                                    if all(pt in new_cand_set for pt in sub):
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_p = p
                            if best_p < 0:
                                break
                            candidate.append(best_p)
                            cand_set.add(best_p)
                            remaining_points.remove(best_p)
                            # Update remaining_needed
                            remaining_needed = set()
                            for idx in unique_subsets:
                                sub = all_t_subsets[idx]
                                if not all(pt in cand_set for pt in sub):
                                    remaining_needed.add(idx)

                        if len(candidate) == k and not remaining_needed:
                            candidate.sort()
                            # Verify full coverage
                            new_blocks = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                            cov = np.zeros(n_t_subsets, dtype=np.int32)
                            for b in new_blocks:
                                for s in itertools.combinations(b, t):
                                    cov[t_subset_to_idx[s]] += 1
                            if np.all(cov > 0):
                                blocks = new_blocks
                                blocks.append(candidate)
                                improved = True
                                break
                                continue

                if improved:
                    break

        return blocks

    def perturb_and_reduce(blocks, rng):
        """Perturb the covering by removing random blocks and greedily fixing gaps, then reduce."""
        if len(blocks) < 2:
            return blocks

        blocks = list(blocks)
        n_remove = max(1, len(blocks) // 10)
        removed = rng.sample(range(len(blocks)), n_remove)
        removed.sort(reverse=True)
        for idx in removed:
            blocks.pop(idx)

        # Greedily fix gaps
        uncovered = np.zeros(n_t_subsets, dtype=bool)
        for block in blocks:
            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                uncovered[idx] = False

        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            return reduce_blocks(blocks)

        for idx in uncovered_indices:
            if time.time() > deadline - 10:
                return blocks
            subset = all_t_subsets[idx]
            subset_set = set(subset)
            # Find existing blocks that contain most of this subset
            best_block = None
            best_overlap = -1
            for block in blocks:
                overlap = len(set(block) & subset_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_block = block

            if best_block is not None and best_overlap >= t - 1:
                # Extend the block
                block = list(best_block)
                block_set = set(block)
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    block_t1_frozensets = [frozenset(s) for s in itertools.combinations(block, t - 1)]
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for fs in block_t1_frozensets:
                            idxs = point_tmap[x].get(fs)
                            if idxs:
                                for i2 in idxs:
                                    if uncovered[i2]:
                                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x < 0:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                if len(block) == k:
                    block.sort()
                    # Replace the old block
                    for bi, b in enumerate(blocks):
                        if b == best_block:
                            blocks[bi] = block
                            break
                    # Update uncovered
                    for s in itertools.combinations(block, t):
                        i2 = t_subset_to_idx[s]
                        uncovered[i2] = False
            else:
                # Create a new block
                block = list(subset)
                block_set = set(block)
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    block_t1_frozensets = [frozenset(s) for s in itertools.combinations(block, t - 1)]
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for fs in block_t1_frozensets:
                            idxs = point_tmap[x].get(fs)
                            if idxs:
                                for i2 in idxs:
                                    if uncovered[i2]:
                                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x < 0:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                if len(block) == k:
                    block.sort()
                    blocks.append(block)
                    for s in itertools.combinations(block, t):
                        i2 = t_subset_to_idx[s]
                        uncovered[i2] = False

        return reduce_blocks(blocks)

    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy constructions
    seed = 0
    while time.time() < deadline - 30:
        result = greedy_cover(seed)
        if result is not None:
            reduced = reduce_blocks(result)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
        seed += 1

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = np.ones(n_t_subsets, dtype=bool)
        uncovered_set = set(range(n_t_subsets))
        blocks = []

        while uncovered_set:
            start_idx = min(uncovered_set)
            uncovered_set.discard(start_idx)
            uncovered[start_idx] = False

            block = list(all_t_subsets[start_idx])
            block_set = set(block)

            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_t1_frozensets = [frozenset(s) for s in itertools.combinations(block, t - 1)]
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for fs in block_t1_frozensets:
                        idxs = point_tmap[x].get(fs)
                        if idxs:
                            for idx in idxs:
                                if uncovered[idx]:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)

            if len(block) < k:
                break

            block.sort()
            blocks.append(block)

            for subset in itertools.combinations(block, t):
                idx = t_subset_to_idx[subset]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_set.discard(idx)

        best_blocks = blocks

    # Phase 2: Local search on best result
    if best_blocks is not None and time.time() < deadline - 10:
        local = local_search(best_blocks)
        if len(local) < len(best_blocks):
            best_blocks = local

    # Phase 3: Perturbation and reduction
    rng = random.Random(42)
    perturb_time = deadline - 5
    while time.time() < perturb_time:
        perturbed = perturb_and_reduce(best_blocks, rng)
        if len(perturbed) < len(best_blocks):
            best_blocks = perturbed
            # Run local search on the improved result
            if time.time() < perturb_time - 5:
                local = local_search(best_blocks)
                if len(local) < len(best_blocks):
                    best_blocks = local
        else:
            # No improvement, try a different perturbation strength
            continue

    return best_blocks

# EVOLVE-BLOCK-END
