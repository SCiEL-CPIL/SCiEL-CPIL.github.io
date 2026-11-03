"""Vectorized greedy covering with randomized restarts and local polish for C(v,k,t)."""
import time
import numpy as np
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets as integer indices for fast set operations
    # Map each t-subset to a unique integer index
    # For v=24, t=3: C(24,3) = 2024 subsets
    all_t_subsets = list(combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # For each point x, precompute which t-subsets it participates in
    # point_subsets[x] = list of subset indices containing x
    point_subsets = [[] for _ in range(v)]
    for s in all_t_subsets:
        for x in s:
            point_subsets[x].append(subset_to_idx[s])

    # For a partial block (set of points), compute which t-subsets are covered
    # A t-subset is covered by a block if all t points of the subset are in the block
    # We'll track uncovered subsets as a boolean numpy array

    best_blocks = None
    best_count = n_subsets + 1  # worst case

    def greedy_cover(random_seed=None, max_restarts=1):
        """Run greedy covering with optional randomization."""
        nonlocal best_blocks, best_count, deadline

        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random.RandomState()

        uncovered = np.ones(n_subsets, dtype=np.bool_)
        blocks = []

        while uncovered.any():
            if time.time() > deadline - 2.0:
                break

            # Find an uncovered subset to seed the new block
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break

            # Pick a random uncovered subset (or first one for determinism)
            seed_idx = uncovered_indices[rng.randint(len(uncovered_indices))]
            seed_subset = all_t_subsets[seed_idx]

            # Start block with points from seed subset
            block_set = set(seed_subset)
            block_points = list(seed_subset)

            # Greedily add points to maximize coverage of remaining uncovered subsets
            while len(block_points) < k:
                if time.time() > deadline - 2.0:
                    break

                # For each candidate point not in block, compute how many uncovered
                # t-subsets it would help cover given the current block
                # A t-subset {a,b,c} is covered when the block contains all of {a,b,c}
                # With current block B, adding point x covers t-subsets where
                # {x} union (t-1 points from B) forms the subset, and the subset is uncovered

                best_gain = -1
                best_point = None
                candidates = [x for x in range(v) if x not in block_set]

                for x in candidates:
                    # Count uncovered t-subsets that contain x and (t-1) points from current block
                    gain = 0
                    for idx in point_subsets[x]:
                        if not uncovered[idx]:
                            continue
                        # Check if the other t-1 points of this subset are in block
                        subset = all_t_subsets[idx]
                        other = [p for p in subset if p != x]
                        if all(p in block_set for p in other):
                            gain += 1

                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None:
                    break

                block_set.add(best_point)
                block_points.append(best_point)

            # Finalize block
            if len(block_points) >= t:
                block_points.sort()
                blocks.append(block_points)

                # Update uncovered: mark all t-subsets of this block as covered
                for subset in combinations(block_points, t):
                    uncovered[subset_to_idx[subset]] = False

            # Check time
            if time.time() > deadline - 2.0:
                break

        # Local polish: try to swap points in blocks to reduce total count
        # This is a simple 1-swap improvement
        if blocks and time.time() < deadline - 5.0:
            blocks = local_polish(blocks, v, k, t, all_t_subsets, subset_to_idx, point_subsets, deadline)

        if blocks:
            # Verify completeness
            if not uncovered.any() if 'uncovered' in locals() else True:
                # Re-verify
                covered = np.zeros(n_subsets, dtype=np.bool_)
                for block in blocks:
                    for subset in combinations(block, t):
                        covered[subset_to_idx[subset]] = True
                if covered.all():
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]

        return blocks

    # Run multiple restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        greedy_cover(random_seed=i)

    # Also try deterministic version
    if time.time() < deadline - 5.0:
        greedy_cover(random_seed=None)

    if best_blocks is None:
        # Fallback: return a valid (possibly suboptimal) covering
        # Use a simple construction
        blocks = []
        uncovered = np.ones(n_subsets, dtype=np.bool_)
        while uncovered.any():
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[0]
            seed_subset = all_t_subsets[seed_idx]
            block_set = set(seed_subset)
            block_points = list(seed_subset)
            while len(block_points) < k:
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for idx in point_subsets[x]:
                        if not uncovered[idx]:
                            continue
                        subset = all_t_subsets[idx]
                        other = [p for p in subset if p != x]
                        if all(p in block_set for p in other):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block_set.add(best_point)
                block_points.append(best_point)
            if len(block_points) >= t:
                block_points.sort()
                blocks.append(block_points)
                for subset in combinations(block_points, t):
                    uncovered[subset_to_idx[subset]] = False
        best_blocks = blocks

    return best_blocks

def local_polish(blocks, v, k, t, all_t_subsets, subset_to_idx, point_subsets, deadline):
    """Try 1-swap and 1-delete improvements to reduce block count."""
    import time
    changed = True
    max_iters = 20
    iters = 0

    while changed and iters < max_iters:
        changed = False
        iters += 1
        if time.time() > deadline - 3.0:
            break

        # Try deleting each block if the rest still covers everything
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                break
            remaining = blocks[:i] + blocks[i+1:]
            # Check if remaining blocks cover all t-subsets
            covered = set()
            for block in remaining:
                for subset in combinations(block, t):
                    covered.add(subset)
            if len(covered) == len(all_t_subsets):
                blocks = remaining
                changed = True
                break

        if changed:
            continue

        # Try swapping a point in one block with a point from another block
        for i in range(len(blocks)):
            if time.time() > deadline - 3.0:
                break
            for j in range(len(blocks[i])):
                for l in range(len(blocks)):
                    if l == i:
                        continue
                    for m in range(len(blocks[l])):
                        old_pt = blocks[i][j]
                        new_pt = blocks[l][m]
                        if old_pt == new_pt:
                            continue
                        # Swap
                        blocks[i][j] = new_pt
                        blocks[l][m] = old_pt
                        # Sort
                        blocks[i].sort()
                        blocks[l].sort()

                        # Check validity: no duplicate blocks, and coverage maintained
                        # Quick check: does the new configuration still cover all?
                        # This is expensive; do a partial check
                        covered = set()
                        for block in blocks:
                            for subset in combinations(block, t):
                                covered.add(subset)
                        if len(covered) == len(all_t_subsets):
                            changed = True
                            break
                        else:
                            # Revert
                            blocks[i][j] = old_pt
                            blocks[l][m] = new_pt
                            blocks[i].sort()
                            blocks[l].sort()

                    if changed:
                        break
                if changed:
                    break
            if changed:
                break

    return blocks
