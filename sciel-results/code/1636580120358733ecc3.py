"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute: for each point, which t-subsets contain it
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 25

    for restart in range(n_restarts):
        if not time_ok():
            break

        random.seed(12345 + restart * 7919)
        np.random.seed(12345 + restart * 7919)

        covered = np.zeros(n_t, dtype=bool)
        covered_count = 0
        blocks = []

        # Greedy construction with improved block selection
        while covered_count < n_t and time_ok():
            # Find uncovered t-subset indices efficiently
            uncovered_mask = ~covered
            n_uncovered = np.count_nonzero(uncovered_mask)
            if n_uncovered == 0:
                break

            uncovered_indices = np.where(uncovered_mask)[0]

            # Compute point scarcity: how many uncovered t-subsets contain each point
            point_uncovered_count = np.zeros(v, dtype=np.int32)
            for idx in uncovered_indices:
                for pt in all_t_subsets[idx]:
                    point_uncovered_count[pt] += 1

            # Select seed candidates: prioritize t-subsets whose points are scarce
            if len(uncovered_indices) <= 20:
                seed_candidates = uncovered_indices.tolist()
            else:
                # Weight by inverse scarcity sum
                weights = np.zeros(len(uncovered_indices), dtype=np.float64)
                for i, idx in enumerate(uncovered_indices):
                    w = 0.0
                    for pt in all_t_subsets[idx]:
                        w += 1.0 / max(1, point_uncovered_count[pt])
                    weights[i] = w
                total_w = weights.sum()
                if total_w == 0:
                    seed_candidates = [random.choice(uncovered_indices.tolist())]
                else:
                    n_samples = min(12, len(uncovered_indices))
                    seed_candidates = []
                    for _ in range(n_samples):
                        r = random.random() * total_w
                        cum = 0.0
                        for i, w in enumerate(weights):
                            cum += w
                            if cum >= r:
                                seed_candidates.append(int(uncovered_indices[i]))
                                break
                    if not seed_candidates:
                        seed_candidates = [random.choice(uncovered_indices.tolist())]

            # For each seed, build a block and compute its gain
            best_block = None
            best_gain = -1
            best_block_tsubs = None

            for seed_idx in seed_candidates:
                if not time_ok():
                    break
                seed_subset = all_t_subsets[seed_idx]
                block_set = set(seed_subset)

                # Extend block greedily
                while len(block_set) < k and time_ok():
                    block_tuple = tuple(sorted(block_set))
                    candidates_pts = [x for x in all_points if x not in block_set]
                    random.shuffle(candidates_pts)

                    best_x = None
                    best_x_gain = -1

                    for x in candidates_pts:
                        gain = 0
                        for rest in itertools.combinations(block_tuple, t - 1):
                            sub = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx[sub]
                            if not covered[idx]:
                                gain += 1
                        tie = random.random() * 0.005
                        if gain + tie > best_x_gain:
                            best_x_gain = gain + tie
                            best_x = x

                    if best_x is None:
                        break
                    block_set.add(best_x)

                # Fill remaining spots randomly
                if len(block_set) < k:
                    remaining = [x for x in all_points if x not in block_set]
                    random.shuffle(remaining)
                    for x in remaining[:k - len(block_set)]:
                        block_set.add(x)

                block = tuple(sorted(block_set))

                # Calculate total gain and collect covered t-subset indices
                gain = 0
                block_tsubs = []
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    block_tsubs.append(idx)
                    if not covered[idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    best_block_tsubs = block_tsubs

            if best_block is None or best_gain <= 0:
                # Fallback: random block
                block_set = set(random.sample(all_points, k))
                best_block = tuple(sorted(block_set))
                best_block_tsubs = []
                best_gain = 0
                for sub in itertools.combinations(best_block, t):
                    idx = t_sub_to_idx[sub]
                    best_block_tsubs.append(idx)
                    if not covered[idx]:
                        best_gain += 1

            if best_gain <= 0:
                break

            # Apply the block
            for idx in best_block_tsubs:
                if not covered[idx]:
                    covered[idx] = True
                    covered_count += 1
            blocks.append(best_block)

        # Deletion phase: remove redundant blocks
        if time_ok() and covered_count == n_t:
            tsub_block_count = [0] * n_t
            for b in blocks:
                for sub in itertools.combinations(b, t):
                    idx = t_sub_to_idx[sub]
                    tsub_block_count[idx] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    removable = True
                    for sub in itertools.combinations(blocks[i], t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for sub in itertools.combinations(blocks[i], t):
                            idx = t_sub_to_idx[sub]
                            tsub_block_count[idx] -= 1
                        blocks.pop(i)
                        changed = True
                        break

            # Local search: try replacing each block with a better one
            if time_ok():
                for i in range(len(blocks)):
                    if not time_ok():
                        break
                    # Find t-subsets uniquely covered by block i
                    unique_tsubs = []
                    for sub in itertools.combinations(blocks[i], t):
                        idx = t_sub_to_idx[sub]
                        if tsub_block_count[idx] == 1:
                            unique_tsubs.append(idx)

                    if not unique_tsubs:
                        continue

                    # Try to find a replacement block that covers all unique t-subsets
                    # and has better overall coverage
                    required_points = set()
                    for idx in unique_tsubs:
                        for pt in all_t_subsets[idx]:
                            required_points.add(pt)

                    if len(required_points) > k:
                        continue

                    # Build candidate replacement blocks
                    base_block = list(required_points)
                    extra_points = [x for x in all_points if x not in required_points]
                    random.shuffle(extra_points)

                    # Try several completions
                    best_rep = None
                    best_rep_gain = -1
                    n_tries = min(5, len(extra_points) + 1)
                    for _ in range(n_tries):
                        rep_block = list(base_block)
                        if len(rep_block) < k:
                            rep_block.extend(extra_points[:k - len(rep_block)])
                        rep_block = tuple(sorted(rep_block))

                        # Calculate gain of replacement
                        rep_gain = 0
                        for sub in itertools.combinations(rep_block, t):
                            idx = t_sub_to_idx[sub]
                            if not covered[idx] or tsub_block_count[idx] == 1:
                                rep_gain += 1

                        if rep_gain > best_rep_gain:
                            best_rep_gain = rep_gain
                            best_rep = rep_block

                    if best_rep is not None and best_rep != blocks[i]:
                        # Check if replacement is valid
                        valid = True
                        for idx in unique_tsubs:
                            sub = all_t_subsets[idx]
                            if not all(pt in best_rep for pt in sub):
                                valid = False
                                break

                        if valid:
                            # Apply replacement
                            for sub in itertools.combinations(blocks[i], t):
                                idx = t_sub_to_idx[sub]
                                tsub_block_count[idx] -= 1
                            for sub in itertools.combinations(best_rep, t):
                                idx = t_sub_to_idx[sub]
                                tsub_block_count[idx] += 1
                            blocks[i] = best_rep

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        # Early stop if we've done very well
        if best_count <= 145:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
