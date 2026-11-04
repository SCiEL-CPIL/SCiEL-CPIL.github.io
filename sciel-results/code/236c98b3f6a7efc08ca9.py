"""Combined covering design: fast greedy with index-based coverage tracking,
plus redundancy removal and swap-based local optimization."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute: for each point, which t-subsets contain it
    # point_to_tsubs[p] = list of indices of t-subsets containing p
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_to_tsubs[p].append(i)

    # For each pair of points, which t-subsets contain both
    # This is used to speed up gain computation
    # Actually, let's use a different approach:
    # For a block B of size m, the t-subsets it covers are all C(B, t)
    # We need to count how many of those are currently uncovered.

    # Use a coverage counter: cov_count[i] = number of blocks covering t-sub i
    cov_count = np.zeros(n_t, dtype=np.int32)
    uncovered_count = n_t
    blocks = []

    def time_ok():
        return time.time() < deadline

    def get_block_tsub_indices(block):
        """Get indices of all t-subsets contained in block."""
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    # Phase 1: Greedy construction with vectorized gain computation
    while uncovered_count > 0 and time_ok():
        # Find the first uncovered t-subset
        first_uncovered = None
        for i in range(n_t):
            if cov_count[i] == 0:
                first_uncovered = i
                break
        if first_uncovered is None:
            break

        base = list(all_t_subs[first_uncovered])
        block = list(base)
        block_set = set(block)

        # Current t-subsets in block
        current_tsubs = list(itertools.combinations(block, t))
        current_tsub_indices = [t_sub_to_idx[sub] for sub in current_tsubs]

        while len(block) < k and time_ok():
            # For each candidate point x not in block, compute gain
            # gain(x) = number of t-subsets in (block ∪ {x}) that are not in block
            #          and are currently uncovered
            # = number of t-subsets containing x and (t-1) points from block
            #   that are currently uncovered

            best_x = -1
            best_gain = -1

            # Precompute t-1 subsets of current block
            t_minus_1_list = list(itertools.combinations(block, t - 1))

            for x in all_points:
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1_list:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and cov_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                # Fallback: fill with arbitrary points
                for x in all_points:
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                break

            block.append(best_x)
            block_set.add(best_x)
            # Update current_tsubs and current_tsub_indices
            current_tsubs = list(itertools.combinations(block, t))
            current_tsub_indices = [t_sub_to_idx[sub] for sub in current_tsubs]

        block.sort()
        blocks.append(block)

        # Mark newly covered subsets
        for idx in get_block_tsub_indices(block):
            if cov_count[idx] == 0:
                uncovered_count -= 1
            cov_count[idx] += 1

    if uncovered_count > 0:
        # Can't finish in time; return what we have (may be invalid)
        return blocks

    # Phase 2: Remove redundant blocks using cov_count
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            block_i = blocks[i]
            block_tsub_indices = get_block_tsub_indices(block_i)
            redundant = True
            for idx in block_tsub_indices:
                if cov_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_tsub_indices:
                    cov_count[idx] -= 1
                blocks.pop(i)
                changed = True
                break

    # Phase 3: Swap-based local optimization
    if time_ok():
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(blocks)):
                if not time_ok():
                    break
                block_i = list(blocks[i])
                block_i_set = set(block_i)
                old_tsub_indices = get_block_tsub_indices(block_i)
                # Count how many of these are uniquely covered (cov_count == 1)
                unique_count = sum(1 for idx in old_tsub_indices if cov_count[idx] == 1)

                for pos in range(k):
                    if not time_ok():
                        break
                    old_pt = block_i[pos]
                    for new_pt in all_points:
                        if new_pt == old_pt or new_pt in block_i_set:
                            continue
                        candidate = block_i[:]
                        candidate[pos] = new_pt
                        candidate.sort()
                        cand_set = set(candidate)

                        new_tsub_indices = get_block_tsub_indices(candidate)
                        # Count newly covered (currently uncovered) subsets
                        new_gain = sum(1 for idx in new_tsub_indices if cov_count[idx] == 0)
                        # Count lost uniquely covered subsets
                        lost_unique = 0
                        for idx in old_tsub_indices:
                            if cov_count[idx] == 1 and idx not in new_tsub_indices:
                                lost_unique += 1

                        if new_gain > lost_unique:
                            # Apply swap
                            # Decrement old block's coverage
                            for idx in old_tsub_indices:
                                cov_count[idx] -= 1
                            # Increment new block's coverage
                            for idx in new_tsub_indices:
                                if cov_count[idx] == 0:
                                    uncovered_count -= 1
                                cov_count[idx] += 1
                            blocks[i] = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    return blocks
# EVOLVE-BLOCK-END
