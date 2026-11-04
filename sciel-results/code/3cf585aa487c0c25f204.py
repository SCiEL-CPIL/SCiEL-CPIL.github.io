# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Track uncovered t-subsets as a set for O(1) discard and membership
    uncovered = set(range(n_t))
    blocks = []

    def time_ok():
        return time.time() < deadline

    def count_uncovered_in_block(block_set):
        """Count how many t-subsets of block_set are still uncovered."""
        cnt = 0
        for sub in itertools.combinations(block_set, t):
            idx = t_sub_to_idx[sub]
            if idx in uncovered:
                cnt += 1
        return cnt

    def gain(x, block):
        """How many new t-subsets would be covered by adding x to block."""
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx[sub]
            if idx in uncovered:
                cnt += 1
        return cnt

    # Phase 1: Greedy set cover with block construction
    while uncovered and time_ok():
        # Find an uncovered t-subset to start a new block
        # Pick the one with fewest "supporting" uncovered t-subsets for diversity
        # For speed, just pick the minimum index (deterministic)
        first_idx = min(uncovered)
        start_sub = all_t_subs[first_idx]

        # Start block with all elements of start_sub
        block = list(start_sub)
        block_set = set(block)

        # Greedily add points that maximize coverage of uncovered t-subsets
        while len(block) < k and time_ok():
            best_x = -1
            best_gain_val = -1
            # Precompute t-1 subsets of current block
            t_minus_1_list = list(itertools.combinations(block, t - 1))

            for x in all_points:
                if x in block_set:
                    continue
                g = 0
                for rest in t_minus_1_list:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and idx in uncovered:
                        g += 1
                if g > best_gain_val:
                    best_gain_val = g
                    best_x = x

            if best_x == -1 or best_gain_val <= 0:
                # Can't improve, fill with arbitrary points
                for x in all_points:
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                        if len(block) == k:
                            break
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update coverage
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            uncovered.discard(idx)

    if uncovered:
        # Can't finish in time; return what we have (may be invalid)
        return blocks

    # Phase 2: Iterative redundancy removal
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            block_i = blocks[i]
            redundant = True
            for sub in itertools.combinations(block_i, t):
                idx = t_sub_to_idx[sub]
                found = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if all(p in blocks[j] for p in sub):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                changed = True
                break  # Restart scan since indices shifted

    # Phase 3: Swap-based local optimization
    # Try swapping one point in a block for another to free up coverage
    # so that another block can be removed.
    if time_ok():
        improved = True
        max_iterations = 500
        iters = 0
        while improved and time_ok() and iters < max_iterations:
            improved = False
            iters += 1
            n_blocks = len(blocks)
            for i in range(n_blocks):
                if not time_ok():
                    break
                block_i = list(blocks[i])
                for pos in range(k):
                    old_pt = block_i[pos]
                    for new_pt in all_points:
                        if new_pt == old_pt or new_pt in block_i:
                            continue
                        candidate = block_i[:]
                        candidate[pos] = new_pt
                        candidate.sort()
                        candidate_set = set(candidate)

                        # Check: does the candidate block still cover all t-subsets
                        # that block_i uniquely covers?
                        # i.e., for each t-subset in candidate, if it was only covered
                        # by block_i, then candidate must still cover it.
                        # Actually, we need: every t-subset in block_i that is NOT
                        # covered by any other block must also be in candidate.
                        unique_to_i = []
                        for sub in itertools.combinations(block_i, t):
                            covered_elsewhere = False
                            for j in range(n_blocks):
                                if j == i:
                                    continue
                                if all(p in blocks[j] for p in sub):
                                    covered_elsewhere = True
                                    break
                            if not covered_elsewhere:
                                unique_to_i.append(sub)

                        # Check if candidate covers all unique_to_i subsets
                        valid_swap = True
                        for sub in unique_to_i:
                            if not all(p in candidate_set for p in sub):
                                valid_swap = False
                                break

                        if not valid_swap:
                            continue

                        # Apply the swap
                        blocks[i] = candidate

                        # Now try to remove any redundant blocks
                        changed2 = True
                        while changed2 and time_ok():
                            changed2 = False
                            for ri in range(len(blocks)):
                                if not time_ok():
                                    break
                                block_ri = blocks[ri]
                                redundant2 = True
                                for sub2 in itertools.combinations(block_ri, t):
                                    found2 = False
                                    for rj in range(len(blocks)):
                                        if rj == ri:
                                            continue
                                        if all(p in blocks[rj] for p in sub2):
                                            found2 = True
                                            break
                                    if not found2:
                                        redundant2 = False
                                        break
                                if redundant2:
                                    blocks.pop(ri)
                                    changed2 = True
                                    break

                        improved = True
                        break  # Restart outer loop after a successful swap
                    if improved:
                        break
                if improved:
                    break

    return blocks

# EVOLVE-BLOCK-END
