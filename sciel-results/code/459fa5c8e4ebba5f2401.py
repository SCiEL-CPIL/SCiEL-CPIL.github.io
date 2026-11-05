"""Covering design C(v,k,t) via bitmask greedy + swap-based local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # For each k-block, precompute its t-subset indices (lazy)
    # We'll compute on the fly

    def time_ok():
        return time.time() < deadline

    # covered: list of cover counts per t-subset
    cover_count = [0] * n_t
    uncovered_count = n_t

    blocks = []  # list of sorted tuples
    block_t_indices = []  # for each block, list of t-subset indices it covers

    def compute_block_t_indices(block):
        """Return list of t-subset indices covered by this block."""
        result = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            result.append(idx)
        return result

    def add_block(block):
        """Add a block, update cover_count and uncovered_count."""
        nonlocal uncovered_count
        block = tuple(sorted(block))
        blocks.append(block)
        t_indices = compute_block_t_indices(block)
        block_t_indices.append(t_indices)
        for idx in t_indices:
            if cover_count[idx] == 0:
                uncovered_count -= 1
            cover_count[idx] += 1

    def remove_block(b_idx):
        """Remove a block, update cover_count and uncovered_count."""
        nonlocal uncovered_count
        for idx in block_t_indices[b_idx]:
            cover_count[idx] -= 1
            if cover_count[idx] == 0:
                uncovered_count += 1
        blocks.pop(b_idx)
        block_t_indices.pop(b_idx)

    # Phase 1: Weighted Greedy Construction
    # Maintain a weight for each t-subset: initially 1, but we can boost
    # rare ones. Actually, standard greedy: pick block covering most uncovered.
    # To speed up, we build block by adding points one at a time.

    # Precompute: for each point x and each (t-1)-subset S, the t-subset S ∪ {x}
    # This lets us quickly compute gain when adding x to a partial block.

    # For efficiency, precompute for each point x, the list of t-subsets containing x
    # point_tsubs[x] = list of (t-subset_tuple, index) where x is in the t-subset
    point_tsubs = [[] for _ in range(v)]
    for sub, idx in zip(all_t_subs, range(n_t)):
        for x in sub:
            point_tsubs[x].append((sub, idx))

    while uncovered_count > 0 and time_ok():
        # Find first uncovered t-subset
        start_idx = -1
        for i in range(n_t):
            if cover_count[i] == 0:
                start_idx = i
                break
        if start_idx == -1:
            break

        start_sub = all_t_subs[start_idx]
        block = list(start_sub)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count uncovered t-subsets in (block ∪ {x}) that contain x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(sub)
                    if idx is not None and cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        add_block(tuple(block))

    # Phase 2: Reduction - remove redundant blocks
    if time_ok():
        changed = True
        while changed and time_ok():
            changed = False
            for b_idx in range(len(blocks) - 1, -1, -1):
                if not time_ok():
                    break
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b_idx)
                    changed = True
                    break  # restart scan

    # Phase 3: Swap-based local search
    # For each block, try swapping one point for another.
    # Accept swap if it doesn't lose coverage of any t-subset and gains coverage.
    if time_ok() and len(blocks) > 0:
        # We'll iterate: for each block, try all single-point swaps
        # A swap is valid if:
        #   - For every t-subset in old block, it's still covered by other blocks OR by new block
        #   - New block covers at least as many uncovered t-subsets as old block did
        # Simplification: accept swap if new block's set of t-subsets is a superset
        # of old block's t-subsets that are uniquely covered by old block.

        improved = True
        while improved and time_ok():
            improved = False
            for b_idx in range(len(blocks)):
                if not time_ok():
                    break
                old_block = blocks[b_idx]
                old_t_set = set(block_t_indices[b_idx])
                old_unique = set()
                for idx in old_t_set:
                    if cover_count[idx] == 1:
                        old_unique.add(idx)

                # Try swapping each point in old_block with each point not in old_block
                best_swap = None
                best_gain = 0
                for i, old_x in enumerate(old_block):
                    for new_x in range(v):
                        if new_x in old_block:
                            continue
                        # Form new block
                        new_block_list = list(old_block)
                        new_block_list[i] = new_x
                        new_block = tuple(sorted(new_block_list))

                        # Compute new t-subsets
                        new_t_indices = compute_block_t_indices(new_block)
                        new_t_set = set(new_t_indices)

                        # Check: all uniquely-covered t-subsets must still be covered
                        if not old_unique.issubset(new_t_set):
                            continue

                        # Count new uncovered t-subsets covered
                        new_gain = 0
                        for idx in new_t_set:
                            if cover_count[idx] == 0:
                                new_gain += 1

                        if new_gain > best_gain:
                            best_gain = new_gain
                            best_swap = (i, old_x, new_x, new_block, new_t_indices)

                if best_swap is not None and best_gain > 0:
                    i, old_x, new_x, new_block, new_t_indices = best_swap
                    # Apply swap: remove old block, add new block
                    remove_block(b_idx)
                    add_block(new_block)
                    improved = True
                    # After swap, do a quick reduction pass
                    if time_ok():
                        sub_changed = True
                        while sub_changed and time_ok():
                            sub_changed = False
                            for rb_idx in range(len(blocks) - 1, -1, -1):
                                if not time_ok():
                                    break
                                redundant = True
                                for idx in block_t_indices[rb_idx]:
                                    if cover_count[idx] <= 1:
                                        redundant = False
                                        break
                                if redundant:
                                    remove_block(rb_idx)
                                    sub_changed = True
                                    break

    # Final reduction pass
    if time_ok():
        changed = True
        while changed and time_ok():
            changed = False
            for b_idx in range(len(blocks) - 1, -1, -1):
                if not time_ok():
                    break
                redundant = True
                for idx in block_t_indices[b_idx]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block(b_idx)
                    changed = True
                    break

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
