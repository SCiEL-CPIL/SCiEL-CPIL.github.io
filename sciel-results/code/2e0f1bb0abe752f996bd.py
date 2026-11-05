# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    # Evaluator kills at 150s; keep a safety margin.
    deadline = start_time + 145.0

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Bitmask representation for fast coverage updates
    # Each block is a bitmask of v bits
    # covered[i] = number of blocks covering t-subset i (for redundancy removal)
    covered_count = np.zeros(n_t, dtype=np.int32)
    uncovered_count = n_t

    def time_ok():
        return time.time() < deadline

    def block_to_mask(block):
        m = 0
        for x in block:
            m |= (1 << x)
        return m

    def mask_to_block(m):
        return [i for i in range(v) if (m >> i) & 1]

    def update_coverage(block, delta):
        """Update covered_count for all t-subsets of block by delta (+1 or -1)."""
        nonlocal uncovered_count
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            old = covered_count[idx]
            new = old + delta
            covered_count[idx] = new
            if old == 0 and new > 0:
                uncovered_count -= 1
            elif old > 0 and new == 0:
                uncovered_count += 1

    def gain(x, block):
        """How many new t-subsets would be covered by adding x to block."""
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx[sub]
            if covered_count[idx] == 0:
                cnt += 1
        return cnt

    def greedy_build_block(start_sub):
        """Build a block greedily starting from start_sub."""
        block = list(start_sub)
        block_set = set(block)
        while len(block) < k and time_ok():
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block_set:
                    continue
                g = gain(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None or best_gain <= 0:
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
        return block

    def redundancy_remove(blocks):
        """Remove blocks whose t-subsets are all covered by other blocks."""
        n_blocks = len(blocks)
        if n_blocks <= 1:
            return blocks
        to_remove = []
        for bi in range(n_blocks):
            if not time_ok():
                break
            block = blocks[bi]
            redundant = True
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if covered_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
        if to_remove:
            to_remove_set = set(to_remove)
            new_blocks = []
            for i, b in enumerate(blocks):
                if i in to_remove_set:
                    update_coverage(b, -1)
                else:
                    new_blocks.append(b)
            return new_blocks
        return blocks

    def local_optimize(blocks, max_iters=50):
        """Try to improve by swapping points in blocks."""
        for _ in range(max_iters):
            if not time_ok():
                break
            improved = False
            for bi in range(len(blocks)):
                if not time_ok():
                    break
                block = blocks[bi]
                block_set = set(block)
                # Try swapping each point in block with each point not in block
                best_swap = None
                best_gain = 0
                for i, x in enumerate(block):
                    for y in all_points:
                        if y in block_set:
                            continue
                        new_block = block[:]
                        new_block[i] = y
                        new_block.sort()
                        # Compute gain: how many uncovered t-subsets does new_block cover
                        # that old_block didn't
                        gain_new = 0
                        for sub in itertools.combinations(new_block, t):
                            idx = t_sub_to_idx[sub]
                            if covered_count[idx] == 0:
                                gain_new += 1
                        # How many t-subsets of old_block become uncovered?
                        loss = 0
                        for sub in itertools.combinations(block, t):
                            idx = t_sub_to_idx[sub]
                            if covered_count[idx] == 1 and all(p in new_block for p in sub):
                                # This subset is covered by old_block and will be in new_block
                                pass
                            elif covered_count[idx] == 1:
                                # Check if new_block also covers it
                                if not all(p in new_block for p in sub):
                                    loss += 1
                        net_gain = gain_new - loss
                        if net_gain > best_gain:
                            best_gain = net_gain
                            best_swap = (i, x, y)
                if best_swap and best_gain > 0:
                    i, x, y = best_swap
                    update_coverage(block, -1)
                    block[i] = y
                    block.sort()
                    update_coverage(block, +1)
                    improved = True
            if not improved:
                break
        return blocks

    # Phase 1: Greedy set cover with block construction
    blocks = []
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset
        start_sub = None
        for i in range(n_t):
            if covered_count[i] == 0:
                start_sub = all_t_subs[i]
                break
        if start_sub is None:
            break

        block = greedy_build_block(start_sub)
        blocks.append(block)
        update_coverage(block, +1)

    # Phase 2: Redundancy removal
    if time_ok():
        blocks = redundancy_remove(blocks)

    # Phase 3: Local optimization
    if time_ok():
        blocks = local_optimize(blocks, max_iters=30)

    # Phase 4: Second pass redundancy removal
    if time_ok():
        blocks = redundancy_remove(blocks)

    return blocks

# EVOLVE-BLOCK-END
