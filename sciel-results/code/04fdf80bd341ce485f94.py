# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin before 160s limit (using 150s as stated)
    # Actually the task says 120s evaluator, 150s kill. Let's use 145s to be safe.
    deadline = start_time + 145.0

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # covered[i] = 1 if t-subset i is covered
    covered = bytearray(n_t)
    uncovered_count = n_t

    # For fast gain computation: for each block (as a frozenset of points), 
    # we need to know which t-subsets it covers.
    # Precompute: for each point, list of t-subsets containing it? Too big.
    # Instead, when building a block, compute gain by checking which t-subsets 
    # formed by (t-1) elements of current block + candidate x are uncovered.

    def time_ok():
        return time.time() < deadline

    def count_uncovered_in_block(block_set):
        """Count how many t-subsets of block_set are still uncovered."""
        cnt = 0
        for sub in itertools.combinations(block_set, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                cnt += 1
        return cnt

    def gain(x, block):
        """How many new t-subsets would be covered by adding x to block."""
        cnt = 0
        for rest in itertools.combinations(block, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                cnt += 1
        return cnt

    blocks = []

    # Phase 1: Greedy set cover with block construction
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset to start a new block
        # Use the first uncovered one for determinism, or random for diversity
        # Let's try: pick uncovered t-subset that has fewest "potential" blocks? 
        # For simplicity, pick the first uncovered.
        # Actually, let's pick one that maximizes something? Just first for speed.
        
        # Find first uncovered t-subset
        start_sub = None
        for i in range(n_t):
            if not covered[i]:
                start_sub = all_t_subs[i]
                break
        if start_sub is None:
            break

        # Start block with all elements of start_sub
        block = list(start_sub)
        block_set = set(block)

        # Greedily add points that maximize coverage of uncovered t-subsets
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
            if not covered[idx]:
                covered[idx] = 1
                uncovered_count -= 1

    # Phase 2: Redundancy removal - remove blocks whose t-subsets are all covered by other blocks
    if time_ok():
        n_blocks = len(blocks)
        # For each block, check if removing it would leave any t-subset uncovered
        # i.e., for each t-subset in the block, is there another block covering it?
        to_remove = []
        for bi in range(n_blocks):
            if not time_ok():
                break
            block = blocks[bi]
            redundant = True
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                # Check if any other block covers this t-subset
                found = False
                for bj in range(n_blocks):
                    if bj == bi:
                        continue
                    if all(p in blocks[bj] for p in sub):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)

        if to_remove:
            to_remove_set = set(to_remove)
            blocks = [b for i, b in enumerate(blocks) if i not in to_remove_set]

    # Phase 3: Local optimization - try swapping points in blocks to reduce count
    # This is expensive, so only do a few iterations if time permits
    if time_ok():
        # Try to merge or optimize: for each pair of blocks, see if we can
        # replace them with fewer blocks. This is complex, so skip for now.
        # Instead, try random restarts with different random seeds for block construction
        pass

    return blocks

# EVOLVE-BLOCK-END
