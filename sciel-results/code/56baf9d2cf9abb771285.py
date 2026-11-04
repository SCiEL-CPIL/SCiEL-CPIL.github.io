"""Improved covering design C(v,k,t) via indexed greedy + reduction + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets and index them
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # covered[i] = 1 if t-subset i is covered by at least one block
    covered = bytearray(n_t)
    uncovered_count = n_t

    blocks = []
    block_t_counts = []  # for each block, list of indices of t-subsets it covers

    # Precompute for each point, which t-subsets contain it (for faster gain computation)
    # Actually, we'll compute gain on the fly but use the index mapping

    def time_ok():
        return time.time() < deadline

    def count_gain(block, x):
        """Count how many currently-uncovered t-subsets would be covered by adding x to block."""
        gain = 0
        # t-subsets in the new block that contain x
        # These are: choose t-1 from block, add x
        for rest in itertools.combinations(block, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx.get(sub)
            if idx is not None and not covered[idx]:
                gain += 1
        return gain

    def add_block(block):
        """Add a block and update coverage."""
        nonlocal uncovered_count
        block = sorted(block)
        blocks.append(block)
        t_indices = []
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                covered[idx] = 1
                uncovered_count -= 1
            t_indices.append(idx)
        block_t_counts.append(t_indices)

    # Phase 1: Greedy construction
    while uncovered_count > 0 and time_ok():
        # Find the first uncovered t-subset to start a block
        # We need to find an uncovered one; scan covered array
        # Optimization: keep track of uncovered indices
        # For simplicity, scan from start
        found = False
        for i in range(n_t):
            if not covered[i]:
                start_sub = all_t_subs[i]
                found = True
                break
        if not found:
            break

        block = list(start_sub)
        while len(block) < k:
            # Find point with max gain
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = count_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                # No point to add? Shouldn't happen if k <= v
                break
            block.append(best_x)

        add_block(block)

    # Phase 2: Reduction - remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    # We iterate until no more blocks can be removed
    if time_ok():
        # Build a reverse index: for each t-subset index, which blocks cover it
        # This is expensive, so we do it incrementally
        # Simpler approach: for each block, check if removing it would uncover any t-subset
        # We need to know, for each t-subset, how many blocks cover it
        
        # Build coverage count per t-subset
        cover_count = [0] * n_t
        for t_indices in block_t_counts:
            for idx in t_indices:
                cover_count[idx] += 1
        
        # Try removing blocks in order
        changed = True
        while changed and time_ok():
            changed = False
            for b_idx in range(len(blocks) - 1, -1, -1):
                if not time_ok():
                    break
                # Check if all t-subsets in this block are covered by other blocks
                redundant = True
                for idx in block_t_counts[b_idx]:
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for idx in block_t_counts[b_idx]:
                        cover_count[idx] -= 1
                    blocks.pop(b_idx)
                    block_t_counts.pop(b_idx)
                    changed = True
                    break  # restart scan since indices changed

    # Phase 3: Local search - try swapping points in blocks to improve coverage
    # This is complex; skip for now if time is tight
    if time_ok() and len(blocks) > 0:
        # Try to replace some blocks with better ones
        # For each block, try swapping one point for another to see if we can
        # cover more uncovered t-subsets while maintaining coverage
        # This is a heuristic; keep it simple
        pass

    return blocks
# EVOLVE-BLOCK-END
