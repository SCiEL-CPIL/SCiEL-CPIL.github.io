"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using multi-start greedy with reduction.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_cover(uncovered_set, seed=None):
        """Run one greedy pass starting from uncovered_set. Returns blocks list."""
        if seed is not None:
            random.seed(seed)
        uncovered = set(uncovered_set)
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to start a new block
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            while len(block) < k:
                # Find point that maximizes coverage of uncovered t-subsets
                best_gain = -1
                best_point = None
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                block.append(best_point)
            block.sort()
            blocks.append(block)
            # Remove all t-subsets covered by this block
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        return blocks
    
    def reduce_blocks(blocks):
        """Try to remove redundant blocks from the covering."""
        if not blocks:
            return blocks
        # For each block, check if removing it still covers all t-subsets
        # Build coverage map: for each t-subset, which blocks cover it
        # This is expensive, so we do a simple greedy removal
        reduced = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(reduced)):
                # Check if block i is redundant
                test_blocks = reduced[:i] + reduced[i+1:]
                # Check if all t-subsets are still covered
                covered = set()
                for b in test_blocks:
                    covered.update(itertools.combinations(b, t))
                if len(covered) == N_t:
                    reduced.pop(i)
                    changed = True
                    break
        return reduced
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        seed = restart_count * 12345
        blocks = greedy_cover(all_t_subsets, seed=seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Try reduction
        if time.time() < deadline:
            reduced = reduce_blocks(best_blocks)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
