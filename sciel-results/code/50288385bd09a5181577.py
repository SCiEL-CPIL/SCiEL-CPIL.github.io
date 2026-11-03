# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets it's in (as a set of indices)
    # Actually, for a block B, the t-subsets covered are all t-subsets of B
    # Precompute for each possible (k-1)-subset the list of t-subsets that extend it
    # Better: for each point x and each (t-1)-subset S of the current block,
    # the t-subsets covered by adding x are {S ∪ {x}} for S in C(block, t-1)
    
    # Represent coverage as a numpy boolean array
    covered = np.zeros(n_t, dtype=bool)
    blocks = []
    
    # Precompute: for each point x, for each (t-1) tuple, the t-subset index
    # Actually let's precompute for each (t-1)-subset and each possible x not in it,
    # the index of the t-subset
    t_minus_1_combos = list(itertools.combinations(range(v), t - 1))
    # Map (t-1)-subset to list of (x, t_index) for x not in subset
    # This is expensive to precompute fully, so we'll compute on the fly but cache
    
    # Precompute: for each t-subset index, the set of points in it (as a frozenset or bitmask)
    # Use bitmask for points
    t_subset_points = []
    for s in all_t_subsets:
        mask = 0
        for p in s:
            mask |= (1 << p)
        t_subset_points.append(mask)
    t_subset_points = np.array(t_subset_points, dtype=np.int64)
    
    def block_covers_t_subsets(block):
        """Return boolean array of t-subsets covered by this block."""
        block_set = set(block)
        # All t-subsets of block
        result = np.zeros(n_t, dtype=bool)
        for ts in itertools.combinations(block, t):
            result[t_index[ts]] = True
        return result
    
    def greedy_block(current_block, covered_arr):
        """Extend current_block to size k by adding point with max new coverage."""
        block = list(current_block)
        while len(block) < k:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            # For each candidate x, count how many uncovered t-subsets of block∪{x} we'd cover
            # t-subsets of block∪{x} that include x: for each (t-1)-subset of block, form t-subset
            # t-subsets of block∪{x} that don't include x: already covered by block (if |block| >= t)
            # So new coverage from x = t-subsets of the form S∪{x} where S is (t-1)-subset of block
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(S + (x,)))
                    idx = t_index[ts]
                    if not covered_arr[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        return sorted(block)
    
    # Phase 1: Greedy construction
    while not covered.all():
        if time.time() > deadline - 30:
            break
        # Find first uncovered t-subset
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break
        first_uncovered = all_t_subsets[uncovered_indices[0]]
        
        # Build block greedily starting with this t-subset
        block = greedy_block(list(first_uncovered), covered)
        blocks.append(block)
        
        # Update coverage
        new_cover = block_covers_t_subsets(block)
        covered |= new_cover
    
    # If we ran out of time during greedy, we might not have full coverage
    # Check if covered
    if not covered.all():
        # Continue greedy with remaining time
        while not covered.all() and time.time() < deadline - 10:
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            first_uncovered = all_t_subsets[uncovered_indices[0]]
            block = greedy_block(list(first_uncovered), covered)
            blocks.append(block)
            new_cover = block_covers_t_subsets(block)
            covered |= new_cover
    
    n_blocks = len(blocks)
    
    # Phase 2: Local search - try to remove or replace blocks
    # Recompute coverage from scratch to verify
    covered_check = np.zeros(n_t, dtype=bool)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered_check[t_index[ts]] = True
    
    if not covered_check.all():
        # Fallback: add more blocks
        while not covered_check.all() and time.time() < deadline - 5:
            uncovered_indices = np.where(~covered_check)[0]
            first_uncovered = all_t_subsets[uncovered_indices[0]]
            block = greedy_block(list(first_uncovered), covered_check)
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                covered_check[t_index[ts]] = True
    
    n_blocks = len(blocks)
    
    # Phase 3: Try to remove redundant blocks
    # For each block, check if all its t-subsets are covered by other blocks
    if time.time() < deadline - 20:
        # Build for each t-subset, which blocks cover it
        # t_subset -> list of block indices
        t_to_blocks = [[] for _ in range(n_t)]
        for bi, block in enumerate(blocks):
            for ts in itertools.combinations(block, t):
                t_to_blocks[t_index[ts]].append(bi)
        
        # Try removing blocks in random order
        random.seed(42)
        remove_order = list(range(len(blocks)))
        random.shuffle(remove_order)
        
        new_blocks = list(blocks)
        for bi in remove_order:
            if time.time() > deadline - 10:
                break
            if bi >= len(new_blocks):
                continue
            block = new_blocks[bi]
            # Check if all t-subsets of this block are covered by other blocks
            can_remove = True
            for ts in itertools.combinations(block, t):
                idx = t_index[ts]
                # Count how many blocks cover this t-subset (excluding bi)
                count = 0
                for other_bi in t_to_blocks[idx]:
                    if other_bi != bi and other_bi < len(new_blocks):
                        count += 1
                if count == 0:
                    can_remove = False
                    break
            if can_remove:
                # Remove block bi
                for idx in range(n_t):
                    if bi in t_to_blocks[idx]:
                        t_to_blocks[idx].remove(bi)
                new_blocks.pop(bi)
        
        blocks = new_blocks
        n_blocks = len(blocks)
    
    # Verify final coverage
    final_covered = np.zeros(n_t, dtype=bool)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            final_covered[t_index[ts]] = True
    
    # If not fully covered, add more blocks
    if not final_covered.all() and time.time() < deadline - 2:
        while not final_covered.all() and time.time() < deadline - 1:
            uncovered_indices = np.where(~final_covered)[0]
            first_uncovered = all_t_subsets[uncovered_indices[0]]
            block = greedy_block(list(first_uncovered), final_covered)
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                final_covered[t_index[ts]] = True
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for block in blocks:
        key = tuple(block)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(block)
    
    return unique_blocks
# EVOLVE-BLOCK-END
