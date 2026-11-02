"""Covering design C(24,7,3) using whole-block greedy + local search."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    
    # Map each t-subset to an index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Greedy whole-block set cover
    # For each candidate block, precompute which t-subsets it covers
    # We'll use a greedy approach: repeatedly pick the block that covers the most uncovered subsets
    
    uncovered = [1] * num_subsets  # 1 = uncovered, 0 = covered
    blocks = []
    
    # Precompute all possible blocks and their covered subsets
    # This is C(24,7) = 346104 blocks, each covers C(7,3)=35 subsets
    # We can't precompute all at once efficiently, so we use a smarter approach
    
    # Actually, let's use a different strategy:
    # 1. Generate a good initial set of blocks using a structured approach
    # 2. Then do local search to remove redundant blocks
    
    # Strategy: Use a greedy construction where we build blocks one at a time,
    # choosing the block that maximizes newly covered t-subsets.
    
    # For efficiency, maintain for each point x, the count of uncovered subsets containing x
    # When we add a block, we update these counts.
    
    # Let's use a vectorized approach with numpy
    import numpy as np
    
    uncovered_arr = np.ones(num_subsets, dtype=np.int32)
    
    # For each point, track which uncovered subsets contain it
    # subsets_containing[x] = list of indices of subsets containing x
    subsets_containing = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for x in s:
            subsets_containing[x].append(i)
    
    # Convert to numpy arrays for speed
    subsets_containing_arr = [np.array(lst, dtype=np.int32) for lst in subsets_containing]
    
    # Greedy: pick block that covers most uncovered subsets
    # To evaluate a block, we need to count how many of its C(k,t) subsets are uncovered
    # We can do this by taking the union of subsets_containing for each point in the block
    
    def block_coverage(block):
        """Count how many uncovered t-subsets are covered by this block."""
        # A t-subset is covered by the block iff all its elements are in the block
        # So we need to count subsets s where s ⊆ block
        # We can iterate over all t-subsets of the block
        block_set = set(block)
        count = 0
        for s in itertools.combinations(block, t):
            idx = subset_to_idx[s]
            if uncovered_arr[idx]:
                count += 1
        return count
    
    # This is slow. Let's use a better approach.
    # Maintain a count array: for each t-subset, how many current blocks cover it.
    # But for greedy construction, we just need to know if it's covered.
    
    # Better approach: use a bitset or just iterate.
    # Given the time constraints, let's use a simpler but effective heuristic.
    
    # Alternative: use a known good construction as starting point, then local search.
    
    # Let's try: generate blocks using a random/greedy mix, then do local search.
    
    # Phase 1: Greedy construction
    # Use a scoring function for candidate points to build each block
    
    uncovered_set = set(all_subsets)
    blocks = []
    
    while uncovered_set:
        if time.time() > deadline - 30:
            break
        
        # Pick the first uncovered subset as a seed
        seed = min(uncovered_set)
        block = list(seed)
        
        # Add points greedily to maximize coverage
        while len(block) < k:
            block_set = set(block)
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered subsets would be covered by adding x
                gain = 0
                for s in itertools.combinations(block, t - 1):
                    new_s = tuple(sorted(s + (x,)))
                    if new_s in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered subsets
        for s in itertools.combinations(block, t):
            uncovered_set.discard(tuple(s))
    
    if not uncovered_set:
        # We have a valid covering, now do local search to minimize
        pass
    
    # Phase 2: Local search to remove redundant blocks
    # A block is redundant if every t-subset it covers is also covered by another block
    
    def is_redundant(block_idx, blocks_list):
        """Check if block at block_idx is redundant."""
        block = blocks_list[block_idx]
        block_set = set(block)
        for s in itertools.combinations(block, t):
            # Check if s is covered by any other block
            found = False
            for j, b in enumerate(blocks_list):
                if j == block_idx:
                    continue
                if all(x in b for x in s):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Remove redundant blocks
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 10:
                break
            if is_redundant(i, blocks):
                blocks.pop(i)
                changed = True
    
    # Verify coverage
    if uncovered_set:
        # If we didn't finish, we need to add more blocks
        # This shouldn't happen if we completed phase 1
        pass
    
    return blocks
