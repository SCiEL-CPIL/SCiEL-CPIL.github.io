"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses bitmask-based greedy with multiple random restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    # Precompute all t-subsets as bitmasks
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_index = {}
    for i, combo in enumerate(all_t_subsets):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        t_subset_index[mask] = i
    
    # Precompute for each block (k-subset), which t-subsets it covers
    # We'll compute this on the fly during greedy
    
    def greedy_cover(seed):
        """Run a greedy covering algorithm with a given random seed."""
        random.seed(seed)
        
        # uncovered: set of indices of uncovered t-subsets
        uncovered = set(range(n_t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_idx = random.choice(list(uncovered))
            start_combo = all_t_subsets[start_idx]
            block = list(start_combo)
            random.shuffle(block)
            
            while len(block) < k:
                # For each candidate point, compute how many uncovered t-subsets it would cover
                best_gain = -1
                best_point = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets contain x and the other t-1 points from block
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_tuple = tuple(sorted(rest + (x,)))
                        mask = 0
                        for y in new_tuple:
                            mask |= (1 << y)
                        if mask in t_subset_index:
                            idx = t_subset_index[mask]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            block_set = set(block)
            for combo in itertools.combinations(block, t):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                idx = t_subset_index[mask]
                uncovered.discard(idx)
        
        return blocks
    
    # Try multiple random restarts
    best_blocks = None
    best_len = float('inf')
    
    # First, try a deterministic greedy (seed=0)
    blocks = greedy_cover(0)
    best_blocks = blocks
    best_len = len(blocks)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_blocks
    
    # Try more random restarts
    for seed in range(1, 200):
        if time.time() > deadline - 5:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Local search: try to remove redundant blocks
    def try_remove_blocks(blocks, deadline):
        """Try to remove blocks that are redundant."""
        n = len(blocks)
        # For each t-subset, track which blocks cover it
        # We'll use a simpler approach: for each block, check if all its t-subsets are covered by other blocks
        
        # Precompute t-subsets for each block
        block_t_subsets = []
        for block in blocks:
            subs = []
            for combo in itertools.combinations(block, t):
                mask = 0
                for x in combo:
                    mask |= (1 << x)
                subs.append(mask)
            block_t_subsets.append(subs)
        
        # Count coverage for each t-subset
        coverage_count = {}
        for i, subs in enumerate(block_t_subsets):
            for mask in subs:
                coverage_count[mask] = coverage_count.get(mask, 0) + 1
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 2:
                break
            for i in range(len(blocks)):
                if time.time() > deadline - 2:
                    break
                # Check if block i is redundant
                redundant = True
                for mask in block_t_subsets[i]:
                    if coverage_count.get(mask, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for mask in block_t_subsets[i]:
                        coverage_count[mask] -= 1
                    blocks.pop(i)
                    block_t_subsets.pop(i)
                    changed = True
                    break
        
        return blocks
    
    best_blocks = try_remove_blocks(best_blocks, deadline)
    
    return best_blocks

# EVOLVE-BLOCK-END
