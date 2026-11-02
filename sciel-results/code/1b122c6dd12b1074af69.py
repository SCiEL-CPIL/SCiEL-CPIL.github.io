"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, keep safety margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Track which blocks cover which t-subsets
    # covered[i] = number of blocks that cover t-subset i
    covered = np.zeros(num_t_subs, dtype=np.int32)
    
    blocks = []
    block_t_subs = []  # For each block, the list of t-subset indices it covers
    
    def time_check():
        return time.time() < deadline - 2.0
    
    def add_block(block):
        """Add a block and update coverage counts."""
        blocks.append(tuple(sorted(block)))
        t_indices = []
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            covered[idx] += 1
            t_indices.append(idx)
        block_t_subs.append(t_indices)
    
    def remove_block(block_idx):
        """Remove a block and update coverage counts."""
        t_indices = block_t_subs[block_idx]
        for idx in t_indices:
            covered[idx] -= 1
        blocks.pop(block_idx)
        block_t_subs.pop(block_idx)
    
    def is_covered():
        return np.all(covered >= 1)
    
    def greedy_pick_start(uncovered_mask):
        """Pick a random uncovered t-subset to start a new block."""
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return None
        return all_t_subs[uncovered_indices[random.randint(0, len(uncovered_indices)-1)]]
    
    def greedy_build_block(start_subset, existing_blocks_set):
        """Build a block starting from start_subset using greedy heuristic."""
        block = list(start_subset)
        block_set = set(block)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets this point would help cover
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in t_sub_to_idx:
                        idx = t_sub_to_idx[candidate]
                        if covered[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is not None:
                block.append(best_point)
                block_set.add(best_point)
        
        block.sort()
        return tuple(block)
    
    def local_search_2_exchange():
        """Perform 2-exchanges: try swapping one block for another to improve coverage."""
        if not time_check():
            return False
        
        improved = False
        num_blocks = len(blocks)
        
        for i in range(num_blocks):
            if not time_check():
                break
            # Try removing block i and see if we can replace it with a better one
            # First, check if removing block i would leave any t-subset uncovered
            t_indices_i = block_t_subs[i]
            would_uncover = False
            for idx in t_indices_i:
                if covered[idx] == 1:
                    would_uncover = True
                    break
            
            if not would_uncover:
                # Block i is redundant, remove it
                remove_block(i)
                improved = True
                # Re-index
                continue
            
            # Try to find a replacement block that covers the same critical t-subsets
            # with less redundancy
            critical_t_subs = [idx for idx in t_indices_i if covered[idx] == 1]
            
            if not critical_t_subs:
                continue
            
            # Try greedy replacement
            best_new_block = None
            best_score = -1
            
            for attempt in range(3):
                # Pick a critical t-subset as seed
                seed_idx = critical_t_subs[random.randint(0, len(critical_t_subs)-1)]
                seed_subset = all_t_subs[seed_idx]
                
                new_block = greedy_build_block(seed_subset, set(blocks))
                
                if new_block in blocks:
                    continue
                
                # Evaluate: how many of block i's critical t-subsets does this cover?
                new_t_indices = []
                for ts in itertools.combinations(new_block, t):
                    idx = t_sub_to_idx[ts]
                    new_t_indices.append(idx)
                
                new_set = set(new_t_indices)
                covered_critical = sum(1 for idx in critical_t_subs if idx in new_set)
                
                # Also count how many additional t-subsets it covers (bonus)
                additional = sum(1 for idx in new_t_indices if covered[idx] == 0)
                
                score = covered_critical * 100 + additional
                
                if score > best_score:
                    best_score = score
                    best_new_block = new_block
            
            if best_new_block is not None and best_score >= len(critical_t_subs) * 100:
                # Replace block i with best_new_block
                remove_block(i)
                add_block(best_new_block)
                improved = True
        
        return improved
    
    # Phase 1: Greedy construction with randomization
    random.seed(42)
    
    while not is_covered() and time_check():
        uncovered_mask = (covered == 0)
        if not uncovered_mask.any():
            break
        
        start_subset = greedy_pick_start(uncovered_mask)
        if start_subset is None:
            break
        
        block = greedy_build_block(start_subset, set(blocks))
        
        # Check if this block is already in the set
        if block in blocks:
            continue
        
        add_block(block)
    
    # Phase 2: Local search with 2-exchanges
    if time_check():
        max_iterations = 50
        for iteration in range(max_iterations):
            if not time_check():
                break
            improved = local_search_2_exchange()
            if not improved:
                break
    
    # Phase 3: Final cleanup - remove any redundant blocks
    if time_check():
        i = 0
        while i < len(blocks):
            if not time_check():
                break
            t_indices_i = block_t_subs[i]
            redundant = True
            for idx in t_indices_i:
                if covered[idx] == 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
            else:
                i += 1
    
    # Return as list of lists
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
