"""Probabilistic covering design with simulated annealing for C(19,8,5)."""
import random
import itertools
import time
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a hybrid approach:
    1. Greedy initialization to get a reasonable starting point
    2. Simulated annealing to improve the covering by minimizing uncovered t-subsets
    """
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Greedy initialization
    def greedy_cover():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick the t-subset with the smallest index (arbitrary but deterministic)
            min_idx = min(uncovered)
            block = list(all_t_subsets[min_idx])
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        if ts in t_subset_index and t_subset_index[ts] in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Remove covered t-subsets
            for ts in itertools.combinations(block, t):
                idx = t_subset_index.get(ts)
                if idx is not None and idx in uncovered:
                    uncovered.discard(idx)
        
        return list(blocks)
    
    # Initialize with greedy
    blocks = greedy_cover()
    
    # Build coverage count for each t-subset
    def build_coverage(blocks):
        cov = [0] * num_t_subsets
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_index.get(ts)
                if idx is not None:
                    cov[idx] += 1
        return cov
    
    def count_uncovered(cov):
        return sum(1 for c in cov if c == 0)
    
    # Simulated annealing
    cov = build_coverage(blocks)
    current_uncovered = count_uncovered(cov)
    best_blocks = list(blocks)
    best_uncovered = current_uncovered
    
    temperature = 100.0
    cooling = 0.995
    min_temp = 0.1
    
    # Precompute for each block, which t-subsets it covers
    def get_block_tsubsets(block):
        return [t_subset_index[ts] for ts in itertools.combinations(block, t)]
    
    # For each block, store its t-subset indices
    block_tsubs = [get_block_tsubsets(b) for b in blocks]
    
    while temperature > min_temp:
        if time.time() - start_time > timeout:
            break
        
        # Choose a random block to modify
        block_idx = random.randint(0, len(blocks) - 1)
        old_block = blocks[block_idx]
        old_tsubs = block_tsubs[block_idx]
        
        # Remove old block's coverage
        for idx in old_tsubs:
            cov[idx] -= 1
        
        # Try to create a new block by swapping points
        # Strategy: take the old block, remove one random point, add a different one
        old_block_list = list(old_block)
        remove_pos = random.randint(0, len(old_block_list) - 1)
        removed_point = old_block_list[remove_pos]
        old_block_list.remove(removed_point)
        
        # Find a point to add that maximizes coverage gain
        best_new_point = None
        best_gain = -1
        
        # Candidate points: those not in the current block
        candidates = [x for x in range(v) if x not in old_block_list]
        random.shuffle(candidates)
        
        # Limit candidates for speed
        for x in candidates[:20]:
            new_block_list = old_block_list + [x]
            new_block_list.sort()
            
            # Compute gain: how many newly covered t-subsets
            gain = 0
            for ts in itertools.combinations(new_block_list, t):
                idx = t_subset_index.get(ts)
                if idx is not None and cov[idx] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_new_point = x
        
        if best_new_point is None:
            # Restore old block
            for idx in old_tsubs:
                cov[idx] += 1
            continue
        
        new_block_list = old_block_list + [best_new_point]
        new_block_list.sort()
        new_block = tuple(new_block_list)
        
        # Check if this block already exists (to avoid duplicates)
        if new_block in blocks and blocks.index(new_block) != block_idx:
            # Duplicate block, skip
            for idx in old_tsubs:
                cov[idx] += 1
            continue
        
        # Add new block's coverage
        new_tsubs = get_block_tsubsets(new_block)
        for idx in new_tsubs:
            cov[idx] += 1
        
        new_uncovered = count_uncovered(cov)
        delta = new_uncovered - current_uncovered
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < pow(2.718281828, -delta / temperature):
            # Accept the move
            blocks[block_idx] = new_block
            block_tsubs[block_idx] = new_tsubs
            current_uncovered = new_uncovered
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_blocks = list(blocks)
        else:
            # Reject: restore old block
            for idx in new_tsubs:
                cov[idx] -= 1
            for idx in old_tsubs:
                cov[idx] += 1
        
        temperature *= cooling
    
    # Return the best solution found
    return [list(b) for b in best_blocks]
