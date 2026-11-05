import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute t-subsets and their indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    
    # Helper to get indices of t-subsets within a k-block
    def get_block_indices(block):
        return [subset_to_idx[subset] for subset in itertools.combinations(block, t)]

    best_blocks = None
    best_size = float('inf')
    
    # Function to run one full attempt (Greedy + Local Search)
    def run_attempt(seed):
        random.seed(seed)
        
        # 1. Greedy Construction
        cover_count = [0] * num_t_subsets
        uncovered_indices = set(range(num_t_subsets))
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered t-subset to anchor the new block
            # This helps in diversity and speed
            anchor_subset = random.choice(list(uncovered_indices))
            anchor_tuple = t_subsets[anchor_subset]
            
            # Generate candidate k-blocks containing anchor_tuple
            # Points not in anchor
            remaining_points = [p for p in range(v) if p not in anchor_tuple]
            # Need to choose k-t more points from remaining
            needed = k - t
            
            best_gain = -1
            best_block = None
            
            # Iterate through combinations of remaining points
            for combo in itertools.combinations(remaining_points, needed):
                candidate_block = tuple(sorted(anchor_tuple + combo))
                
                # Calculate gain
                gain = 0
                for idx in get_block_indices(candidate_block):
                    if idx in uncovered_indices:
                        gain += 1
                        if gain > best_gain:
                            # Early exit if we found a perfect block? No, just track max
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            # Add best block
            if best_block is None:
                break # Should not happen if v is large enough
                
            blocks.append(best_block)
            
            # Update cover counts and uncovered set
            for idx in get_block_indices(best_block):
                if cover_count[idx] == 0:
                    uncovered_indices.remove(idx)
                cover_count[idx] += 1
        
        # 2. Local Search: Greedy Removal
        # Re-calculate cover_count from scratch to be safe and clean
        cover_count = [0] * num_t_subsets
        for b in blocks:
            for idx in get_block_indices(b):
                cover_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            # Shuffle blocks to avoid bias? Or just iterate.
            # Iterating and removing requires care with indices.
            # We iterate from 0 to len-1. If we remove, we break and restart the while loop.
            for i in range(len(blocks)):
                b = blocks[i]
                b_indices = get_block_indices(b)
                
                can_remove = True
                for idx in b_indices:
                    if cover_count[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for idx in b_indices:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart the for loop with new len
            
        return blocks

    # Run multiple attempts
    attempt_id = 0
    while time.time() < deadline:
        blocks = run_attempt(attempt_id)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we hit the theoretical lower bound or a very good number, we might stop early?
            # Lower bound is 69. If we get 76, it's the known best.
            # If we get < 76, it's a breakthrough!
            if best_size <= 76:
                # We can continue to try to find even better, but 76 is the record.
                # Actually, 76 is the *best known*. If we find 75, it's huge.
                pass
        
        attempt_id += 1
        
        # If we have been running for a while and haven't improved in a long time, 
        # we can increase the complexity or just keep going.
        # Given the speed of run_attempt, we can do many.
        
    if best_blocks is None:
        # Fallback: return a trivial valid covering (though likely huge)
        # Or just return the last attempt
        return run_attempt(0)
        
    return [list(b) for b in best_blocks]
