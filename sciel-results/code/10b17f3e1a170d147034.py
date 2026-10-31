import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(all_points, k))
    
    # Precompute for each block:
    # 1. List of t-subset indices it covers
    # 2. Bitmask of t-subset indices it covers
    block_tsub_indices = {}
    block_masks = {}
    
    for b in all_blocks:
        indices = []
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            indices.append(idx)
            mask |= (1 << idx)
        block_tsub_indices[b] = indices
        block_masks[b] = mask

    # Greedy Initialization
    def greedy_init():
        uncovered_mask = (1 << n_tsubsets) - 1
        blocks = []
        block_set = set()
        
        while uncovered_mask != 0:
            best_block = None
            best_gain = -1
            
            for b in all_blocks:
                if b in block_set:
                    continue
                # Gain is number of bits in (uncovered_mask & block_masks[b])
                gain = (uncovered_mask & block_masks[b]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None or best_gain == 0:
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return blocks

    def simulated_annealing(init_blocks):
        current_blocks = list(init_blocks)
        current_block_set = set(current_blocks)
        
        # Initialize coverage counts and uncovered mask
        coverage = [0] * n_tsubsets
        uncovered_mask = (1 << n_tsubsets) - 1
        
        for b in current_blocks:
            for idx in block_tsub_indices[b]:
                coverage[idx] += 1
                if coverage[idx] == 1:
                    uncovered_mask &= ~(1 << idx)
        
        current_energy = uncovered_mask.bit_count()
        current_size = len(current_blocks)
        current_cost = current_energy * 10000 + current_size
        
        best_blocks = current_blocks[:]
        best_cost = current_cost
        
        T = 50.0
        T_min = 0.01
        alpha = 0.995
        max_iterations = 200000 # Can run more iterations due to speed
        
        current_time = time.time()
        
        for _ in range(max_iterations):
            if (time.time() - start_time) > time_limit:
                break
                
            # Choose move: Swap
            if len(current_blocks) == 0:
                continue
                
            idx = random.randint(0, len(current_blocks) - 1)
            b_old = current_blocks[idx]
            
            # Pick random b_new not in set
            # To avoid infinite loop if set is large, we can pick random from all_blocks
            # and retry if in set. Since all_blocks is ~50k and set is ~150, collision is low.
            b_new = random.choice(all_blocks)
            if b_new in current_block_set:
                continue
            
            # Calculate Delta Cost
            # 1. Simulate removal of b_old
            # 2. Simulate addition of b_new
            
            # We need to compute new_uncovered_mask efficiently.
            # Instead of updating global state and reverting, we can compute the delta
            # or just apply and revert.
            # Given the speed of bitmask ops, apply and revert is fine.
            
            # Apply Removal
            old_uncovered_mask = uncovered_mask
            for idx_ts in block_tsub_indices[b_old]:
                coverage[idx_ts] -= 1
                if coverage[idx_ts] == 0:
                    uncovered_mask |= (1 << idx_ts)
            
            # Apply Addition
            for idx_ts in block_tsub_indices[b_new]:
                coverage[idx_ts] += 1
                if coverage[idx_ts] == 1:
                    uncovered_mask &= ~(1 << idx_ts)
            
            new_energy = uncovered_mask.bit_count()
            new_size = current_size # Size stays same in a swap
            new_cost = new_energy * 10000 + new_size
            
            delta_cost = new_cost - current_cost
            
            # Acceptance
            accept = False
            if delta_cost < 0:
                accept = True
            elif delta_cost == 0:
                # If cost is same, we might accept to shuffle state, 
                # but strictly speaking we don't need to if we are minimizing.
                # However, for SA, we usually accept equal moves to explore.
                accept = True 
            else:
                accept = (random.random() < math.exp(-delta_cost / T))
            
            if accept:
                # Commit the move
                current_blocks[idx] = b_new
                current_block_set.remove(b_old)
                current_block_set.add(b_new)
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_blocks = current_blocks[:]
            else:
                # Revert
                # Reverse Addition
                for idx_ts in block_tsub_indices[b_new]:
                    coverage[idx_ts] -= 1
                    if coverage[idx_ts] == 0:
                        uncovered_mask |= (1 << idx_ts)
                # Reverse Removal
                for idx_ts in block_tsub_indices[b_old]:
                    coverage[idx_ts] += 1
                    if coverage[idx_ts] == 1:
                        uncovered_mask &= ~(1 << idx_ts)
                
                # State should be back to original
                # uncovered_mask should be old_uncovered_mask
                # coverage should be original
                # We don't strictly need to check, but it's good practice.
                # Note: The revert logic above exactly reverses the apply logic.
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Run
    init_blocks = greedy_init()
    result_blocks = simulated_annealing(init_blocks)
    
    # Final Check and Cleanup
    # Ensure validity
    # The SA maintains best_blocks only if it's valid? 
    # Wait, the cost function penalizes energy heavily. 
    # If best_cost has energy > 0, it's invalid. 
    # We should ensure we return a valid covering. 
    # If the SA ends with an invalid best, we should fall back to the last valid one or the init.
    
    # Let's verify best_blocks validity
    # Re-calculate energy for best_blocks to be sure
    coverage_check = [0] * n_tsubsets
    for b in best_blocks:
        for idx in block_tsub_indices[b]:
            coverage_check[idx] += 1
            
    is_valid = all(c > 0 for c in coverage_check)
    
    if not is_valid:
        # Fallback to init if somehow invalid (shouldn't happen if init is valid)
        result_blocks = init_blocks

    return [list(b) for b in result_blocks]
