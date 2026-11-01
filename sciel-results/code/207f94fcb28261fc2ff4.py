import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_bitsets(v, k, t):
    """
    Precompute all t-sets and map each block to a bitset of the t-sets it covers.
    Also maps each t-set to the list of blocks that cover it.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_masks = []
    for block in all_blocks:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # tset_to_blocks: for each t-set index, which blocks cover it?
    # This is useful for finding candidate replacements.
    tset_to_blocks = defaultdict(list)
    for b_idx, mask in enumerate(block_masks):
        # Iterate through set bits? Or just iterate through all tsets?
        # 2024 tsets is small.
        for ts_idx in range(n_tsets):
            if mask & (1 << ts_idx):
                tset_to_blocks[ts_idx].append(b_idx)
                
    return all_tsets, n_tsets, all_blocks, n_blocks, block_masks, tset_to_blocks

def _greedy_construct(v, k, t, all_blocks, block_masks, tset_to_blocks, n_tsets, seed=0):
    """
    Construct a valid covering using a randomized greedy algorithm.
    """
    rng = random.Random(seed)
    full_mask = (1 << n_tsets) - 1
    uncovered_mask = full_mask
    active_blocks = set()
    
    while uncovered_mask:
        best_block = -1
        best_gain = 0
        
        # Pick a random uncovered t-set to anchor the search
        # This makes the greedy stochastic
        found = False
        while not found:
            rand_ts_idx = rng.randint(0, n_tsets - 1)
            if uncovered_mask & (1 << rand_ts_idx):
                found = True
                break
        
        candidates = tset_to_blocks[rand_ts_idx]
        
        # Evaluate candidates
        for b_idx in candidates:
            if b_idx in active_blocks:
                continue
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
            elif gain == best_gain and best_block != -1:
                # Tie-breaking: random choice among best gainers to diversify
                if rng.random() < 0.5:
                    best_block = b_idx
                    
        if best_block == -1:
            # Fallback: scan all blocks (should rarely happen if tset_to_blocks is complete)
            for b_idx in range(len(all_blocks)):
                if b_idx in active_blocks:
                    continue
                gain = (block_masks[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
        if best_block == -1:
            break
            
        active_blocks.add(best_block)
        uncovered_mask &= ~block_masks[best_block]
        
    return active_blocks

def _local_search(state, block_masks, full_mask, n_tsets, time_limit=5.0, seed=0):
    """
    Perform local search on a valid state to minimize block count.
    Uses 1-out, 1-in, and 2-out, 2-in moves.
    """
    rng = random.Random(seed)
    start_time = time.time()
    current_state = set(state)
    covered_mask = 0
    for b in current_state:
        covered_mask |= block_masks[b]
    
    current_cost = len(current_state)
    best_state = set(current_state)
    best_cost = current_cost
    
    # Precompute list of active blocks for faster random choice
    # We will maintain a list parallel to the set for O(1) random access
    
    def get_active_list():
        return list(current_state)
        
    no_improve_count = 0
    max_no_improve = 1000 # Reheat threshold
    
    while time.time() - start_time < time_limit:
        active_list = get_active_list()
        if not active_list:
            break
            
        # Choose move type
        # 1. Remove (1-out)
        # 2. Swap (1-out, 1-in)
        # 3. Double Swap (2-out, 2-in) - more expensive but escapes deeper minima
        
        r = rng.random()
        
        if r < 0.4:
            # 1-out: Try to remove a block
            b_remove = rng.choice(active_list)
            new_mask = covered_mask & ~block_masks[b_remove]
            if new_mask == full_mask:
                # Valid removal
                current_state.discard(b_remove)
                covered_mask = new_mask
                current_cost -= 1
                if current_cost < best_cost:
                    best_state = set(current_state)
                    best_cost = current_cost
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif r < 0.8:
            # 1-in, 1-out: Swap
            b_old = rng.choice(active_list)
            b_new = rng.randint(0, len(block_masks) - 1)
            if b_new in current_state:
                continue
                
            temp_mask = covered_mask & ~block_masks[b_old]
            new_mask = temp_mask | block_masks[b_new]
            
            if new_mask == full_mask:
                # Valid swap
                current_state.discard(b_old)
                current_state.add(b_new)
                covered_mask = new_mask
                # Cost is same, but state changed. 
                # We don't update best_cost here as it's not an improvement in count,
                # but it might lead to future removals.
                no_improve_count += 1
                
        else:
            # 2-out, 2-in: Double Swap
            # Pick 2 distinct blocks to remove
            if len(current_state) < 2:
                continue
                
            b_old1, b_old2 = rng.sample(active_list, 2)
            
            # Pick 2 distinct new blocks
            # To save time, we might just pick random ones and check
            b_new1 = rng.randint(0, len(block_masks) - 1)
            b_new2 = rng.randint(0, len(block_masks) - 1)
            
            if b_new1 in current_state or b_new2 in current_state or b_new1 == b_new2:
                continue
                
            # Calculate new mask
            temp_mask = covered_mask & ~block_masks[b_old1] & ~block_masks[b_old2]
            new_mask = temp_mask | block_masks[b_new1] | block_masks[b_new2]
            
            if new_mask == full_mask:
                current_state.discard(b_old1)
                current_state.discard(b_old2)
                current_state.add(b_new1)
                current_state.add(b_new2)
                covered_mask = new_mask
                no_improve_count += 1
                
        # Reheat logic: if stuck, add a random block to escape
        if no_improve_count > max_no_improve:
            # Add a random block
            b_add = rng.randint(0, len(block_masks) - 1)
            if b_add not in current_state:
                current_state.add(b_add)
                covered_mask |= block_masks[b_add]
                current_cost += 1
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 0.9) # Decay threshold
            
    return best_state, best_cost

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Precomputation
    all_tsets, n_tsets, all_blocks, n_blocks, block_masks, tset_to_blocks = _precompute_bitsets(v, k, t)
    
    full_mask = (1 << n_tsets) - 1
    
    best_overall_state = None
    best_overall_cost = float('inf')
    
    # Portfolio approach: Multiple restarts
    # Estimate time per restart
    # Greedy is fast. Local search takes ~5s.
    # We can afford ~20-25 restarts in 120s.
    
    num_restarts = 25
    time_per_restart = (deadline - start_time) / num_restarts - 1.0 # Reserve 1s per restart for overhead
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Greedy Construction
        seed = i
        initial_state = _greedy_construct(v, k, t, all_blocks, block_masks, tset_to_blocks, n_tsets, seed=seed)
        
        # Verify validity (should be valid by construction, but just in case)
        # If invalid, skip to next restart? Or repair?
        # Greedy ensures validity.
        
        # 2. Local Search
        ls_state, ls_cost = _local_search(initial_state, block_masks, full_mask, n_tsets, time_limit=time_per_restart, seed=seed)
        
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_state = set(ls_state)
            
    # If we didn't find anything (shouldn't happen), fallback to one greedy
    if best_overall_state is None:
        best_overall_state = _greedy_construct(v, k, t, all_blocks, block_masks, tset_to_blocks, n_tsets, seed=999)
        best_overall_cost = len(best_overall_state)
        
    # Final cleanup: try to remove any redundant blocks from best_overall_state
    final_state = set(best_overall_state)
    best_covered_mask = 0
    for b in final_state:
        best_covered_mask |= block_masks[b]
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for b in list(final_state):
            if best_covered_mask & ~block_masks[b] == full_mask:
                # b is redundant
                final_state.discard(b)
                best_covered_mask &= ~block_masks[b]
                improved = True
                break # Restart loop after modification
            
    # Verify final result
    result_blocks = [list(all_blocks[idx]) for idx in final_state]
    
    # Double check validity
    final_check_mask = 0
    for b in final_state:
        final_check_mask |= block_masks[b]
        
    if final_check_mask != full_mask:
        # Fallback to initial greedy if something went wrong
        fallback_state = _greedy_construct(v, k, t, all_blocks, block_masks, tset_to_blocks, n_tsets, seed=123)
        result_blocks = [list(all_blocks[idx]) for idx in fallback_state]
        
    return result_blocks

# EVOLVE-BLOCK-END
