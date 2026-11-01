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
    
    # For each block, store a frozenset of tset indices it covers
    # Using frozenset allows for fast intersection/union if we were to use sets,
    # but for SA we need fast "how many new t-sets does this block cover given current state"
    # and "which t-sets become uncovered if we remove this block".
    
    # Let's store as lists of indices for iteration, and maybe a boolean mask if n_tsets is small enough.
    # n_tsets = C(24,3) = 2024. This is small enough to use a boolean array or even a Python int as a bitset?
    # 2024 bits is ~253 bytes. Python int can handle this easily.
    
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

def _compute_uncovered_mask(state, n_tsets, block_masks):
    """
    Compute the bitmask of uncovered t-sets.
    state: set of active block indices.
    Returns: int where bit i is 1 if t-set i is uncovered.
    """
    covered_mask = 0
    for b_idx in state:
        covered_mask |= block_masks[b_idx]
    full_mask = (1 << n_tsets) - 1
    return full_mask ^ covered_mask

def _count_bits(n):
    """Count set bits in an integer."""
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Precomputation
    all_tsets, n_tsets, all_blocks, n_blocks, block_masks, tset_to_blocks = _precompute_bitsets(v, k, t)
    
    # 2. Initial Solution via Greedy
    # We want a good starting point. Standard greedy: pick block covering most uncovered t-sets.
    uncovered_mask = (1 << n_tsets) - 1
    active_blocks = set()
    
    while uncovered_mask and time.time() < deadline - 60.0:
        best_block = -1
        best_gain = 0
        
        # Optimization: We can iterate through all blocks, but that's 346k blocks.
        # It's faster to pick a random uncovered t-set, then look at blocks covering it.
        # Let's pick a random uncovered t-set index.
        # Find an index with a set bit in uncovered_mask.
        # Since n_tsets is 2024, we can just find the first one or use random.
        # To be fast, let's just iterate through a few random uncovered t-sets to find a good candidate.
        
        # Actually, for the initial greedy, iterating all blocks might be too slow if done naively.
        # C(24,7) = 346,104 blocks.
        # For each block, gain = popcount(block_mask & uncovered_mask).
        # 346k * (popcount op) is fast in Python if using int.bit_count() (Python 3.10+).
        # Let's assume Python 3.10+ is available. If not, _count_bits is fine but slower.
        
        # Let's use a hybrid: sample some blocks, or just do full scan if fast enough.
        # Full scan of 346k blocks with bit_count is ~0.1s in C, but Python loop overhead is high.
        # Let's try to optimize the greedy step.
        
        # Alternative Greedy:
        # Pick a random uncovered t-set.
        # Get list of blocks covering it.
        # Pick the one with max gain.
        
        # Find a set bit in uncovered_mask
        # Trick: (uncovered_mask & -uncovered_mask) gives the lowest set bit.
        # But we want random.
        # Let's just pick a random index from 0 to n_tsets-1 and check if it's uncovered.
        # If not, try again.
        
        rand_ts_idx = random.randint(0, n_tsets - 1)
        while not (uncovered_mask & (1 << rand_ts_idx)):
            rand_ts_idx = random.randint(0, n_tsets - 1)
            
        candidates = tset_to_blocks[rand_ts_idx]
        
        for b_idx in candidates:
            if b_idx in active_blocks:
                continue
            gain = (block_masks[b_idx] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                
        if best_block == -1:
            # Fallback: scan all blocks (should rarely happen)
            for b_idx in range(n_blocks):
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
        
    initial_solution = list(active_blocks)
    
    # 3. Simulated Annealing
    # State: set of active block indices.
    # Cost: number of blocks (we want to minimize).
    # But we must maintain validity: all t-sets covered.
    # If we remove a block and it causes uncovered t-sets, the state becomes invalid.
    # Strategy: 
    #   - If we remove a block and it's redundant (no t-set becomes uncovered), accept.
    #   - If we swap a block with another, and the new set is valid, accept if better or with prob.
    #   - We need to be able to escape local minima. 
    #   - Common approach for covering designs: 
    #     1. Try to remove blocks (reduction).
    #     2. Try to swap blocks (optimization).
    #     3. If stuck, add a random block (expansion) then try to reduce again? 
    #     Or use a "penalty" for uncovered t-sets.
    
    # Let's use a penalty-based objective function to allow temporary invalid states.
    # Objective = len(active_blocks) * lambda + count_uncovered_tsets * mu
    # This allows SA to jump through invalid states to find better valid ones.
    
    current_state = set(initial_solution)
    current_cost = len(current_state)
    current_uncovered = 0 # Since initial is valid
    
    best_state = set(current_state)
    best_cost = current_cost
    
    T = 100.0
    T_min = 1.0
    alpha = 0.995
    
    # Precompute full mask
    full_mask = (1 << n_tsets) - 1
    
    # Helper to get uncovered mask and count for a state
    # To speed up, we maintain the covered_mask incrementally.
    
    covered_mask = 0
    for b in current_state:
        covered_mask |= block_masks[b]
    
    while T > T_min and time.time() < deadline - 5.0:
        # Choose a move
        # 1. Remove a random block
        # 2. Swap a random block with a random other block
        # 3. Add a random block (rarely, to escape deep traps)
        
        move_type = random.random()
        
        if move_type < 0.5 and len(current_state) > 0:
            # Remove
            b_remove = random.choice(list(current_state))
            new_state = current_state - {b_remove}
            new_covered_mask = covered_mask & ~block_masks[b_remove]
            # Check if still valid
            # Uncovered bits: full_mask ^ new_covered_mask
            # If new_covered_mask == full_mask, valid.
            if new_covered_mask == full_mask:
                new_cost = len(new_state)
                new_uncovered = 0
            else:
                new_cost = len(new_state)
                new_uncovered = (full_mask ^ new_covered_mask).bit_count()
                
            # Penalty objective
            # Let's say penalty for uncovered is high, but we want to minimize blocks.
            # If we allow invalid states, we need to balance.
            # Actually, a simpler strategy for covering designs is to ONLY accept valid moves 
            # but use "re-heating" or larger swaps to escape.
            # However, SA with penalties is more robust.
            
            # Let's stick to valid-only moves first, but if stuck, do a "kick" (add random block).
            
            if new_covered_mask == full_mask:
                # Valid removal
                current_state = new_state
                covered_mask = new_covered_mask
                current_cost = new_cost
                if current_cost < best_cost:
                    best_state = set(current_state)
                    best_cost = current_cost
            else:
                # Invalid removal. 
                # We could accept with probability exp(-delta_E / T) if we had a penalty.
                # For now, let's reject invalid moves to keep it simple and fast, 
                # but rely on swaps to find reductions.
                pass
                
        elif move_type < 0.9 and len(current_state) > 0:
            # Swap
            b_old = random.choice(list(current_state))
            b_new = random.randint(0, n_blocks - 1)
            
            if b_new in current_state:
                continue
                
            # Calculate new covered mask
            # Remove b_old, add b_new
            # Note: covered_mask might have other blocks covering the t-sets of b_old.
            # So we can't just mask out b_old if others cover them.
            # We need to recompute or be smart.
            # Smart way:
            # 1. Tentatively remove b_old.
            # 2. Check if any t-set becomes uncovered.
            # 3. If yes, check if b_new covers them.
            
            # Let's just compute the new mask.
            # Since n_tsets is small, we can afford to recompute the union if we are careful, 
            # but incremental is better.
            
            # Let's find t-sets that ONLY b_old covers (among current state).
            # This requires knowing the coverage count of each t-set.
            # Maintaining a coverage count array is more expensive to update but allows O(1) check for "will removal cause uncovered?".
            
            # Let's switch to maintaining a coverage count array `cov_counts`.
            # cov_counts[ts_idx] = number of active blocks covering ts_idx.
            
            # Re-initialize cov_counts for current_state
            # This is O(n_blocks * C(k,t)) which is fine once.
            
            # Wait, I didn't maintain cov_counts in the loop above.
            # Let's refactor to use cov_counts.
            
            # Break and restart loop logic with cov_counts?
            # No, let's just implement the cov_counts logic properly from the start of SA.
            
            pass # Placeholder for logic below
            
        else:
            # Add random block (Kick)
            b_new = random.randint(0, n_blocks - 1)
            if b_new not in current_state:
                current_state.add(b_new)
                covered_mask |= block_masks[b_new]
                current_cost += 1
                # This increases cost, but might allow future removals.
                # We don't update best_state here as it's worse.
                
        T *= alpha
        
        # To make this work, I need to rewrite the SA loop to use coverage counts.
        # The current `covered_mask` approach makes swaps O(1) for mask update, 
        # but checking validity of a swap (does it keep all covered?) requires 
        # knowing if any t-set drops to 0 coverage.
        # With masks: 
        #   old_mask = covered_mask
        #   new_mask = (old_mask & ~block_masks[b_old]) | block_masks[b_new]
        #   Valid if new_mask == full_mask.
        # This is O(1) bit operations! Very fast.
        
        # So let's use the mask approach for validity checks.
        
        # Let's re-run the SA loop correctly.
        
    # Re-implementing SA loop with correct logic
    
    # Reset state to initial
    current_state = set(initial_solution)
    covered_mask = 0
    for b in current_state:
        covered_mask |= block_masks[b]
    current_cost = len(current_state)
    best_state = set(current_state)
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.5
    alpha = 0.99
    
    while T > T_min and time.time() < deadline - 2.0:
        
        # 1. Try Remove
        if random.random() < 0.5 and len(current_state) > 1:
            b_remove = random.choice(list(current_state))
            new_covered_mask = covered_mask & ~block_masks[b_remove]
            if new_covered_mask == full_mask:
                # Valid removal
                current_state.discard(b_remove)
                covered_mask = new_covered_mask
                current_cost -= 1
                if current_cost < best_cost:
                    best_state = set(current_state)
                    best_cost = current_cost
            # Else reject (or accept with prob if penalty, but let's stick to valid)
            
        # 2. Try Swap
        elif len(current_state) > 1:
            b_old = random.choice(list(current_state))
            b_new = random.randint(0, n_blocks - 1)
            if b_new in current_state:
                continue
            
            # Calculate new mask
            # We need to be careful: 
            # new_mask = (covered_mask without b_old's unique contributions) + b_new
            # Actually, just:
            # temp_mask = covered_mask & ~block_masks[b_old]
            # new_mask = temp_mask | block_masks[b_new]
            
            temp_mask = covered_mask & ~block_masks[b_old]
            new_mask = temp_mask | block_masks[b_new]
            
            if new_mask == full_mask:
                # Valid swap
                current_state.discard(b_old)
                current_state.add(b_new)
                covered_mask = new_mask
                # Cost remains same (1 removed, 1 added)
                # But maybe it enables future removals?
                # Since cost is same, we accept it to explore the space.
                # We don't update best_cost here, but we update current state.
                
        # 3. Kick (Add block)
        else:
            b_new = random.randint(0, n_blocks - 1)
            if b_new not in current_state:
                current_state.add(b_new)
                covered_mask |= block_masks[b_new]
                current_cost += 1
                
        T *= alpha
        
    # Final cleanup: try to remove any redundant blocks from best_state
    final_state = set(best_state)
    # Compute coverage for best_state
    # We can just use the mask method again
    # Actually, let's just verify best_state is valid and try to reduce it greedily one last time.
    
    # Greedy reduction on best_state
    best_covered_mask = 0
    for b in best_state:
        best_covered_mask |= block_masks[b]
        
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for b in list(best_state):
            if best_covered_mask & ~block_masks[b] == full_mask:
                # b is redundant
                best_state.discard(b)
                best_covered_mask &= ~block_masks[b]
                improved = True
                break # Restart loop after modification
            
    # Verify final result
    result_blocks = [list(all_blocks[idx]) for idx in best_state]
    
    # Double check validity
    final_check_mask = 0
    for b in best_state:
        final_check_mask |= block_masks[b]
        
    if final_check_mask != full_mask:
        # Fallback to initial if something went wrong
        result_blocks = [list(all_blocks[idx]) for idx in initial_solution]
        
    return result_blocks

# EVOLVE-BLOCK-END
