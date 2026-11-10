import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute bitset mask for each block
    # mask[bi] is an integer where bit i is 1 if tsub_to_idx[combi] is in block bi
    block_masks = [0] * num_blocks
    for bi, b in enumerate(all_blocks):
        mask = 0
        for tsub in itertools.combinations(b, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        block_masks[bi] = mask
            
    full_mask = (1 << num_tsubsets) - 1
    
    # 1. Initial Solution: Greedy Set Cover
    # We use a fast greedy to get a valid starting point.
    # Instead of checking all blocks, we can just pick random blocks that cover uncovered triples?
    # No, greedy is better.
    # To speed up greedy, we precompute for each triple, the list of blocks covering it.
    # This is memory heavy but num_tsubsets * avg_blocks is manageable for v=24.
    # C(24,3)=2024, C(24,7)=346k. Avg blocks per triple ~ 346k * 35 / 2024 ~ 5900?
    # 2024 * 5900 ~ 12M integers. ~100MB. Might be risky.
    
    # Alternative Greedy:
    # Maintain list of uncovered triples.
    # For each step, pick a random uncovered triple, and among blocks covering it, pick the one with max gain.
    # This is O(NumBlocks) per step if we scan all.
    # Better: Precompute inverted index but maybe just scan all blocks is fast enough with bitsets?
    # 346k blocks * 100 steps = 34M bit_count ops. In Python this is ~30s. Too slow.
    
    # Let's use a smarter initial construction:
    # Randomized Greedy:
    # Pick a random uncovered triple.
    # Consider only blocks containing that triple.
    # Pick the one with max marginal gain.
    
    # We need the inverted index for this to be fast.
    # Let's build it.
    triple_to_blocks = [[] for _ in range(num_tsubsets)]
    for bi in range(num_blocks):
        m = block_masks[bi]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lsb
            
    # Greedy Construction
    selected_blocks = []
    covered_mask = 0
    
    # To avoid O(N^2) in greedy, we just run it.
    # We need a list of uncovered triples to pick from.
    # Maintaining it is hard.
    # We will just pick a random triple index and check if it's uncovered.
    
    while covered_mask != full_mask and time.time() < deadline - 10:
        # Find an uncovered triple
        # We can iterate through all_tsubsets indices and check bit.
        # Or keep a list of uncovered?
        # Let's just pick random indices until we find an uncovered one.
        # With 2024 triples, this is fast.
        
        while True:
            t_idx = random.randint(0, num_tsubsets - 1)
            if not (covered_mask >> t_idx) & 1:
                break
                
        candidates = triple_to_blocks[t_idx]
        best_block = -1
        max_gain = 0
        
        for bi in candidates:
            gain = (block_masks[bi] & ~covered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block = bi
                
        if best_block == -1:
             # Should not happen
             break
             
        selected_blocks.append(best_block)
        covered_mask |= block_masks[best_block]
        
    # If we timed out in greedy, we might not have a valid solution.
    # We need to ensure validity for the SA start.
    # If invalid, we must fill it.
    if covered_mask != full_mask:
         # Emergency fill
         while covered_mask != full_mask and time.time() < deadline - 5:
             while True:
                 t_idx = random.randint(0, num_tsubsets - 1)
                 if not (covered_mask >> t_idx) & 1:
                     break
             candidates = triple_to_blocks[t_idx]
             # Pick best
             best_block = -1
             max_gain = 0
             for bi in candidates:
                 gain = (block_masks[bi] & ~covered_mask).bit_count()
                 if gain > max_gain:
                     max_gain = gain
                     best_block = bi
             if best_block == -1: break
             selected_blocks.append(best_block)
             covered_mask |= block_masks[best_block]
             
    # If still invalid (should be impossible unless v,k,t impossible), return empty or best effort?
    # Task says valid when every t-subset lies in at least one block.
    # If we can't cover, we fail. But C(24,7,3) is coverable.
    
    # 2. Simulated Annealing
    # State: Set of selected block indices.
    # Objective: Minimize len(selected_blocks).
    # Constraint: covered_mask == full_mask.
    
    # Moves:
    # 1. Remove a block (if it doesn't uncover any triple).
    #    Check: (block_mask & ~covered_mask_without_this_block) == 0?
    #    Actually, if we remove B, new_mask = covered_mask ^ (block_mask[B] & ~ (covered_mask_without_B))?
    #    Simpler: Calculate unique contribution of B.
    #    unique = block_mask[B] & ~ (OR of all other blocks)
    #    If unique == 0, we can remove B.
    #    Calculating OR of all others is O(N) bit ops.
    
    # 2. Swap a block with another.
    #    Remove B_old, add B_new.
    #    Check if new set is valid.
    #    If valid, accept if it reduces size (impossible for swap, size same) or with probability?
    #    Wait, swap keeps size same. We want to reduce size.
    #    So Swap is only useful if it enables a subsequent Removal.
    #    Or, we can do "Swap" where we replace 1 block with 2 blocks? No, increases size.
    #    Replace 2 blocks with 1? Hard.
    
    # Better SA Moves for Covering:
    # 1. **Removal**: Try to remove a random block. If valid, accept (improvement).
    # 2. **Swap**: Pick a random block B_old. Pick a random block B_new (not in set).
    #    Replace B_old with B_new.
    #    If valid, accept with probability exp(-delta/T)?
    #    But delta is 0 (size unchanged).
    #    We accept if it improves "potential" for removal?
    #    Or we just accept valid swaps with some probability to escape local optima?
    #    Standard SA accepts non-improving moves. Here, non-improving in terms of cost (size) is neutral.
    #    But we need to explore the space of valid sets of the same size to find a removable block.
    
    # 3. **Add**: Pick a random block not in set, add it.
    #    Increases size. Only accept if it enables a removal later?
    #    In SA, we accept worse moves with prob exp(-delta/T).
    #    Here delta = +1.
    
    # Let's implement:
    # Cost = len(selected_blocks)
    # We maintain the set as a Python set of indices for O(1) membership.
    # And a list for iteration?
    # Set is fine.
    
    current_set = set(selected_blocks)
    current_cost = len(current_set)
    
    # Recalculate covered_mask for current_set to be sure
    covered_mask = 0
    for bi in current_set:
        covered_mask |= block_masks[bi]
        
    if covered_mask != full_mask:
         # Fallback: Return greedy result if SA start is invalid (shouldn't happen)
         return [list(all_blocks[bi]) for bi in selected_blocks]
         
    best_set = set(current_set)
    best_cost = current_cost
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # Precompute list of all block indices for random selection
    all_indices = list(range(num_blocks))
    
    while time.time() < deadline - 2 and T > min_T:
        # 1. Try Removal
        if current_cost > 0:
            # Pick random block from current_set
            # To pick random from set, convert to list? Expensive if large.
            # We can keep a list representation and a set for lookup.
            # Let's maintain a list `current_list` and a set `current_set`.
            # If we remove, we remove from both.
            # If we add, we add to both.
            
            # Let's switch representation to list + set for efficiency.
            pass # We'll implement this in the loop below
            
        # Let's restart the loop structure with list+set
        
    # Re-implementing loop with list+set
    current_list = list(current_set)
    current_set = set(current_list)
    current_cost = len(current_list)
    
    # We need to be able to remove an item from the list efficiently?
    # Removing from middle of list is O(N). N is ~80. OK.
    
    T = 1.0
    while time.time() < deadline - 2 and T > min_T:
        # Random Move Type
        r = random.random()
        
        if r < 0.5:
            # Try Removal
            if current_cost == 0:
                T *= cooling_rate
                continue
            idx = random.randint(0, current_cost - 1)
            bi_to_remove = current_list[idx]
            
            # Check if removal is valid
            # Calculate mask of all OTHER blocks
            # mask_others = 0
            # for j, bj in enumerate(current_list):
            #     if j != idx:
            #         mask_others |= block_masks[bj]
            
            # This is O(N) per move. N~80. 80 * 10000 moves = 800k ops. Fast.
            
            mask_others = 0
            for j, bj in enumerate(current_list):
                if j != idx:
                    mask_others |= block_masks[bj]
            
            if (block_masks[bi_to_remove] & ~mask_others) == 0:
                # Valid removal
                # Remove from list and set
                current_list.pop(idx)
                current_set.remove(bi_to_remove)
                current_cost -= 1
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = set(current_list)
                    
                # Accept (Improvement)
                # Do not cool? Or cool slowly?
                # Usually accept improvements always.
                
            else:
                # Invalid removal. Reject.
                pass
                
        elif r < 0.9:
            # Try Swap
            if current_cost == 0:
                T *= cooling_rate
                continue
            # Pick block to remove
            idx = random.randint(0, current_cost - 1)
            bi_old = current_list[idx]
            
            # Pick block to add (random from all_blocks not in current_set)
            # Sampling without replacement from large set?
            # Just pick random index, check if in set.
            while True:
                bi_new = random.randint(0, num_blocks - 1)
                if bi_new not in current_set:
                    break
                    
            # Calculate new mask
            # old_mask = covered_mask
            # new_mask = old_mask ^ (block_masks[bi_old] & ~ (covered_mask_without_old)) | block_masks[bi_new]
            
            # Simpler:
            # mask_others = 0
            # for j, bj in enumerate(current_list):
            #    if j != idx: mask_others |= block_masks[bj]
            
            # new_covered = mask_others | block_masks[bi_new]
            # if new_covered == full_mask: Valid
            
            mask_others = 0
            for j, bj in enumerate(current_list):
                if j != idx:
                    mask_others |= block_masks[bj]
            
            new_covered = mask_others | block_masks[bi_new]
            
            if new_covered == full_mask:
                # Valid swap. Cost unchanged.
                # Accept with probability 1? Or exp(0/T)=1?
                # We accept to explore.
                current_list[idx] = bi_new
                current_set.remove(bi_old)
                current_set.add(bi_new)
                # Cost same
            else:
                # Invalid. Reject.
                pass
                
        else:
            # Try Add
            # Pick random block not in set
            while True:
                bi_new = random.randint(0, num_blocks - 1)
                if bi_new not in current_set:
                    break
                    
            # Add it
            current_list.append(bi_new)
            current_set.add(bi_new)
            current_cost += 1
            
            # Cost increased by 1.
            # Accept with prob exp(-1/T)
            if random.random() < (2.718281828 ** (-1.0 / T)):
                # Accepted
                pass
            else:
                # Rejected. Revert
                current_list.pop()
                current_set.remove(bi_new)
                current_cost -= 1
                
        T *= cooling_rate
        
    # Finalize
    # Use best_set found
    final_blocks = [list(all_blocks[bi]) for bi in best_set]
    
    # Verify validity of best_set (paranoia)
    # The evaluator will check, but we want to ensure we don't return invalid if best_set was tracked incorrectly.
    # However, best_set is only updated on valid improvements.
    # Initial state was valid.
    # So best_set should be valid.
    
    return final_blocks

if __name__ == "__main__":
    # Quick test
    start = time.time()
    blocks = construct(24, 7, 3)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
    
    # Verify
    all_tsubs = set(itertools.combinations(range(24), 3))
    covered = set()
    for b in blocks:
        for tsub in itertools.combinations(b, 3):
            covered.add(tsub)
    if all_tsubs == covered:
        print("Valid")
    else:
        print("Invalid")
