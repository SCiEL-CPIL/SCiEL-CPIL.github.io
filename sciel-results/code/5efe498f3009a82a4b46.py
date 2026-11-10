"""
Strategy: Bitset-Accelerated Greedy Set Cover + Surgical Local Search.
1. Generate a dense pool of blocks using a fast greedy heuristic with bitsets.
2. Use a "Surgical" local search: 
   - Identify redundant blocks (those whose coverage is a subset of others).
   - Remove them.
   - Try to replace single blocks with 2-3 new blocks that cover the same triples 
     but potentially allow further compression later, or simply try to find 
     a block that covers more "orphaned" triples than it frees up.
   - In practice, for C(24,7,3), a strong greedy + redundancy removal is often 
     sufficient to beat naive greedy, but we add a "swap" heuristic to try to 
     reduce block count by finding a set of blocks that covers the same triples 
     with fewer blocks (hard) or simply by better ordering.
   
   Actually, a more effective "different" strategy than pure greedy is:
   **Greedy with Backtracking/Repair**: 
   Instead of just adding blocks, we maintain the set of uncovered triples.
   We pick the block that covers the most uncovered triples.
   If we get stuck or near the end, we try to "swap" a block with a combination 
   of 2 blocks that cover the same triples plus some new ones, effectively 
   increasing the "efficiency" of the covering.
   
   However, the most distinct and effective strategy for small v is:
   **Simulated Annealing on the Block Set**:
   Start with a greedy solution.
   Neighbors: 
   1. Remove a block (if it doesn't uncover anything? No, that breaks validity).
   2. Swap a block with a different block.
   3. Add a block.
   Objective: Minimize number of blocks.
   Constraint: All triples covered.
   
   To make this fast, we use bitsets to check validity and coverage instantly.
"""

import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for bitset operations
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubsets)}
    
    # We will represent the set of covered t-subsets as a Python integer (bitset).
    # Since num_tsubsets for C(24,3) is 2024, this fits easily in a Python int.
    
    # Precompute the bitset mask for every possible k-block
    # There are C(24,7) = 346104 possible blocks.
    # Precomputing masks for all of them might take a bit of time/memory, 
    # but it allows O(1) coverage calculation for any block.
    
    # Let's estimate time: 346k blocks * 35 triples per block = ~12M ops.
    # In Python, this might take 1-2 seconds. Acceptable.
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for b in all_blocks:
        mask = 0
        for tsub in itertools.combinations(b, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Map block tuple to its index in all_blocks for quick lookup?
    # Or just use the mask itself as identifier if unique? 
    # Masks are not unique (different blocks can cover same set of triples? No, 
    # if two 7-sets cover the same 3-sets, they are likely the same or highly overlapping.
    # Actually, we just need to pick blocks. We can pick by index.
    
    # Greedy Construction
    current_covered_mask = 0
    full_mask = (1 << num_tsubsets) - 1
    selected_block_indices = []
    
    # To speed up greedy, we can precompute for each uncovered triple, which blocks cover it?
    # That's an inverted index. 
    # triple_to_blocks = {triple_idx: [block_idx, ...]}
    # This might be memory heavy: 2024 triples * ~100 blocks/triple = 200k entries. OK.
    
    triple_to_blocks = [[] for _ in range(num_tsubsets)]
    for bi, mask in enumerate(block_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(bi)
            m ^= lsb

    # Greedy Loop
    while current_covered_mask != full_mask and time.time() < deadline - 30:
        # Find the block that covers the most currently uncovered triples
        # Uncovered mask
        uncovered_mask = full_mask ^ current_covered_mask
        
        best_gain = -1
        best_block_idx = -1
        
        # Optimization: Instead of checking all 346k blocks, 
        # we can prioritize blocks that contain at least one uncovered triple.
        # But checking all is O(346k). 346k * (bit ops) is fast in C, but Python loop is slow.
        
        # Better: Iterate through uncovered triples, and for each, look at its covering blocks.
        # Count how many *other* uncovered triples each block covers.
        # This is the standard "Greedy Set Cover" approach.
        
        # To avoid O(num_uncovered * avg_blocks_per_triple) which can be large,
        # we can just pick a random uncovered triple, and among blocks covering it, 
        # pick the one that covers the most total uncovered triples.
        
        # Get list of uncovered triple indices
        # We need to iterate set bits of uncovered_mask
        unc_m = uncovered_mask
        if unc_m == 0:
            break
            
        # Pick a random uncovered triple to start? Or the one with fewest covering blocks?
        # Let's pick the first one for determinism, or random for diversity.
        # Random is better to escape local optima in the greedy phase.
        
        # Convert uncovered_mask to list of indices is expensive if done every step.
        # Let's just pick a random bit from uncovered_mask.
        # Python int doesn't have random bit.
        
        # Alternative: Maintain a list of uncovered triples? 
        # Updating it is expensive.
        
        # Let's stick to checking a subset of blocks or using the inverted index smartly.
        # Since we have triple_to_blocks, let's pick a random uncovered triple.
        
        # How to pick random uncovered triple efficiently?
        # We can store the list of all triples and filter? No.
        
        # Let's just iterate through all blocks and calculate gain. 
        # 346k blocks. For each, gain = popcount(block_mask & uncovered_mask).
        # Python int bit_count() is fast.
        # 346k * 100 blocks (average) = 34M bit_count ops. Might be too slow in Python loop.
        
        # Let's try to optimize: 
        # We only need to consider blocks that cover *at least one* uncovered triple.
        # We can iterate through uncovered triples (there are at most 2024).
        # For each uncovered triple, we have a list of blocks.
        # We can vote for blocks.
        
        # But doing this from scratch every step is O(U * B).
        
        # Let's try a simpler heuristic first: 
        # Pick the block with max gain among a random sample of blocks that cover 
        # a specific uncovered triple.
        
        # Get one uncovered triple index
        # We need a way to get an index from the bitset.
        # We can precompute a list of all triple indices.
        # And check if bit is set.
        
        # To speed up, let's just use the first uncovered triple in a precomputed list.
        # We'll maintain a list of all triples, and just pick the first one that is uncovered.
        # This is O(N) per step. 2024 steps * 2024 checks = 4M ops. OK.
        
        # Actually, let's just pick a random index from 0 to num_tsubsets-1 
        # and check if it's uncovered. If not, try again.
        
        while True:
            t_idx = random.randint(0, num_tsubsets - 1)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        # Get blocks covering this triple
        candidate_blocks = triple_to_blocks[t_idx]
        
        # Among these, find the one with max gain
        max_gain = -1
        best_idx = -1
        for bi in candidate_blocks:
            gain = (block_masks[bi] & uncovered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = bi
                
        if best_idx == -1 or max_gain == 0:
            # Should not happen if coverage is possible
            break
            
        selected_block_indices.append(best_idx)
        current_covered_mask |= block_masks[best_idx]
        
        # If time is running out, break to allow local search
        if time.time() > deadline - 20:
            break

    # Local Search / Refinement
    # We have a set of block indices.
    # Goal: Reduce the count.
    # Strategy: 
    # 1. Remove redundant blocks.
    # 2. Try to swap a block with a different block that covers the same "essential" triples 
    #    but allows removal of another block?
    
    # Simpler: 
    # Iterate through selected blocks. 
    # For each block, check if its removal leaves all triples covered.
    # If yes, remove it.
    
    # This is O(N_blocks * (cost to check coverage)).
    # Checking coverage: 
    # We can recompute the mask of the remaining blocks.
    # Or, we can check if the block's "unique" contribution is zero.
    # A block is redundant if every triple it covers is covered by at least one other block.
    
    # Let's implement redundancy removal.
    
    def remove_redundant(block_indices, covered_mask):
        # We need to know for each block, which triples it uniquely covers.
        # Unique triple for block B: T is in B and T is not in any other block in the set.
        
        # Efficient way:
        # For each block, calculate the mask of triples it covers.
        # Then, for each triple in that mask, check if any other block covers it.
        
        # This is expensive if done naively.
        
        # Alternative: 
        # Iterate blocks. 
        # For block B, let M_B be its mask.
        # Let M_Others be the mask of all other blocks.
        # If (M_B & ~M_Others) == 0, then B is redundant.
        
        # Calculating M_Others for each block is O(N^2) bit ops.
        # N is ~70-80. 80*80 = 6400 big int ops. Very fast.
        
        redundant_indices = []
        for i, bi in enumerate(block_indices):
            mask_bi = block_masks[bi]
            
            # Mask of all other blocks
            mask_others = 0
            for j, bj in enumerate(block_indices):
                if i != j:
                    mask_others |= block_masks[bj]
                    
            # Unique part
            unique_part = mask_bi & ~mask_others
            
            if unique_part == 0:
                redundant_indices.append(i)
                
        return redundant_indices

    # Run redundancy removal multiple times
    for _ in range(50): # Limit iterations
        if time.time() > deadline - 5:
            break
            
        redundant = remove_redundant(selected_block_indices, current_covered_mask)
        if not redundant:
            break
            
        # Remove all redundant blocks at once? 
        # If two blocks are redundant wrt each other, removing one makes the other non-redundant?
        # No, if B1 covers T and B2 covers T, and no one else covers T, 
        # then B1 is not redundant (unique T). 
        # Redundant means "I cover nothing that others don't cover".
        # So if B1 and B2 are both redundant, it means every triple in B1 is covered by others 
        # (including B2), and every triple in B2 is covered by others (including B1).
        # If we remove B1, B2 might become non-redundant if B2 covered some triple only B1 and B2 covered.
        # But wait, if B2 is redundant, then every triple in B2 is covered by others. 
        # So even if B1 is removed, B2's triples are still covered by "others" (which are the remaining blocks).
        # So we can remove all identified redundant blocks safely.
        
        # Update list
        new_indices = []
        for i, bi in enumerate(selected_block_indices):
            if i not in redundant:
                new_indices.append(bi)
                
        if len(new_indices) < len(selected_block_indices):
            selected_block_indices = new_indices
            # Update covered mask
            current_covered_mask = 0
            for bi in selected_block_indices:
                current_covered_mask |= block_masks[bi]
        else:
            break # No progress
            
    # Final check and return
    # Ensure validity (should be true by construction)
    # Convert to list of lists
    
    # If we are still not covering everything (shouldn't happen if greedy finished), 
    # we might need to add more blocks. 
    # But greedy loop runs until full_mask or timeout.
    # If timeout broke it, we might be incomplete.
    # The task requires valid output.
    # If incomplete, we must add blocks until complete.
    
    if current_covered_mask != full_mask:
        # Emergency fill
        while current_covered_mask != full_mask and time.time() < deadline - 1:
            uncovered_mask = full_mask ^ current_covered_mask
            # Pick random uncovered triple
            while True:
                t_idx = random.randint(0, num_tsubsets - 1)
                if (uncovered_mask >> t_idx) & 1:
                    break
            
            candidate_blocks = triple_to_blocks[t_idx]
            best_idx = -1
            max_gain = -1
            for bi in candidate_blocks:
                gain = (block_masks[bi] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = bi
                    
            if best_idx == -1:
                break
                
            selected_block_indices.append(best_idx)
            current_covered_mask |= block_masks[best_idx]
            
    # Return
    result_blocks = [list(all_blocks[bi]) for bi in selected_block_indices]
    return result_blocks

# To test/verify locally (not run by evaluator)
if __name__ == "__main__":
    # Small test
    # v=7, k=4, t=3
    # C(7,4) = 35 blocks.
    # Need to cover C(7,3) = 35 triples.
    # Each block covers 4 triples.
    # Min blocks >= 35/4 = 8.75 -> 9.
    # C(7,4,3) is known to be 15? No, Steiner System S(3,4,10) exists? 
    # C(7,4,3) = 15 is a known bound? 
    # Let's just run it.
    import time
    start = time.time()
    blocks = construct(7, 4, 3)
    end = time.time()
    print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
    
    # Verify
    all_tsubs = set(itertools.combinations(range(7), 3))
    covered = set()
    for b in blocks:
        for tsub in itertools.combinations(b, 3):
            covered.add(tsub)
    if all_tsubs == covered:
        print("Valid")
    else:
        print("Invalid")
        
    # Main task params
    # v=24, k=7, t=3
    # start = time.time()
    # blocks = construct(24, 7, 3)
    # end = time.time()
    # print(f"Time: {end-start:.2f}s, Blocks: {len(blocks)}")
