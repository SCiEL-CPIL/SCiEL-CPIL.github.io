import time
import random
import itertools
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Strategy: Construct a covering using Greedy Set Cover with precomputation,
    then apply Iterated Local Search (Reduction + Swap) to minimize the number of blocks.
    
    C(19, 8, 5):
    - Universe: All 5-subsets of {0..18}. Size = C(19,5) = 11628.
    - Sets: All 8-subsets of {0..18}. Size = C(19,8) = 75582.
    - Each set covers C(8,5) = 56 subsets.
    
    1. Precomputation:
       - Map each 5-subset to an integer ID.
       - For each 8-block, compute the list of 5-subset IDs it covers.
       - Build an inverse index: for each 5-subset ID, the list of 8-block IDs that cover it.
       
    2. Construction:
       - Greedy Set Cover: Iteratively pick the block that covers the most currently uncovered 5-subsets.
       - To handle ties and local optima, we might use a randomized greedy or multiple restarts.
       - Given the size, a single good greedy run might be enough to get near 335, but local search is key.
       
    3. Local Search:
       - Maintain `cover_count` for each 5-subset (how many blocks in current solution cover it).
       - Maintain `current_blocks` (list of block IDs).
       
       Phase A: Reduction (Deletion)
       - Iterate through blocks in random order.
       - For a block B, check if any 5-subset it covers has `cover_count == 1`.
       - If yes, B is essential.
       - If no, B is redundant. Remove it, decrement cover_counts for its subsets.
       - Repeat until no more blocks can be removed.
       
       Phase B: Swap (2-opt style)
       - Pick a random block B_i from current solution.
       - Pick a random block B_j from the universe (or from a candidate set).
       - Check if swapping B_i for B_j is valid:
         - For every 5-subset s covered by B_i:
           - If `cover_count[s] == 1` (only covered by B_i), then B_j MUST cover s.
           - If B_j does not cover s, swap is invalid.
       - If valid, perform swap:
         - Remove B_i: decrement cover_count for subsets of B_i.
         - Add B_j: increment cover_count for subsets of B_j.
         - Update current_blocks.
       - After swap, run Phase A (Reduction) to remove any newly redundant blocks.
       
       Acceptance:
       - We only accept swaps if they lead to a valid cover (which they do by construction check) 
         and we keep the state if the block count decreases (after reduction).
         Since we always reduce to minimal, the count either stays same or decreases.
         We want to explore the state space to find a configuration where reduction yields a smaller core.
         
    4. Time Management:
       - Precomputation is fast (~1 sec).
       - Greedy is fast (~1-2 sec).
       - Local search loop until deadline.
    
    Optimization Details:
    - Representing subsets as IDs:
      Use combinatorial number system (combinadics) to map combinations to unique integers.
      Or simply use a dictionary {tuple: id}. 11628 entries is small.
    - Block subsets:
      For each of the 75582 blocks, store the list of 56 subset IDs.
      This is 75582 * 56 ints ~ 4M ints. ~32MB memory. Acceptable.
    - Inverse index:
      For each subset ID, list of block IDs.
      Total entries = 75582 * 56 ~ 4M. ~32MB memory. Acceptable.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # --- 1. Precomputation ---
    
    # Generate all t-subsets and map to IDs
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {sub: i for i, sub in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Generate all k-blocks
    # We might not need to generate all if we use on-the-fly, but 75k is small enough to precompute
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # Precompute covered subset IDs for each block
    # block_covered[block_id] = list of subset_ids
    block_covered = [None] * n_blocks
    
    # Precompute inverse index
    # subset_to_blocks[subset_id] = list of block_ids
    subset_to_blocks = [[] for _ in range(n_subsets)]
    
    for b_id, block in enumerate(k_blocks):
        covered_ids = []
        for sub in itertools.combinations(block, t):
            s_id = subset_to_id[sub]
            covered_ids.append(s_id)
            subset_to_blocks[s_id].append(b_id)
        block_covered[b_id] = covered_ids
        
    # --- 2. Construction (Greedy) ---
    
    # We can use a simple greedy here. 
    # To speed it up, we can track uncovered count.
    # But since n_blocks is only 75k, scanning all is O(75k * 56) per step.
    # 335 steps * 75k * 56 ~ 1.4B operations. This might be too slow in pure Python.
    
    # Optimization for Greedy:
    # Instead of scanning all blocks, we can use the "uncovered subset" approach.
    # Pick a random uncovered subset s.
    # The candidates are blocks covering s (from subset_to_blocks[s]).
    # Among these, pick the one with the highest gain (covering most other uncovered subsets).
    # This is a standard "Greedy with Candidate Restriction".
    # It's much faster because the number of blocks covering a specific subset is C(v-t, k-t) = C(14, 3) = 364.
    # So per step, we only check ~364 candidates.
    # 335 steps * 364 * 56 ~ 6.8M operations. Very fast.
    
    def greedy_construct():
        cover_count = np.zeros(n_subsets, dtype=np.int32)
        uncovered = set(range(n_subsets))
        selected_blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            # To avoid O(N) to pick random from set, we could maintain a list,
            # but set is fine for 11k.
            s_id = random.choice(list(uncovered))
            
            # Get all blocks covering this subset
            candidates = subset_to_blocks[s_id]
            
            best_block_id = -1
            best_gain = -1
            
            # Evaluate candidates
            for b_id in candidates:
                # Gain is number of subsets in this block that are currently uncovered
                # We can compute this by iterating over block_covered[b_id]
                # and checking if in uncovered.
                # To make this faster, we can maintain a boolean array `is_uncovered`?
                # Or just check membership in the set. Set lookup is O(1).
                
                gain = 0
                for sub_id in block_covered[b_id]:
                    if sub_id in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_id = b_id
            
            # Add best block
            if best_block_id != -1:
                selected_blocks.append(best_block_id)
                # Update cover_count and uncovered
                for sub_id in block_covered[best_block_id]:
                    if sub_id in uncovered:
                        uncovered.discard(sub_id)
                    cover_count[sub_id] += 1
            
            # Safety check
            if time.time() > deadline - 20:
                break
        
        return selected_blocks, cover_count
    
    # Run one greedy construction
    current_blocks, cover_count = greedy_construct()
    
    # --- 3. Local Search ---
    
    # Helper: Remove redundant blocks
    def reduce_blocks(blocks, cover_count):
        # Iterate until stable
        changed = True
        while changed:
            changed = False
            # Shuffle to break symmetry
            random.shuffle(blocks)
            i = 0
            while i < len(blocks):
                b_id = blocks[i]
                # Check if any subset covered by b_id has cover_count == 1
                is_essential = False
                for sub_id in block_covered[b_id]:
                    if cover_count[sub_id] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Redundant, remove it
                    blocks.pop(i)
                    for sub_id in block_covered[b_id]:
                        cover_count[sub_id] -= 1
                    changed = True
                    # Don't increment i, check next block at same index
                else:
                    i += 1
            if time.time() > deadline - 10:
                break
        return blocks, cover_count
    
    # Initial reduction
    current_blocks, cover_count = reduce_blocks(current_blocks, cover_count)
    best_count = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # Local Search Loop
    while time.time() < deadline - 5:
        
        # Phase A: Reduction (might have been done after previous swap, but ensure clean state)
        # We assume current state is reduced.
        
        # Phase B: Swap
        # Pick a random block to remove
        if not current_blocks:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        b_remove = current_blocks[i]
        
        # Find subsets uniquely covered by b_remove
        unique_subs = []
        for sub_id in block_covered[b_remove]:
            if cover_count[sub_id] == 1:
                unique_subs.append(sub_id)
        
        # If no unique subs, we can swap with ANY block? 
        # Actually, if no unique subs, removing it keeps cover valid.
        # Then we can add any block and reduce.
        # But we want to improve.
        
        # Pick a candidate block to add
        # Strategy: Pick a random block, or a block that covers many of the unique_subs (if any).
        # If unique_subs is empty, we just add a random block to explore, then reduce.
        # If unique_subs is not empty, the new block MUST cover all of them.
        
        if unique_subs:
            # Find blocks that cover all unique_subs
            # Intersection of subset_to_blocks[s] for all s in unique_subs
            # Since unique_subs is small (usually 1 or 2), this is fast.
            candidates = set(subset_to_blocks[unique_subs[0]])
            for s in unique_subs[1:]:
                candidates.intersection_update(set(subset_to_blocks[s]))
            
            if not candidates:
                # No valid swap
                continue
                
            # Pick a random candidate from the intersection
            b_add = random.choice(list(candidates))
        else:
            # No unique constraints. Pick a random block.
            # To be efficient, pick a random block from the universe.
            b_add = random.randint(0, n_blocks - 1)
            
        if b_add == b_remove:
            continue
            
        # Perform Swap
        # 1. Remove b_remove
        for sub_id in block_covered[b_remove]:
            cover_count[sub_id] -= 1
        
        # 2. Add b_add
        for sub_id in block_covered[b_add]:
            cover_count[sub_id] += 1
            
        # 3. Update block list
        current_blocks[i] = b_add
        
        # 4. Reduce
        new_blocks, new_count = reduce_blocks(current_blocks, cover_count)
        
        if len(new_blocks) < best_count:
            best_count = len(new_blocks)
            best_blocks = new_blocks.copy()
            
        # Keep the state for further search (even if not better than best, it's a valid state)
        current_blocks = new_blocks
        cover_count = new_count

    # Convert block IDs back to lists of points
    result = [list(k_blocks[b_id]) for b_id in best_blocks]
    return result

# EVOLVE-BLOCK-START
# The code above is the implementation.
# EVOLVE-BLOCK-END
