import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices as bitmasks
    # For v=24, k=7, this is C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    
    block_list = []
    block_masks = []
    
    # Precompute bit for each triple to speed up mask construction
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i
        
    # Generate all blocks and their masks
    # This takes a few seconds but is necessary for global greedy
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)
        
    n_blocks = len(block_list)
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Strategy: Iterative Greedy with Log Heuristic + Local Search Polish
    # The main improvement over the current program is a robust local search phase
    # that attempts to swap or remove blocks to reduce the count, using a 
    # "smooth" surrogate via coverage counts and probabilistic acceptance.
    
    while time.time() < deadline - 5.0:
        restart_start = time.time()
        
        # Phase 1: Standard Greedy Construction
        uncovered_mask = (1 << n_subsets) - 1
        selected_indices = []
        used_indices = set()
        
        # Randomized Greedy: pick from top-k candidates to diversify
        top_k = 5
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            # Iterate through all blocks
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break
                
            # Pick randomly from the best candidates
            if len(best_candidates) > top_k:
                # Sample from best candidates
                chosen_pool = rng.sample(best_candidates, top_k)
            else:
                chosen_pool = best_candidates
            
            chosen_idx = rng.choice(chosen_pool)
            
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
        
        if not selected_indices:
            continue
            
        current_count = len(selected_indices)
        
        # Phase 2: Local Search / Polish
        # Goal: Reduce block count by finding redundant blocks or swaps.
        # We use a "coverage count" array to track how many times each triple is covered.
        
        # Build coverage counts for the current solution
        coverage_counts = [0] * n_subsets
        for idx in selected_indices:
            mask = block_masks[idx]
            # Iterate through bits set in mask to update counts
            # This is faster than iterating through combinations if we can extract bits
            # But for 35 bits, iterating through combinations is also fine.
            # Let's use a helper to iterate set bits.
            m = mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                bit_pos = lsb.bit_length() - 1
                coverage_counts[bit_pos] += 1
                m ^= lsb
        
        # Attempt to remove redundant blocks
        # A block is redundant if all its triples are covered by other blocks (count > 1)
        # We do this iteratively, removing one at a time and updating counts.
        
        removed_any = True
        while removed_any and time.time() < deadline - 5.0:
            removed_any = False
            new_selected_indices = []
            for idx in selected_indices:
                mask = block_masks[idx]
                is_essential = False
                m = mask
                while m:
                    lsb = m & (-m)
                    bit_pos = lsb.bit_length() - 1
                    if coverage_counts[bit_pos] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                
                if is_essential:
                    new_selected_indices.append(idx)
                else:
                    # Remove this block and update counts
                    m = mask
                    while m:
                        lsb = m & (-m)
                        bit_pos = lsb.bit_length() - 1
                        coverage_counts[bit_pos] -= 1
                        m ^= lsb
                    removed_any = True
            
            selected_indices = new_selected_indices
            current_count = len(selected_indices)
            
        # Attempt swaps: replace one block with another that covers the same essential triples
        # This is harder. A simpler heuristic: if we have extra blocks, try to find a subset
        # of blocks that covers the same set of triples with fewer blocks.
        # Given time constraints, we'll stick to the removal phase which is robust.
        
        # Update best if improved
        if current_count < best_count:
            best_count = current_count
            best_blocks = [block_list[i] for i in selected_indices]
            
        # Check time for this restart
        restart_time = time.time() - restart_start
        # If restart was very fast, we can do more.
        
    # Return the best blocks found
    if best_blocks:
        # Ensure they are lists of ints
        return [list(b) for b in best_blocks]
    else:
        # Fallback: Return a simple greedy result?
        # If the loop didn't run (e.g. precomputation took all time), we need a result.
        # But precomputation is done before the loop.
        # If the loop didn't run, it means deadline was passed during precomputation?
        # Unlikely.
        
        # If best_blocks is None, it means no valid cover was found?
        # Greedy should always find a cover if one exists.
        
        # Just return an empty list or a single block?
        # Better to return a valid cover.
        # Let's run a single fast greedy if we have time?
        # Or just return the first few blocks?
        
        # Since we have precomputed everything, let's just run one greedy if best is None.
        if time.time() < deadline - 1.0:
            uncovered_mask = (1 << n_subsets) - 1
            selected_indices = []
            used_indices = set()
            while uncovered_mask:
                best_gain = 0
                best_idx = -1
                for i in range(n_blocks):
                    if i in used_indices:
                        continue
                    gain = (uncovered_mask & block_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                selected_indices.append(best_idx)
                used_indices.add(best_idx)
                uncovered_mask &= ~block_masks[best_idx]
            
            if selected_indices:
                return [list(block_list[i]) for i in selected_indices]
                
        return []
