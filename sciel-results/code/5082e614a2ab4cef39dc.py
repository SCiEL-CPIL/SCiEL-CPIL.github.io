import time
import random
import itertools
import numpy as np
import heapq

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for 120s limit

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    
    # Map t-subset to index
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # All possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    N_blocks = len(all_blocks)
    
    # For each block, which t-subsets does it cover?
    # block_cover[b] = list of indices of t-subsets contained in block b
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    # Convert to numpy arrays for faster set operations if needed, 
    # but lists are fine for small N_t (11628) and N_blocks (75582).

    def greedy_cover_fast():
        """
        Optimized Greedy Set Cover.
        Instead of scanning all blocks, we maintain a priority queue of potential blocks.
        However, the gain of a block changes as we select other blocks.
        A standard efficient heuristic is to repeatedly pick the block that covers the most 
        currently uncovered t-subsets.
        
        To avoid O(N_blocks * N_t) per step, we can use a "lazy" approach:
        1. Start with a priority queue of all blocks, sorted by their initial coverage size (all C(8,5)=56).
        2. Since initial sizes are equal, we need a tie-breaker or dynamic update.
        
        Alternative Fast Strategy:
        Randomized Greedy with Restart.
        Given the symmetry, simple greedy is often near-optimal.
        We will implement a greedy that picks the block covering the most uncovered subsets.
        To speed this up, we note that initially all blocks cover 56 subsets.
        As we cover subsets, the "gain" of a block decreases.
        
        We can approximate the gain by sampling.
        Or, we can use a "most constrained first" approach:
        1. Pick the uncovered t-subset with the fewest candidate blocks.
        2. Among the blocks covering it, pick the one that covers the most OTHER uncovered subsets.
        
        This is often much faster than global greedy because the candidate set is small.
        """
        # counts[t] = number of selected blocks covering t
        # uncovered_mask[t] = True if t is not covered by any selected block
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected_blocks = []
        
        # Static count of how many blocks cover each t-subset (for tie-breaking/urgency)
        # In C(19,8,5), every t-subset is covered by C(19-5, 8-5) = C(14,3) = 364 blocks.
        # So static urgency is uniform. Dynamic urgency is what matters.
        
        while num_uncovered > 0:
            # Find an uncovered t-subset
            # To make the search efficient, we can pick a random uncovered one, 
            # or one that is "hard" to cover. Since all are symmetric initially, 
            # random is fine, but as we go, some might become "isolated" if we were 
            # using a non-uniform distribution? No, the problem is symmetric.
            # However, the *set of selected blocks* breaks symmetry.
            
            # Let's pick a random uncovered t-subset to guide the candidate selection
            # This avoids scanning all N_t every time if we maintain a list, 
            # but np.where is O(N_t) = 11k, which is fast.
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-subset
            target_t = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Get all blocks that cover this target_t
            candidates = t_to_blocks[target_t]
            
            # Among these candidates, find the one that covers the most *other* uncovered t-subsets
            best_block = -1
            best_gain = -1
            
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            # Select the best block
            selected_blocks.append(best_block)
            
            # Update uncovered mask
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected_blocks

    def local_search_drop_and_repair(blocks):
        """
        Local Search: Try to remove blocks.
        If removing a block breaks validity (some t-subset becomes uncovered),
        we must add new blocks to cover them. If we can add fewer blocks than we removed,
        we improve. If we add the same number, it's neutral (useful for escaping local minima).
        If we add more, we reject.
        
        Strategy:
        1. Iterate through blocks.
        2. For each block B, tentatively remove it.
        3. Identify t-subsets that are now uncovered (count went from 1 to 0).
        4. If none, keep it removed (improvement).
        5. If some, solve a small set cover problem for these uncovered t-subsets 
           using blocks NOT in the current set (or even in the set, but that's redundant).
           Actually, we can use ANY block.
        6. If the number of new blocks added is < 1 (impossible, min 1) or == 1 (neutral),
           accept the change.
           Wait, removing 1 and adding 1 is neutral. Removing 1 and adding 0 is improvement.
           We want to minimize size.
        
        To make this more powerful, we can try to remove a block and see if the resulting
        "hole" can be filled by a single block that is *better* (covers more other weakly covered sets).
        """
        current_blocks = list(blocks)
        if not current_blocks:
            return current_blocks
            
        # Calculate counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        improved = True
        while improved:
            improved = False
            # Try to remove each block
            # We iterate in reverse to handle index shifts easily, or just use a flag
            for i in range(len(current_blocks) - 1, -1, -1):
                bi = current_blocks[i]
                
                # Check if bi is the unique cover for any t-subset
                unique_ts = []
                for ti in block_cover[bi]:
                    if counts[ti] == 1:
                        unique_ts.append(ti)
                        
                if not unique_ts:
                    # Redundant! Remove it.
                    current_blocks.pop(i)
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    improved = True
                    continue
                
                # If it has unique coverage, can we swap it with a better block?
                # Or just try to remove it and cover the holes with 1 block?
                # If we can cover all unique_ts with 1 block B_new, we swap.
                # This is neutral in size, but might enable future reductions.
                
                # Find intersection of blocks covering all t in unique_ts
                if len(unique_ts) == 1:
                    candidates = set(t_to_blocks[unique_ts[0]])
                else:
                    # Start with smallest candidate set
                    min_len = float('inf')
                    min_ti = unique_ts[0]
                    for ti in unique_ts:
                        if len(t_to_blocks[ti]) < min_len:
                            min_len = len(t_to_blocks[ti])
                            min_ti = ti
                            
                    candidates = set(t_to_blocks[min_ti])
                    for ti in unique_ts[1:]:
                        candidates &= set(t_to_blocks[ti])
                        
                candidates.discard(bi)
                
                if candidates:
                    # Pick a random candidate to swap
                    # We prefer candidates that cover many *other* uncovered/weakly covered sets?
                    # For now, random swap to explore.
                    new_bi = random.choice(list(candidates))
                    
                    # Perform swap
                    # Remove bi
                    for ti in block_cover[bi]:
                        counts[ti] -= 1
                    # Add new_bi
                    for ti in block_cover[new_bi]:
                        counts[ti] += 1
                    current_blocks[i] = new_bi
                    improved = True
                    # Break to restart the scan? Or continue?
                    # Continuing is fine, but the counts have changed.
                    # Let's break to restart the outer loop to ensure consistency.
                    break
                    
        return current_blocks

    # Main Search Loop
    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Multiple Greedy Restarts
    # The "target t-subset" choice in greedy_cover_fast is random.
    # Running multiple restarts finds different local optima.
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        random.seed(rng.randint(0, 1000000)) # Re-seed global random for reproducibility of the run? 
        # Actually, construct uses local random in greedy_cover_fast.
        # Let's pass a seed or use a local RNG.
        
        # Modify greedy_cover_fast to accept an RNG
        # For simplicity, we'll just call it. The internal random.randint uses global state.
        # We should isolate RNGs.
        
        # Let's refactor greedy to use a local RNG
        def greedy_cover_with_rng(seed_rng):
            uncovered_mask = np.ones(N_t, dtype=bool)
            num_uncovered = N_t
            selected_blocks = []
            
            while num_uncovered > 0:
                uncovered_indices = np.where(uncovered_mask)[0]
                if len(uncovered_indices) == 0:
                    break
                
                target_t = uncovered_indices[seed_rng.randint(0, len(uncovered_indices)-1)]
                candidates = t_to_blocks[target_t]
                
                best_block = -1
                best_gain = -1
                
                for bi in candidates:
                    gain = 0
                    for ti in block_cover[bi]:
                        if uncovered_mask[ti]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = bi
                
                selected_blocks.append(best_block)
                
                for ti in block_cover[best_block]:
                    if uncovered_mask[ti]:
                        uncovered_mask[ti] = False
                        num_uncovered -= 1
                            
            return selected_blocks

        local_rng = random.Random(rng.randint(0, 1000000))
        blocks = greedy_cover_with_rng(local_rng)
        blocks = local_search_drop_and_repair(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_solution = blocks
            
    # Phase 2: Intensive Local Search on the best solution
    if best_solution is not None:
        current_blocks = list(best_solution)
        
        # Run local search multiple times with different randomizations
        for _ in range(20):
            if time.time() > deadline - 2.0:
                break
            # Shuffle the order of blocks to try different removal sequences
            random.shuffle(current_blocks)
            new_blocks = local_search_drop_and_repair(current_blocks)
            if len(new_blocks) < best_len:
                best_len = len(new_blocks)
                best_solution = new_blocks
                current_blocks = list(new_blocks)
            else:
                # If no improvement, maybe revert to best_solution for next iteration?
                # Or keep current if it's the same?
                # local_search_drop_and_repair modifies in place or returns new list.
                # It returns a new list.
                # If it didn't improve, we can try to "kick" it with a random swap.
                if len(current_blocks) > 0:
                    idx = random.randint(0, len(current_blocks)-1)
                    # Random swap with a valid block?
                    # Just restart local search on a perturbed version?
                    pass

    if best_solution is None:
        # Fallback
        local_rng = random.Random(123)
        best_solution = greedy_cover_with_rng(local_rng)
        best_solution = local_search_drop_and_repair(best_solution)
        
    return [list(all_blocks[bi]) for bi in best_solution]
