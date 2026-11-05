import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute all triples and their indices
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Precompute all blocks and their triple indices
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For speed, we will use numpy arrays for block-triple incidence if possible,
    # but for v=24, k=7, t=3, the number of triples is C(24,3)=2024.
    # We can represent the coverage of a block as a set of triple indices.
    # To speed up "gain" calculation, we can precompute the list of triple indices for each block.
    
    block_triples = []
    for b in all_blocks:
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(set(tris))
        
    # We also need to know, for a given uncovered set, which blocks cover the most uncovered triples.
    # Instead of bitsets on 2024 bits (which is ~256 bytes, manageable in Python ints but slow to AND),
    # we can maintain a count of how many times each triple is covered by the *current* selection.
    # But for Greedy, we need "gain" = |Block \ Covered|.
    # If we maintain `covered_count` for each triple, then gain = sum(1 for tri in block if covered_count[tri]==0).
    # This is O(k choose t) per block check.
    # With ~17k blocks and ~70 steps, that's 17k * 35 * 70 ~ 40M operations. In Python this is too slow.
    
    # Strategy: Use a "Candidate Pool" approach or Bitsets.
    # Let's use Python integers as bitsets. 2024 bits is fine.
    # Precompute bitset for each block.
    block_bitsets = []
    for tris in block_triples:
        mask = 0
        for idx in tris:
            mask |= (1 << idx)
        block_bitsets.append(mask)
        
    # Initial uncovered is all 1s
    full_mask = (1 << num_triples) - 1
    
    def greedy_construct(randomize=False, top_k=1):
        """
        Builds a covering using greedy selection.
        If randomize is True, it picks from the top_k candidates.
        """
        uncovered = full_mask
        selected = []
        selected_set = set()
        
        while uncovered:
            # Find best block
            best_gain = -1
            candidates = []
            
            # Optimization: If we have many blocks, scanning all is expensive.
            # But 17k blocks is small enough if the inner loop is fast.
            # The bottleneck is (block_bitsets[i] & uncovered).bit_count()
            
            # To speed this up, we can pre-filter? No, gain changes.
            # Let's just iterate.
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (block_bitsets[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and best_gain > 0:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            if randomize and len(candidates) > 1:
                # Pick from top candidates
                # To make it more diverse, pick from top min(top_k, len(candidates))
                # But we only kept the ones with *max* gain.
                # A better randomization: pick from blocks with gain >= best_gain - threshold?
                # Or just pick randomly from the max gain ones.
                idx = random.choice(candidates)
            else:
                idx = candidates[0]
                
            selected.append(idx)
            selected_set.add(idx)
            uncovered &= ~block_bitsets[idx]
            
        return selected, uncovered

    def local_search(selected, uncovered):
        """
        Attempts to reduce the size of the selected set.
        Returns (new_selected, new_uncovered, improved)
        """
        # 1. Remove redundant blocks
        # Count coverage of each triple in current selection
        counts = [0] * num_triples
        for idx in selected:
            for tri_idx in block_triples[idx]:
                counts[tri_idx] += 1
                
        # Identify blocks that can be removed
        to_remove = []
        for idx in selected:
            # Check if all triples in this block are covered by >= 2 blocks
            removable = True
            for tri_idx in block_triples[idx]:
                if counts[tri_idx] == 1:
                    removable = False
                    break
            if removable:
                to_remove.append(idx)
                
        if to_remove:
            # Remove them
            for idx in to_remove:
                selected.remove(idx)
                for tri_idx in block_triples[idx]:
                    counts[tri_idx] -= 1
            # Recalculate uncovered (should be 0 if valid, but let's check)
            # Actually, if we remove redundant blocks, uncovered stays 0.
            return selected, uncovered, True
            
        # 2. Swap
        # Try to swap a block with another that covers its "critical" triples and adds more coverage?
        # Or just: try to replace a block with a different one that covers the same critical triples
        # but allows removal of another block later?
        
        # Simpler Swap: 
        # For each block B in selected:
        #   Find critical triples (covered only by B).
        #   Find a block C not in selected that covers all critical triples of B.
        #   If such C exists, and C covers *more* uncovered triples than B did? 
        #   No, B covers 0 uncovered (since we are in a valid covering).
        #   The goal of swap is to enable further removals.
        #   So, swap B with C if C covers the critical triples of B.
        #   Then, run "Remove Redundant" again.
        
        improved = False
        changed = True
        while changed:
            changed = False
            # Recount if we made changes
            counts = [0] * num_triples
            for idx in selected:
                for tri_idx in block_triples[idx]:
                    counts[tri_idx] += 1
                    
            # Try swaps
            for i, idx in enumerate(list(selected)):
                # Find critical triples for idx
                critical = []
                for tri_idx in block_triples[idx]:
                    if counts[tri_idx] == 1:
                        critical.append(tri_idx)
                        
                if not critical:
                    # This block is redundant!
                    # Remove it
                    selected.remove(idx)
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                    changed = True
                    break # Restart loop
                    
                # Find candidate C
                # We need C to cover all critical triples.
                # Iterate through all unselected blocks? Too slow.
                # Use the critical triple with the fewest candidates.
                min_candidates = None
                for tri_idx in critical:
                    # Who covers this triple?
                    # We didn't precompute "blocks_containing_triple" as a list of lists for speed?
                    # Let's precompute it.
                    pass
                
            if not changed:
                break
                
        return selected, uncovered, improved

    # Precompute blocks_containing_triple for faster swap search
    blocks_containing_triple = [[] for _ in range(num_triples)]
    for i, tris in enumerate(block_triples):
        for tri_idx in tris:
            blocks_containing_triple[tri_idx].append(i)

    def fast_local_search(selected):
        """
        Optimized local search using precomputed incidence.
        """
        # 1. Redundancy Removal
        counts = [0] * num_triples
        for idx in selected:
            for tri_idx in block_triples[idx]:
                counts[tri_idx] += 1
                
        # Loop until no more removals
        removed_any = True
        while removed_any:
            removed_any = False
            for idx in list(selected):
                # Check if redundant
                redundant = True
                for tri_idx in block_triples[idx]:
                    if counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    selected.remove(idx)
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] -= 1
                    removed_any = True
                    break # Restart count check for safety, or just continue? 
                    # Breaking is safer to ensure counts are consistent with "current" selected.
                    # Actually, if we remove one, counts change. 
                    # It's better to recalculate counts from scratch or update carefully.
                    # Updating carefully is fine.
                    
        # 2. Swap Search
        # We want to reduce size. A swap replaces 1 block with 1 block (size unchanged).
        # The benefit is that the new block might make other blocks redundant.
        
        swap_improved = True
        while swap_improved:
            swap_improved = False
            # Recount
            counts = [0] * num_triples
            for idx in selected:
                for tri_idx in block_triples[idx]:
                    counts[tri_idx] += 1
            
            selected_set = set(selected)
            
            for idx in list(selected):
                critical = []
                for tri_idx in block_triples[idx]:
                    if counts[tri_idx] == 1:
                        critical.append(tri_idx)
                        
                if not critical:
                    # Redundant, should have been caught above, but just in case
                    selected.remove(idx)
                    swap_improved = True
                    break
                    
                # Find candidate
                # Pick critical triple with smallest candidate list
                min_tri = min(critical, key=lambda x: len(blocks_containing_triple[x]))
                candidates = blocks_containing_triple[min_tri]
                
                found_swap = False
                for cand_idx in candidates:
                    if cand_idx in selected_set:
                        continue
                    
                    # Check if cand_idx covers all critical triples
                    covers_all = True
                    for tri_idx in critical:
                        if cand_idx not in blocks_containing_triple[tri_idx]:
                            # This check is O(1) if we use sets for blocks_containing?
                            # No, lists. So this is O(len(list)).
                            # Better: check bitset?
                            # We have block_bitsets.
                            # (block_bitsets[cand_idx] & (1 << tri_idx))
                            pass
                            
                    # Let's use bitsets for this check, it's faster.
                    # Create mask for critical triples
                    crit_mask = 0
                    for tri_idx in critical:
                        crit_mask |= (1 << tri_idx)
                        
                    if (block_bitsets[cand_idx] & crit_mask) == crit_mask:
                        # Valid swap
                        # Remove idx, add cand_idx
                        selected.remove(idx)
                        selected_set.remove(idx)
                        selected.append(cand_idx)
                        selected_set.add(cand_idx)
                        
                        # Update counts
                        for tri_idx in block_triples[idx]:
                            counts[tri_idx] -= 1
                        for tri_idx in block_triples[cand_idx]:
                            counts[tri_idx] += 1
                            
                        swap_improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break # Restart outer loop to recount
                    
        return selected

    best_size = float('inf')
    best_solution = []
    
    # Multi-start
    # We have ~115 seconds.
    # Each greedy + local search might take 1-5 seconds?
    # Let's run as many as possible.
    
    # Initial deterministic run
    sol, unc = greedy_construct(randomize=False)
    if not unc: # Valid
        sol = fast_local_search(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
    # Randomized restarts
    while time.time() < deadline - 5:
        # Randomize the greedy step
        # Use a "temperature" or just pick from top 5
        sol, unc = greedy_construct(randomize=True, top_k=5)
        
        if unc:
            # If greedy failed to cover (shouldn't happen with full search, but maybe if we limit candidates?),
            # we need to finish it. But our greedy searches ALL blocks for max gain, so it should always finish.
            # Unless we implemented a limit. We didn't. So unc should be 0.
            continue
            
        # Local Search
        sol = fast_local_search(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
            
        # Optional: Perturbation
        # If we have time, perturb the best solution and re-local-search
        if time.time() < deadline - 10 and len(best_solution) > 0:
            perturbed = best_solution[:]
            # Remove a random block
            if perturbed:
                rem_idx = random.choice(perturbed)
                perturbed.remove(rem_idx)
                # Re-greedy fill the holes
                # This is essentially running greedy on the "uncovered" set created by removal
                # But our greedy_construct starts from scratch.
                # We need a "fill" function.
                
                # Let's just run a quick local search on the perturbed set to see if it fixes itself?
                # No, removing a block makes it invalid.
                # We need to add blocks to cover the holes.
                
                # Calculate uncovered
                counts = [0] * num_triples
                for idx in perturbed:
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] += 1
                uncovered_mask = 0
                for i in range(num_triples):
                    if counts[i] == 0:
                        uncovered_mask |= (1 << i)
                        
                # Greedy fill
                while uncovered_mask:
                    best_gain = -1
                    best_idx = -1
                    for i in range(num_blocks):
                        if i in set(perturbed):
                            continue
                        gain = (block_bitsets[i] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                    if best_idx == -1:
                        break
                    perturbed.append(best_idx)
                    uncovered_mask &= ~block_bitsets[best_idx]
                    
                # Local search on perturbed
                perturbed = fast_local_search(perturbed)
                
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_solution = perturbed[:]

    # Convert indices back to blocks
    return [all_blocks[idx] for idx in best_solution]
