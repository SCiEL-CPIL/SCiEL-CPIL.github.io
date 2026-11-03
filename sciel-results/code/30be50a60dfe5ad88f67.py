import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute all blocks and the triples they cover
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Number of triples in a block
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] is a list of indices of triples covered by blocks[i]
    # Using a 2D numpy array for vectorized operations
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            block_triples[i, idx] = triple_to_idx[triple]
            idx += 1
            
    # Precompute triple frequencies for weighted greedy
    triple_freq = np.zeros(n_triples, dtype=np.float64)
    for i in range(n_blocks):
        triple_freq[block_triples[i]] += 1
    triple_weight = 1.0 / np.maximum(triple_freq, 1.0)
    
    def greedy_cover(use_weighted=False, random_tiebreak=False, rng=None, top_k=1):
        """Run greedy set cover."""
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Calculate gains
            if use_weighted:
                gains = np.sum(uncovered[block_triples] * triple_weight[block_triples], axis=1)
            else:
                gains = np.sum(uncovered[block_triples], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            if random_tiebreak and rng is not None:
                # Pick from top K candidates
                threshold = max_gain - 1e-9
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) > 0:
                    if len(candidates) > top_k:
                        best_idx = rng.choice(candidates)
                    else:
                        best_idx = rng.choice(candidates)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)
            
            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        return selected

    def peel_redundant(block_indices):
        """Remove redundant blocks from a solution using incremental coverage counts."""
        if not block_indices:
            return list(block_indices)
        
        n_sel = len(block_indices)
        # coverage_count[triple] = number of blocks covering it
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_indices = list(block_indices)
        
        # Iterative removal
        # We can remove a block if all its triples have coverage > 1.
        # Removing a block decreases coverage of its triples.
        # This might make other blocks redundant.
        
        while True:
            if time.time() > deadline:
                break
                
            removed_any = False
            new_indices = []
            
            for bidx in current_indices:
                my_triples = block_triples[bidx]
                # Check if any triple in this block is covered only by this block
                # If min(coverage_count[my_triples]) > 1, it is redundant
                if np.min(coverage_count[my_triples]) > 1:
                    # Redundant, remove it
                    coverage_count[my_triples] -= 1
                    removed_any = True
                else:
                    new_indices.append(bidx)
            
            current_indices = new_indices
            
            if not removed_any:
                break
                
        return current_indices

    def local_search(block_indices, rng, time_limit):
        """
        Local search to minimize block count.
        Moves:
        1. Remove redundant blocks.
        2. Swap a block with another that covers the same 'critical' triples but has better overlap.
        3. Kick: Remove a random block, repair greedily.
        """
        if not block_indices:
            return list(block_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        for bidx in block_indices:
            coverage_count[block_triples[bidx]] += 1
        
        current_blocks = list(block_indices)
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        
        # Helper to update best
        def update_best():
            nonlocal best_blocks, best_size
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_blocks = list(current_blocks)
        
        # Precompute for faster swap checks? 
        # For now, we rely on the fact that n_blocks is ~200k, but we only check candidates
        # that cover critical triples.
        
        while time.time() < time_limit:
            # 1. Try to remove redundant blocks
            peeled = peel_redundant(current_blocks)
            if len(peeled) < len(current_blocks):
                current_blocks = peeled
                # Rebuild coverage counts
                coverage_count = np.zeros(n_triples, dtype=np.int32)
                for bidx in current_blocks:
                    coverage_count[block_triples[bidx]] += 1
                update_best()
                continue
            
            # 2. Try swaps
            # Find blocks with critical triples (coverage == 1)
            # We want to swap a block that has critical triples with one that covers them
            # but overlaps more with existing blocks (making them potentially redundant).
            
            # To speed up, we don't scan all blocks every iteration.
            # We pick a random block from current_blocks that has critical triples.
            candidates_for_swap = []
            for bidx in current_blocks:
                my_triples = block_triples[bidx]
                if np.any(coverage_count[my_triples] == 1):
                    candidates_for_swap.append(bidx)
            
            if not candidates_for_swap:
                # No critical triples? Then we can just remove redundant blocks (already done)
                # Or we are in a local minimum where every block has at least one unique triple.
                # In this case, a kick is needed.
                pass
            else:
                old_bidx = rng.choice(candidates_for_swap)
                old_triples = block_triples[old_bidx]
                critical_triples = old_triples[coverage_count[old_triples] == 1]
                
                if len(critical_triples) == 0:
                    continue
                
                # Find a candidate block that covers all critical_triples
                # We sample random blocks to find one
                # Optimization: We can precompute for each triple, the list of blocks covering it.
                # But that's memory heavy. Let's just sample.
                
                best_candidate = -1
                best_score = -1
                
                # Sample 100 random blocks
                for _ in range(100):
                    cand_idx = rng.randrange(n_blocks)
                    if cand_idx == old_bidx:
                        continue
                    
                    cand_triples = block_triples[cand_idx]
                    
                    # Check if candidate covers all critical triples
                    # Use a set for critical_triples for O(1) lookup? 
                    # critical_triples is small (<= 35).
                    # np.isin is fast enough for small arrays.
                    if not np.all(np.isin(critical_triples, cand_triples)):
                        continue
                    
                    # Score: How many triples does this candidate cover that are currently 
                    # covered by exactly 1 or 2 blocks?
                    # Actually, we want to maximize the potential for removing other blocks.
                    # A good proxy: Maximize the number of triples in cand_triples that are 
                    # currently covered by >= 2 blocks (redundant coverage).
                    # This suggests the new block will make other blocks more redundant.
                    
                    # Alternatively, minimize the number of "new" unique triples introduced?
                    # No, we must cover critical ones.
                    
                    # Let's use: Sum of 1/(coverage_count[t]+1) for t in cand_triples
                    # This rewards covering rare triples, but we already know it covers critical ones.
                    # Let's try: Count of triples in cand_triples that have coverage >= 2.
                    # This indicates high overlap, which is good for peeling.
                    
                    overlap_score = np.sum(coverage_count[cand_triples] >= 2)
                    
                    if overlap_score > best_score:
                        best_score = overlap_score
                        best_candidate = cand_idx
                
                if best_candidate >= 0:
                    # Perform swap
                    coverage_count[old_triples] -= 1
                    coverage_count[block_triples[best_candidate]] += 1
                    
                    # Replace in current_blocks
                    # current_blocks is a list of indices
                    for i, b in enumerate(current_blocks):
                        if b == old_bidx:
                            current_blocks[i] = best_candidate
                            break
                    
                    # Update best if size didn't change but we might peel later
                    # We don't update best_size here, but the next loop iteration will peel.
                    # However, to avoid cycling, we might want to accept only if it looks promising.
                    # For now, we accept and let the peel step decide.

            # 3. Kick (if no progress or random decision)
            if rng.random() < 0.1:
                if len(current_blocks) > 1:
                    # Remove a random block
                    j = rng.randrange(len(current_blocks))
                    removed_bidx = current_blocks[j]
                    coverage_count[block_triples[removed_bidx]] -= 1
                    current_blocks.pop(j)
                    
                    # Rebuild to cover uncovered triples
                    uncovered_mask = (coverage_count == 0)
                    # Greedy repair
                    while np.any(uncovered_mask):
                        if time.time() > time_limit:
                            break
                        gains = np.sum(uncovered_mask[block_triples], axis=1)
                        best_add = np.argmax(gains)
                        if gains[best_add] == 0:
                            break
                        coverage_count[block_triples[best_add]] += 1
                        current_blocks.append(best_add)
                        uncovered_mask = (coverage_count == 0)
                    
                    # After repair, peel again
                    peeled = peel_redundant(current_blocks)
                    if len(peeled) < len(current_blocks):
                        current_blocks = peeled
                        coverage_count = np.zeros(n_triples, dtype=np.int32)
                        for bidx in current_blocks:
                            coverage_count[block_triples[bidx]] += 1
                        update_best()
        
        return best_blocks

    # Run multiple strategies
    best_solution = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Strategy 1: Deterministic weighted greedy
    sel = greedy_cover(use_weighted=True, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
            
    # Strategy 2: Deterministic unweighted greedy
    sel = greedy_cover(use_weighted=False, random_tiebreak=False)
    if sel:
        peeled = peel_redundant(sel)
        size = len(peeled)
        if size < best_size:
            best_size = size
            best_solution = peeled
            
    # Strategy 3+: Random restarts with weighted greedy + Local Search
    # Allocate remaining time to LS
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # Run LS for a few restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Randomized greedy
        sel = greedy_cover(use_weighted=True, random_tiebreak=True, rng=rng, top_k=5)
        if sel:
            peeled = peel_redundant(sel)
            size = len(peeled)
            
            # Apply Local Search
            # Give LS a portion of the remaining time
            ls_time_limit = deadline - 5.0
            ls_result = local_search(peeled, rng, ls_time_limit)
            ls_size = len(ls_result)
            
            if ls_size < size:
                size = ls_size
                peeled = ls_result
            
            if size < best_size:
                best_size = size
                best_solution = peeled
                
    # Final polish
    if best_solution is not None:
        final_peeled = peel_redundant(best_solution)
        best_solution = final_peeled
        best_size = len(final_peeled)
        
    if best_solution is None:
        best_solution = []
        
    # Convert indices to lists of points
    return [list(all_blocks[i]) for i in best_solution]
