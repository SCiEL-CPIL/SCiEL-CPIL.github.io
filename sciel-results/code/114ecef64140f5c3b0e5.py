import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Bitsets for blocks
    block_bitsets = [0] * num_blocks
    # For each block, store the list of triple indices it covers (for faster counting in local search)
    block_triple_indices = [[] for _ in range(num_blocks)]
    
    # Precompute for each triple, which blocks cover it
    # This is useful for finding candidates in swap
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for i, b in enumerate(all_blocks):
        bitset = 0
        tris_in_block = []
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
            tris_in_block.append(idx)
            triple_to_blocks[idx].append(i)
        block_bitsets[i] = bitset
        block_triple_indices[i] = tris_in_block

    # Helper: Check if a solution is valid (should always be true if we maintain it)
    # But for safety, we return the best valid one found.
    
    best_blocks_list = None
    best_size = num_blocks # Initialize to worst case
    
    # Multi-start loop
    # We will try a few diverse greedy constructions and then intensify with local search.
    # Number of starts depends on time. Each start should take < 10s.
    
    def greedy_construct(alpha=0.0):
        """
        Construct a covering using randomized greedy.
        alpha: 0.0 means deterministic best-first. >0 means GRASP-like restricted list.
        """
        uncovered = (1 << num_triples) - 1
        selected = set()
        selected_blocks = []
        
        while uncovered:
            # Calculate gains for all unselected blocks
            # Optimization: We can just iterate and compute gain = (bitset & uncovered).bit_count()
            # To save time, we might cache or prune, but N=346k is manageable if we do it carefully.
            # Actually, 346k * bit_count() might be slow if done every step (70 steps * 346k * op).
            # 70 * 346k = 24 million ops. In Python, this might take a few seconds.
            # Let's try to optimize.
            
            best_gain = -1
            best_indices = []
            min_gain = float('inf')
            
            # We need the top candidates.
            # If alpha == 0, we just need the max.
            # If alpha > 0, we need the range.
            
            # To speed up, we can pre-sort blocks by some heuristic? No, gain changes.
            # We will just scan.
            
            # Optimization: If we have a large list, scanning 346k blocks every step is O(N * K).
            # With K~70, that's 24M operations.
            # Python `bit_count()` is fast.
            
            for i in range(num_blocks):
                if i in selected:
                    continue
                gain = (block_bitsets[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                    min_gain = gain
                elif gain == best_gain:
                    best_indices.append(i)
                if gain < min_gain:
                    min_gain = gain
            
            if best_gain <= 0:
                break # Should not happen if uncovered is not 0 and there are blocks
            
            # Determine selection
            if alpha == 0.0:
                # Deterministic: pick first or random from best
                if len(best_indices) > 1:
                    # Random tie-breaking to add slight diversity even if alpha=0?
                    # No, keep it deterministic for the first run to get a strong baseline.
                    chosen = best_indices[0]
                else:
                    chosen = best_indices[0]
            else:
                # GRASP: threshold = best_gain - alpha * (best_gain - min_gain)
                threshold = best_gain - alpha * (best_gain - min_gain)
                # Collect all blocks with gain >= threshold
                candidates = []
                for i in range(num_blocks):
                    if i in selected:
                        continue
                    gain = (block_bitsets[i] & uncovered).bit_count()
                    if gain >= threshold:
                        candidates.append(i)
                
                if not candidates:
                    candidates = best_indices
                    
                chosen = random.choice(candidates)
            
            selected.add(chosen)
            selected_blocks.append(chosen)
            uncovered &= ~block_bitsets[chosen]
            
            # Time check inside construction?
            # If time is tight, break early? No, we need a valid solution.
            
        return selected_blocks

    def local_search(selected_blocks):
        """
        Perform local search to minimize the number of blocks.
        """
        # Use a list for selected blocks for easier modification
        sol = list(selected_blocks)
        
        # Calculate initial counts
        counts = [0] * num_triples
        for idx in sol:
            for tri_idx in block_triple_indices[idx]:
                counts[tri_idx] += 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Pass 1: Remove Redundant Blocks
            # A block is redundant if it doesn't cover any triple exclusively.
            i = 0
            while i < len(sol):
                idx = sol[i]
                is_redundant = True
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove
                    sol.pop(i)
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    improved = True
                    # Don't increment i, re-check same position
                else:
                    i += 1
            
            if time.time() > deadline - 5:
                break
                
            # Pass 2: 1-Swap
            # Try to swap each block with another block that covers its unique triples
            # and provides better overall efficiency (e.g., covers more other triples, or reduces redundancy elsewhere)
            
            # To make this faster, we iterate through blocks.
            # For each block b_i, find its unique triples U_i.
            # If U_i is empty, it was removed in Pass 1.
            # We look for a block b_j not in sol such that b_j covers all U_i.
            # Among such b_j, we prefer one that minimizes the increase in "waste" or maximizes gain.
            # Simple heuristic: Pick b_j that covers the most triples NOT covered by any other block in sol \ {b_i}?
            # Or simply: Pick b_j that has the highest gain on (uncovered by sol\{b_i})?
            # Actually, since we are fully covered, "uncovered" is empty.
            # The goal is to keep coverage while maybe allowing future removals.
            # A common metric: Replace b_i with b_j if b_j covers U_i and b_j's coverage overlaps more with other blocks (making it "less unique" and potentially removable later?)
            # Or, replace if b_j covers U_i and (number of triples covered by b_j that are covered by >=2 blocks in sol) is high?
            
            # Let's try a simpler swap:
            # If we can find b_j that covers U_i, and b_j is "better" than b_i.
            # How to define better?
            # Let's just try the first valid swap we find? No, that might lead to a bad local minimum.
            # Let's try to find a swap that increases the number of redundant blocks?
            
            # Alternative: Just try to swap b_i with b_j if b_j covers U_i.
            # To avoid infinite loops, we only accept a swap if it leads to a removal in the next Pass 1, 
            # or if we use a potential function.
            # Given time limits, let's try swaps that cover U_i and have a high "shared" count.
            
            swap_done = False
            for i, idx in enumerate(sol):
                if time.time() > deadline - 5:
                    break
                    
                # Find unique triples for idx
                unique_tris = []
                for tri_idx in block_triple_indices[idx]:
                    if counts[tri_idx] == 1:
                        unique_tris.append(tri_idx)
                
                if not unique_tris:
                    # Should have been removed, but maybe counts changed?
                    continue
                
                # Find candidates b_j that cover all unique_tris
                # Intersect the lists of blocks covering each unique triple
                # Start with the triple that has the fewest covering blocks
                min_len = float('inf')
                min_tri = -1
                for tri_idx in unique_tris:
                    if len(triple_to_blocks[tri_idx]) < min_len:
                        min_len = len(triple_to_blocks[tri_idx])
                        min_tri = tri_idx
                
                candidates = triple_to_blocks[min_tri]
                
                valid_candidate = -1
                best_score = -1
                
                for cand_idx in candidates:
                    if cand_idx in sol:
                        continue
                    
                    # Check if cand covers all unique_tris
                    # We can use bitsets for speed
                    covers_all = True
                    cand_bitset = block_bitsets[cand_idx]
                    for tri_idx in unique_tris:
                        if not (cand_bitset & (1 << tri_idx)):
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Score the candidate
                        # We want a candidate that is "redundant" with the rest of the solution.
                        # i.e., covers many triples that are already covered by others.
                        # Calculate overlap with (sol \ {idx})
                        # This is expensive.
                        # Heuristic: Just pick the first one?
                        # Or pick one with max total triples (which is constant k choose t)? No.
                        # Let's pick one that minimizes the number of *new* unique triples it introduces?
                        # Actually, if we swap idx with cand_idx, the new unique triples of cand_idx are those covered by cand_idx but NOT by any other block in sol.
                        # We want this number to be small (ideally 0, so it becomes redundant).
                        
                        # Let's approximate: Count how many triples in cand_idx are covered by >=2 blocks in sol.
                        # If this count is high, the new block is "safe" (redundant).
                        
                        # To speed up, we can just check if cand_idx covers any triple with count==1 in the CURRENT solution?
                        # No, those are the unique_tris we are replacing.
                        
                        # Let's use a simple metric: 
                        # The number of triples in cand_idx that have count >= 2 in the current counts (excluding idx's contribution? No, counts include idx).
                        # If we remove idx, counts of its triples drop by 1.
                        # We add cand_idx.
                        
                        # Let's just try the first valid candidate to save time.
                        # If we find a swap, we break and restart the local search.
                        valid_candidate = cand_idx
                        break # Pick first valid to save time? 
                        # This might be too greedy. Let's try a few.
                        # For now, to ensure progress, first valid is okay if we do multiple passes.
                    
                if valid_candidate != -1:
                    # Perform Swap
                    # Remove idx
                    for tri_idx in block_triple_indices[idx]:
                        counts[tri_idx] -= 1
                    
                    # Add valid_candidate
                    sol[i] = valid_candidate
                    for tri_idx in block_triple_indices[valid_candidate]:
                        counts[tri_idx] += 1
                    
                    improved = True
                    swap_done = True
                    break # Restart local search passes
                
            if swap_done:
                continue # Go back to Pass 1
        
        return sol

    # Run multiple starts
    # Start 1: Deterministic Greedy (alpha=0)
    if time.time() < deadline - 20:
        sol = greedy_construct(0.0)
        sol = local_search(sol)
        if len(sol) < best_size:
            best_size = len(sol)
            best_blocks_list = [all_blocks[i] for i in sol]
            
    # Start 2-5: GRASP with varying alpha
    alphas = [0.1, 0.2, 0.3, 0.5]
    for alpha in alphas:
        if time.time() > deadline - 10:
            break
            
        # Run a few iterations for each alpha?
        # Let's do 2 iterations per alpha.
        for _ in range(2):
            if time.time() > deadline - 5:
                break
            sol = greedy_construct(alpha)
            sol = local_search(sol)
            if len(sol) < best_size:
                best_size = len(sol)
                best_blocks_list = [all_blocks[i] for i in sol]
                
    # If we still have time, try a few random restarts of local search on the best solution?
    # Or just return the best.
    
    if best_blocks_list is None:
        # Fallback: Should not happen
        return []
        
    return best_blocks_list
