import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets and map to indices
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all k-blocks and their coverage masks
    # A block is a tuple of k points
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # For each block, compute the bitmask of triples it covers
    # Python integers can handle 2024 bits easily
    block_masks = [0] * num_blocks
    # Also, for each triple, store list of block indices containing it
    blocks_for_triple = [[] for _ in range(num_triples)]
    
    for i, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
            blocks_for_triple[idx].append(i)
        block_masks[i] = mask
        
    # 1. Greedy Construction
    uncovered_mask = (1 << num_triples) - 1
    selected_blocks = []
    selected_indices = set()
    
    while uncovered_mask:
        if time.time() > deadline:
            break
        best_block_idx = -1
        best_gain = 0
        # To speed up, we might not need to check all blocks every time if we use a priority queue, 
        # but with 346k blocks and ~76 iterations, 346k * 76 = 26M checks. 
        # Each check is a bit count. 26M bit counts in Python might be ~2-5 seconds. Acceptable.
        # However, we can optimize by only checking blocks that intersect uncovered_mask.
        
        for i in range(num_blocks):
            if i in selected_indices:
                continue
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == num_triples: # Perfect block? Unlikely
                    break
        
        if best_block_idx == -1:
            break # Should not happen if valid
            
        selected_blocks.append(best_block_idx)
        selected_indices.add(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # Convert to list of blocks for return
    current_solution = [blocks[i] for i in selected_blocks]
    
    # 2. Local Search
    # We maintain the set of selected block indices
    # We need to track which triples are covered by which blocks in the current solution
    # To find "unique" triples of a block, we need to know if any other selected block covers them.
    
    def get_unique_triples(block_idx, selected_set):
        """Returns list of triple indices that are covered by block_idx but NOT by any other block in selected_set."""
        mask = block_masks[block_idx]
        # We need to subtract coverage from other blocks
        # Efficient way: iterate through triples in this block, check if covered by others
        # Since block size is small (7), it has C(7,3)=35 triples.
        unique = []
        block_tups = blocks[block_idx]
        for triple in itertools.combinations(block_tups, t):
            idx = triple_to_idx[triple]
            # Check if any OTHER selected block covers this triple
            # We can check blocks_for_triple[idx]
            covered_by_other = False
            for other_idx in blocks_for_triple[idx]:
                if other_idx != block_idx and other_idx in selected_set:
                    covered_by_other = True
                    break
            if not covered_by_other:
                unique.append(idx)
        return unique

    def try_improve(selected_set):
        """Attempts to reduce the size of selected_set. Returns True if improved."""
        improved = False
        # Iterate over a copy of the list to allow modification
        for b_idx in list(selected_set):
            if time.time() > deadline:
                return improved
            
            # Find unique triples of b_idx
            unique_triples = get_unique_triples(b_idx, selected_set)
            
            if not unique_triples:
                # Redundant block
                selected_set.remove(b_idx)
                improved = True
                continue
            
            # Try to replace b_idx with a single block c that covers all unique_triples
            # Candidates must contain all triples in unique_triples
            candidates = None
            for ut in unique_triples:
                cand_list = set(blocks_for_triple[ut])
                if candidates is None:
                    candidates = cand_list
                else:
                    candidates &= cand_list
            
            if candidates is None:
                continue
                
            # Remove b_idx from candidates (it's the one we're replacing)
            candidates.discard(b_idx)
            
            # We prefer candidates NOT in selected_set (otherwise it's a no-op or swap with existing)
            # If c is in selected_set, it already covers unique_triples, so unique_triples would be empty.
            # So candidates are naturally not in selected_set (for the unique parts).
            # But we should ensure we don't add a block that is already there (which is impossible for unique parts).
            
            # Find the best candidate
            # Heuristic: Pick the candidate that covers the most *total* uncovered triples? 
            # Or just pick the first one?
            # Let's pick the one that maximizes the number of *other* triples it covers (potential for future redundancy)
            best_c = -1
            best_score = -1
            
            for c_idx in candidates:
                if c_idx in selected_set:
                    continue
                # Score: number of triples in c_idx that are already covered by selected_set (excluding b_idx)
                # This indicates how much "redundancy" it adds, which might allow other blocks to be removed later.
                # Or simpler: just accept the first valid candidate.
                # Let's use a simple score: bit_count of mask & (union of others)
                # This is expensive. Let's just accept the first one to keep it fast.
                best_c = c_idx
                break # Or iterate all to find best?
                # To be slightly smarter, let's check if this candidate allows immediate further reduction.
                # That's complex. Let's just take the first.
                
            if best_c != -1:
                selected_set.remove(b_idx)
                selected_set.add(best_c)
                improved = True
                # We could restart the loop or continue. Continuing is fine.
                
        return improved

    selected_set = set(selected_blocks)
    
    while time.time() < deadline:
        if not try_improve(selected_set):
            # Try a different order or randomization?
            # Let's try a second pass with different tie-breaking or just break if no improvement
            # To avoid getting stuck, we can randomize the order of blocks we check
            import random
            random.shuffle(list(selected_set)) # Can't shuffle set, make list
            # Actually, try_improve iterates over list(selected_set). 
            # If no improvement, we might be in a local minimum.
            # Let's try a "kick": randomly remove 1 block and re-greedy?
            # Or just break.
            break
            
    # Convert back to list of blocks
    final_blocks = [blocks[i] for i in selected_set]
    
    # Final check for validity (optional, but good for safety)
    # And ensure distinctness
    final_blocks = list(set(tuple(b) for b in final_blocks))
    
    return [list(b) for b in final_blocks]
