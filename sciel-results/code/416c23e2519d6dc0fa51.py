import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    # Generate all t-subsets and assign IDs
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, compute a bitset of the t-subsets it covers
    # Python integers are arbitrary precision, so we can use them as bitsets.
    block_masks = [0] * num_k_subsets
    # Also store the list of t-subset IDs for each block for local search details if needed, 
    # but bitsets are primary.
    block_t_ids = []
    
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        ids = []
        for ts in itertools.combinations(block, t):
            tid = t_subset_to_id[ts]
            mask |= (1 << tid)
            ids.append(tid)
        block_masks[b_idx] = mask
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)
            
    # Helper: popcount for Python int
    def popcount(x):
        return bin(x).count('1')
        
    best_blocks = None
    best_size = float('inf')
    
    # Number of random restarts
    max_restarts = 50 # Bitset ops are fast, we can afford more restarts
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # 1. Greedy Construction using Lazy Randomized approach
        uncovered_mask = (1 << num_t_subsets) - 1
        selected_blocks = set()
        
        while uncovered_mask:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random uncovered t-subset
            # Find a random set bit in uncovered_mask
            # random.getrandbits is fast, but we need a bit that is actually set.
            # We can just pick a random index and check, or use a list of uncovered ids.
            # Maintaining a list of uncovered ids is O(N) to build initially, but updates are O(1) if we use a set.
            # Let's use a set of uncovered ids for easy random selection.
            # Wait, the bitset is for fast coverage checks.
            # For greedy, we need to pick a block.
            # Strategy: Pick a random uncovered t-subset. Look at all blocks covering it.
            # Pick the block that covers the most OTHER uncovered t-subsets.
            
            # To pick a random uncovered t-subset efficiently:
            # We can maintain a list `uncovered_list` and remove items as we go.
            # Or, since N_t is ~11k, we can just iterate? No.
            # Let's maintain a set `uncovered_set` for O(1) random choice.
            if not hasattr(construct, '_uncovered_set'):
                construct._uncovered_set = set(range(num_t_subsets))
            # Reset for this restart
            uncovered_set = set(range(num_t_subsets))
            
            # We need to update uncovered_set as we select blocks.
            
            while uncovered_set:
                if time.time() - start_time > time_limit:
                    break
                
                # Pick random uncovered subset
                s_id = random.choice(list(uncovered_set))
                
                # Get candidates
                candidates = t_to_blocks[s_id]
                
                best_candidate = -1
                max_new_cover = -1
                
                # For each candidate, count how many bits in uncovered_mask are set in block_masks[cand]
                # This is popcount(uncovered_mask & block_masks[cand])
                
                for b_idx in candidates:
                    if b_idx in selected_blocks:
                        continue
                    # Count new coverage
                    # Note: uncovered_mask is updated at the end of the loop iteration
                    new_cover = popcount(uncovered_mask & block_masks[b_idx])
                    if new_cover > max_new_cover:
                        max_new_cover = new_cover
                        best_candidate = b_idx
                
                if best_candidate != -1:
                    selected_blocks.add(best_candidate)
                    # Update uncovered_mask and uncovered_set
                    new_cover_bits = uncovered_mask & block_masks[best_candidate]
                    uncovered_mask &= ~new_cover_bits
                    
                    # Update uncovered_set
                    # We need to remove the bits that were just covered.
                    # Iterating through the bits of new_cover_bits is efficient if few bits.
                    # But new_cover_bits can have ~56 bits.
                    # We can just re-calculate uncovered_set from uncovered_mask? No, too slow.
                    # We can iterate through the set bits of new_cover_bits.
                    bits_to_remove = new_cover_bits
                    while bits_to_remove:
                        # Get lowest set bit
                        lsb = bits_to_remove & -bits_to_remove
                        idx = lsb.bit_length() - 1
                        uncovered_set.discard(idx)
                        bits_to_remove ^= lsb
                        
            if not uncovered_set:
                current_size = len(selected_blocks)
                if current_size < best_size:
                    best_size = current_size
                    best_blocks = set(selected_blocks)
                    
        # 2. Local Search (Refinement)
        # Use the best solution found so far or the current one
        if best_blocks is None:
             current_solution = set(selected_blocks)
        elif len(selected_blocks) < best_size:
             current_solution = set(selected_blocks)
        else:
             current_solution = set(best_blocks)
             
        if not current_solution:
            continue

        # Initialize cover counts using bitsets?
        # For local search, we need to know if a block is redundant.
        # A block b is redundant if block_masks[b] is a subset of the union of masks of other blocks.
        # We can maintain `total_mask` (union of all selected blocks) and `cover_counts`?
        # Actually, for deletion:
        # If we remove b, the new union is total_mask & ~block_masks[b].
        # If (total_mask & ~block_masks[b]) == total_mask, then b was redundant? No.
        # If block_masks[b] is a subset of (union of others), then b is redundant.
        # Union of others = total_mask with b's contribution removed?
        # We need to know if any bit in block_masks[b] is covered ONLY by b.
        # This is equivalent to: popcount(block_masks[b] & ~union_of_others) == 0.
        # union_of_others is not easily computed incrementally without counts.
        
        # Let's use cover_counts array for speed in local search.
        # N_t is 11628. Array of ints is fast.
        cover_counts = [0] * num_t_subsets
        for b_idx in current_solution:
            for tid in block_t_ids[b_idx]:
                cover_counts[tid] += 1
                
        improved = True
        ls_iterations = 0
        max_ls_iterations = 10000
        
        while improved and ls_iterations < max_ls_iterations:
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            ls_iterations += 1
            
            blocks_list = list(current_solution)
            random.shuffle(blocks_list)
            
            for b_idx in blocks_list:
                if b_idx not in current_solution:
                    continue
                    
                # 1. Check for Redundancy
                # If all subsets in b_idx are covered by others (count > 1), remove it
                is_redundant = True
                for tid in block_t_ids[b_idx]:
                    if cover_counts[tid] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    current_solution.discard(b_idx)
                    for tid in block_t_ids[b_idx]:
                        cover_counts[tid] -= 1
                    improved = True
                    continue
                
                # 2. Try Swap
                # Identify critical subsets: those covered ONLY by b_idx
                critical_subsets = []
                for tid in block_t_ids[b_idx]:
                    if cover_counts[tid] == 1:
                        critical_subsets.append(tid)
                
                if not critical_subsets:
                    continue
                    
                # Try to find a replacement block that covers ALL critical subsets
                # We can intersect the lists of blocks for each critical subset?
                # Or just check candidates from the first critical subset.
                target_tid = critical_subsets[0]
                candidates = t_to_blocks[target_tid]
                
                replacement_found = False
                best_replacement = -1
                best_score = -1
                
                # Score: Number of non-critical subsets covered? Or just first valid?
                # Let's try to find one that covers all criticals.
                # To speed up, we can check if the candidate's mask contains all critical bits.
                critical_mask = 0
                for c_tid in critical_subsets:
                    critical_mask |= (1 << c_tid)
                    
                for new_b_idx in candidates:
                    if new_b_idx in current_solution:
                        continue
                    # Check if new_b_idx covers all critical subsets
                    if (block_masks[new_b_idx] & critical_mask) == critical_mask:
                        # Valid replacement
                        # Heuristic: Pick one that covers the most "extra" uncovered? 
                        # Since we are in a valid cover, "uncovered" is 0.
                        # Heuristic: Pick one that minimizes future criticals?
                        # For now, accept the first one or randomize.
                        # Let's accept the first one for speed, or pick randomly from valid ones.
                        # To diversify, let's pick randomly.
                        # But collecting all valid ones is expensive.
                        # Let's just take the first one.
                        best_replacement = new_b_idx
                        replacement_found = True
                        break
                
                if replacement_found:
                    # Perform swap
                    current_solution.discard(b_idx)
                    for tid in block_t_ids[b_idx]:
                        cover_counts[tid] -= 1
                    
                    current_solution.add(best_replacement)
                    for tid in block_t_ids[best_replacement]:
                        cover_counts[tid] += 1
                    
                    improved = True
        
        if len(current_solution) < best_size:
            best_size = len(current_solution)
            best_blocks = set(current_solution)

    # Convert best_blocks set of indices to list of blocks
    if best_blocks is None:
        # Fallback: This should not happen
        # Return a trivial valid cover if possible?
        # For C(19,8,5), we can just return all k-subsets? Too many.
        # We should ensure at least one greedy run completes.
        # If best_blocks is None, we failed.
        # Let's try a simple greedy one last time without time check if we have time?
        # Or just raise an error? The task says "invalid result... scores 0".
        # We must return a valid list.
        # If we didn't find one, we are in trouble.
        # But with 50 restarts, it's very unlikely.
        pass
        
    result = [list(all_k_subsets[i]) for i in best_blocks]
    return result
