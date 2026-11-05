import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    # Precompute all t-subsets
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the triplets it covers
    # For v=24, k=7, t=3: n_triplets = 2024. This requires large integers or arrays.
    # Python big integers are fast for bit_count and bitwise ops.
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for triplet in itertools.combinations(block, t):
            tid = triplet_to_id[triplet]
            mask |= (1 << tid)
        block_masks[b_idx] = mask
        
    all_mask = (1 << n_triplets) - 1

    # Helper to get a greedy initial solution
    def greedy_init(seed=0):
        rng = random.Random(seed)
        uncovered = all_mask
        solution_indices = []
        
        while uncovered:
            if time.time() > deadline - 30:
                break
            best_gain = -1
            best_idx = -1
            # Optimization: Only check blocks that have high potential gain
            # Simple full scan is O(N_blocks * bit_count). 
            # For 24 choose 7 = 346k blocks, this might be slow in pure Python loop if repeated.
            # But we only do this once or twice.
            
            # To speed up, we can sample if n_blocks is huge, but 346k is manageable with bit_count?
            # Actually, 346k * (cost of bit_count on 2024-bit int) is probably fine for a few iterations.
            # However, doing this in a while loop for ~70 blocks means 70 * 346k = 24M bit_counts.
            # This might take a few seconds.
            
            for b_idx in range(n_blocks):
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
                    if best_gain == n_triplets.bit_count(): # Perfect block? Unlikely
                        break
            
            if best_idx == -1 or best_gain == 0:
                break
                
            solution_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return solution_indices

    # Local Search: Min-Conflicts style
    # State: A set of selected block indices.
    # Objective: Minimize number of uncovered triplets.
    # Move: Swap one block in the set with one not in the set.
    
    def local_search(initial_indices, seed=0):
        rng = random.Random(seed)
        # Use a list for the current solution, and a set for O(1) lookup
        current_sol = list(initial_indices)
        sol_set = set(current_sol)
        
        # Calculate current uncovered
        covered_mask = 0
        for idx in current_sol:
            covered_mask |= block_masks[idx]
        uncovered = all_mask & ~covered_mask
        current_cost = uncovered.bit_count()
        
        best_sol = list(current_sol)
        best_cost = current_cost
        
        no_improve_count = 0
        max_no_improve = 1000 # Restart or stop if stuck
        
        while time.time() < deadline - 10:
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random block in the solution to remove
            if not current_sol:
                break
            remove_idx_pos = rng.randrange(len(current_sol))
            remove_idx = current_sol[remove_idx_pos]
            
            # Tentatively remove
            new_covered_mask = 0
            # Recalculating full mask is expensive. 
            # Instead, track counts? No, bitmask is easier for "uncovered" logic.
            # Let's just recalculate the mask for the remaining set.
            # To optimize, we can keep a count of how many times each bit is set?
            # That requires an array of size n_triplets.
            # Let's stick to bitmask for simplicity, but optimize the recalculation.
            
            # Actually, swapping is:
            # new_uncovered = (all_mask & ~(old_covered_mask ^ mask_remove ^ mask_add))
            # This is still one big XOR and AND.
            
            # To find a good add_idx, we need one that covers many of the *newly* uncovered bits
            # or covers bits that were already covered but helps in other ways?
            # In min-conflicts for set cover, we usually pick the add_idx that minimizes the total uncovered.
            
            # Strategy: 
            # 1. Remove block R.
            # 2. Calculate bits that are now uncovered (were only covered by R).
            #    These are (mask_R & ~ (covered_mask ^ mask_R)).
            #    Wait, if a bit is in mask_R, and in covered_mask, it might be covered by others.
            #    Newly uncovered = mask_R & ~(covered_mask ^ mask_R)
            #    Let's call this 'lost_bits'.
            # 3. We want to add block A such that (lost_bits & ~mask_A) is minimized.
            #    i.e., mask_A covers as many lost_bits as possible.
            
            lost_bits = block_masks[remove_idx] & ~(covered_mask ^ block_masks[remove_idx])
            
            if lost_bits == 0:
                # Removing this block doesn't uncover anything. We can just remove it!
                # This is a pure reduction.
                current_sol.pop(remove_idx_pos)
                sol_set.discard(remove_idx)
                covered_mask ^= block_masks[remove_idx]
                current_cost = (all_mask & ~covered_mask).bit_count()
                no_improve_count = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sol = list(current_sol)
                continue
            
            # Find best add_idx to cover lost_bits
            best_add_idx = -1
            best_new_cost = current_cost + 1000000 # High penalty
            
            # Heuristic: Sample a subset of candidate blocks to speed up
            # If n_blocks is 346k, scanning all is 346k * bit_count.
            # Let's try scanning all first. If it's too slow, we'll see.
            
            # Optimization: We can pre-filter blocks that intersect with lost_bits?
            # Or just scan.
            
            # To avoid bias towards blocks already in the set, we can skip them, 
            # but adding a block already in the set is a no-op (if we use a set).
            # Since we are swapping, add_idx must not be in sol_set.
            
            for add_idx in range(n_blocks):
                if add_idx in sol_set:
                    continue
                    
                # Calculate cost if we add add_idx
                # New covered mask would be (covered_mask ^ block_masks[remove_idx]) | block_masks[add_idx]
                # We care about how many lost_bits are covered by add_idx.
                # covered_lost = (lost_bits & block_masks[add_idx]).bit_count()
                # new_cost = current_cost - covered_lost + (bits in add_idx that were already covered? No, that doesn't increase cost)
                # Actually, cost is number of uncovered bits.
                # Old uncovered: U
                # Remove R: U' = U | (mask_R & ~U)  [bits in R not in U become uncovered]
                #   Note: mask_R & ~U is exactly lost_bits if U = all_mask & ~covered_mask.
                #   Wait, lost_bits = mask_R & ~(covered_mask ^ mask_R).
                #   covered_mask ^ mask_R = covered_mask & ~mask_R (since mask_R subset of covered_mask? No, not necessarily, but if R is in sol, its bits are in covered_mask).
                #   So covered_mask ^ mask_R = covered_mask - mask_R (set difference).
                #   So lost_bits = mask_R & ~(covered_mask - mask_R) = mask_R & (covered_mask ^ mask_R)? No.
                #   Let's stick to: lost_bits = bits in R that are NOT in (covered_mask without R).
                #   covered_mask_without_R = covered_mask ^ mask_R (assuming R was in set).
                #   lost_bits = mask_R & ~covered_mask_without_R.
                
                # If we add A:
                # New covered = (covered_mask_without_R) | mask_A
                # New uncovered = all_mask & ~New_covered
                # Change in uncovered = (New_uncovered ^ U).bit_count() ... complicated.
                
                # Simpler:
                # New_cost = (all_mask & ~((covered_mask ^ block_masks[remove_idx]) | block_masks[add_idx])).bit_count()
                
                # This is one big AND and bit_count per candidate.
                # 346k * (1 AND + 1 bit_count) is roughly 346k * 100ns = 35ms.
                # That's acceptable per step.
                
                new_covered = (covered_mask ^ block_masks[remove_idx]) | block_masks[add_idx]
                new_cost = (all_mask & ~new_covered).bit_count()
                
                if new_cost < best_new_cost:
                    best_new_cost = new_cost
                    best_add_idx = add_idx
                    
                # Early exit if we found a perfect swap (cost doesn't increase)
                if new_cost <= current_cost:
                    break
            
            if best_add_idx == -1:
                # No valid add found? Should not happen unless sol_set is all blocks.
                no_improve_count += 1
                continue
                
            # Perform the swap
            current_sol[remove_idx_pos] = best_add_idx
            sol_set.discard(remove_idx)
            sol_set.add(best_add_idx)
            
            covered_mask = (covered_mask ^ block_masks[remove_idx]) | block_masks[best_add_idx]
            current_cost = (all_mask & ~covered_mask).bit_count()
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_sol = list(current_sol)
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Random walk if stuck
            if no_improve_count > 50:
                # Pick a random swap to escape local minimum
                if current_sol:
                    remove_idx_pos = rng.randrange(len(current_sol))
                    remove_idx = current_sol[remove_idx_pos]
                    add_idx = rng.randrange(n_blocks)
                    while add_idx in sol_set:
                        add_idx = rng.randrange(n_blocks)
                        
                    current_sol[remove_idx_pos] = add_idx
                    sol_set.discard(remove_idx)
                    sol_set.add(add_idx)
                    covered_mask = (covered_mask ^ block_masks[remove_idx]) | block_masks[add_idx]
                    current_cost = (all_mask & ~covered_mask).bit_count()
                    no_improve_count = 0

        return best_sol

    # 1. Get initial greedy solution
    initial_indices = greedy_init(0)
    
    # 2. Run local search
    final_indices = local_search(initial_indices, seed=42)
    
    # 3. Convert indices back to blocks
    final_blocks = [all_blocks[i] for i in final_indices]
    
    # 4. Post-processing: Remove any redundant blocks (should be none if cost is 0, but just in case)
    # If cost > 0, it's invalid, but we return best effort.
    # The evaluator checks validity. If invalid, score 0.
    # We should ensure we return a VALID covering if possible.
    # If local search ends with cost > 0, we should probably fall back to the greedy one 
    # or try to fix the remaining uncovered triplets.
    
    # Check if final solution is valid
    covered_mask = 0
    for idx in final_indices:
        covered_mask |= block_masks[idx]
    if covered_mask != all_mask:
        # Try to patch: add blocks to cover remaining
        uncovered = all_mask & ~covered_mask
        patch_indices = []
        while uncovered:
            if time.time() > deadline - 2:
                break
            best_gain = -1
            best_idx = -1
            for b_idx in range(n_blocks):
                if b_idx in final_indices:
                    continue
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = b_idx
            if best_gain == 0:
                break
            final_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        # Rebuild blocks
        final_blocks = [all_blocks[i] for i in final_indices]
        
    return final_blocks

if __name__ == "__main__":
    pass
