import itertools
import random
import time
import numpy as np
from collections import Counter

# Precomputation for the specific problem v=19, k=7, t=4
# However, the function signature is construct(v, k, t), so we must handle general v, k, t?
# The prompt says: "The evaluator passes this task's parameters v, k and t."
# And "Keep the function names and signatures".
# But the task is specifically C(19,7,4).
# The seed program works for general v, k, t.
# If I hardcode for 19,7,4, it might fail if the evaluator tests other parameters?
# "The evaluator passes this task's parameters v, k and t." usually implies the specific task parameters.
# "Score: the number of blocks... it is an upper bound on the covering number C(v, k, t)."
# Given the context "Covering design C(19,7,4)", it is highly likely v=19, k=7, t=4 is fixed.
# However, to be safe, I should check if v, k, t match 19, 7, 4 and use the optimized path, otherwise fallback to a simpler greedy or the seed logic.
# But the seed is slow.
# Let's assume v=19, k=7, t=4 for the optimized SA, and use a fallback for others.
# Actually, the precomputation $\binom{19}{4}$ is small. For larger v, it explodes.
# If v is larger, SA might not fit in 120s.
# I will implement the SA for the specific case and a greedy fallback.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # If parameters are not the specific target, use a faster greedy or seed-like approach
    # because SA precomputation might be too heavy or slow for large v.
    if v != 19 or k != 7 or t != 4:
        # Fallback: Simple Greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Find block covering most uncovered
            best_block = None
            best_count = -1
            # This is O(C(v,k) * C(k,t)) which is huge for large v.
            # For fallback, let's just use the seed logic but with a time check.
            # Actually, let's just return a valid but potentially large covering if it's not 19,7,4.
            # The task is specifically C(19,7,4).
            # I will assume the evaluator only tests 19,7,4 or similar small cases.
            # If it's a different case, I'll use the seed's greedy but with time guard.
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                # Optimize gain calculation
                best_x = None
                best_g = -1
                for x in range(v):
                    if x in block: continue
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            g += 1
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x is None: break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
            if time.time() > deadline:
                # Return what we have, but it might be invalid.
                # We must return a valid covering.
                # If we break early, we must finish covering.
                # This fallback is risky for large v.
                # But the task is 19,7,4.
                pass
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    # 1. Precompute 4-subsets
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # 2. Precompute blocks and their covered t-subsets using bitsets for speed
    # Number of blocks = C(19,7) = 50388.
    # Each block covers C(7,4) = 35 t-subsets.
    # We represent the set of covered t-subsets as a Python integer bitmask.
    # num_tsubs = C(19,4) = 3876. This fits comfortably in a Python int.
    
    all_blocks = []
    block_masks = [] # list of int masks
    block_tsubs = [] # list of lists of tsub IDs (for delta calculation in SA)
    
    # Precompute mapping from block tuple to index?
    # We need to check if a block is in the current set.
    # Using a set of frozensets is fine, but we also need to map back to index for mask?
    # Or just store the mask in the set? No, we need the block itself to update counts.
    # Let's store block tuples in a set for membership.
    
    block_set = set()
    
    for block_tuple in itertools.combinations(points, k):
        mask = 0
        tsub_ids = []
        for tsub in itertools.combinations(block_tuple, t):
            tid = tsub_to_id[tsub]
            mask |= (1 << tid)
            tsub_ids.append(tid)
        
        all_blocks.append(block_tuple)
        block_masks.append(mask)
        block_tsubs.append(tsub_ids)
        block_set.add(frozenset(block_tuple))
        
    # 3. Greedy Initialization with Bitset Acceleration
    # cover_count is still needed for SA deltas, but greedy can use masks.
    # Actually, for greedy, we just need to know if a tsub is covered.
    # We can maintain a single bitmask `covered_mask` for all currently covered tsubs.
    # Gain of a block = popcount(block_mask & ~covered_mask)
    
    covered_mask = 0
    current_blocks = []
    # We also maintain cover_count for the SA phase later.
    # But greedy is fast with masks.
    
    # To speed up greedy, we can also precompute the list of blocks sorted by some heuristic?
    # No, the gain changes.
    
    # Optimization: Use numpy or just Python int popcount.
    # Python 3.8+ has int.bit_count().
    
    while time.time() < deadline:
        # Check if all covered
        # Total mask has all bits set.
        # (1 << num_tsubs) - 1
        if covered_mask == (1 << num_tsubs) - 1:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        # Iterate all blocks
        # 50k blocks.
        for i, mask in enumerate(block_masks):
            gain = (mask & ~covered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break # Should not happen
            
        current_blocks.append(all_blocks[best_block_idx])
        covered_mask |= block_masks[best_block_idx]
        
        # Time check
        if time.time() > deadline:
            break
            
    # 4. Initialize SA state from Greedy result
    # We need cover_count array for SA.
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    for b in current_blocks:
        # b is a tuple
        fs = frozenset(b)
        # Find index? Or just iterate tsubs.
        # We have block_tsubs list. We need to map block to its tsub ids.
        # We can create a dict: frozenset -> list of tsub ids
        # Or just recompute tsub ids from the block tuple?
        # It's faster to have a dict.
        
    # Let's build the dict now
    block_to_tsubs = {}
    for i, b in enumerate(all_blocks):
        block_to_tsubs[frozenset(b)] = block_tsubs[i]
        
    for b in current_blocks:
        for tid in block_to_tsubs[frozenset(b)]:
            cover_count[tid] += 1
            
    num_uncovered = np.sum(cover_count == 0)
    
    # 5. Simulated Annealing / Local Search
    # The previous SA was slow. 
    # Improvement: 
    # 1. Use bitsets for "is covered" checks if possible? 
    #    SA needs exact counts for delta calculation (1->0 or 0->1).
    #    So we must use cover_count.
    # 2. Optimize the inner loop of SA.
    #    - Remove: iterate tsub_ids of block, check count == 1.
    #    - Add: iterate tsub_ids of block, check count == 0.
    #    This is O(35) per move. Very fast.
    #    The bottleneck is the Python loop overhead and `rng.random()`.
    
    # Let's run a high-throughput local search.
    # Strategy: 
    # - Try to remove random blocks. If valid, keep.
    # - If invalid, try to add a block that fixes the most uncovered (greedy repair) or random?
    # - Or standard SA with high temperature initially.
    
    # Given the tight time, let's do a "Greedy Deletion" loop first, which is very effective.
    # Then use SA to escape local optima.
    
    # Pass 1: Greedy Deletion (Remove any block that doesn't break coverage)
    # We can repeat this until no more can be removed.
    # To avoid getting stuck in a specific local minimum, we can randomize the order of deletion attempts.
    
    rng = random.Random(42)
    
    # Helper to check if a block is redundant
    # A block is redundant if all its tsubs are covered by other blocks.
    # i.e., for all tid in block_tsubs, cover_count[tid] > 1.
    
    def is_redundant(b_idx):
        # b_idx is index in all_blocks? No, we have current_blocks as list of tuples.
        # We need to map current block to its tsub ids.
        pass
        
    # Let's maintain current_blocks as a list of indices into all_blocks?
    # That makes lookup faster.
    # current_indices = []
    # current_blocks = [] (tuples)
    
    # Let's refactor current_blocks to store indices.
    # Greedy phase added tuples. Let's map them back to indices.
    # block_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
    block_to_idx = {}
    for i, b in enumerate(all_blocks):
        block_to_idx[frozenset(b)] = i
        
    current_indices = [block_to_idx[frozenset(b)] for b in current_blocks]
    # Update cover_count is already correct.
    
    # Greedy Deletion Phase
    # Iterate through indices in random order.
    # If block is redundant, remove it.
    # Repeat until no change or time out.
    
    while time.time() < deadline:
        changed = False
        rng.shuffle(current_indices)
        for i, idx in enumerate(current_indices):
            if time.time() > deadline: break
            tids = block_tsubs[idx]
            redundant = True
            for tid in tids:
                if cover_count[tid] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove
                # Update counts
                for tid in tids:
                    cover_count[tid] -= 1
                # Remove from list
                current_indices[i] = None # Mark as removed
                changed = True
        # Clean up list
        if changed:
            current_indices = [i for i in current_indices if i is not None]
        else:
            break
            
    # Now use SA to improve further.
    # We want to minimize len(current_indices).
    # Moves: Remove, Add, Swap?
    # Swap: Remove one, Add one.
    
    T = 1.0
    T_min = 1e-5
    cooling_rate = 0.999 # Slower cooling for more steps?
    # Or adaptive cooling.
    
    # Energy = len(current_indices) + lambda * num_uncovered
    # But we want to stay valid.
    # If we allow invalid states, we need strong penalty.
    # Let's stick to valid states only (Local Search).
    # Move: 
    # 1. Try to remove a block. If valid, accept.
    # 2. Try to swap a block with another. 
    #    - Remove A. If invalid, we must add B such that coverage is restored?
    #    - Or: Remove A, Add B. Check if new state is valid.
    
    # Since we are at a local minimum for deletion (no single block can be removed),
    # we need to try swaps.
    # Swap A with B:
    # 1. Temporarily remove A.
    # 2. Check which tsubs become uncovered (count 0).
    # 3. Find B that covers all these uncovered tsubs (and maybe more).
    # 4. If such B exists and B is not A, accept swap.
    
    # This is a "2-opt" style move.
    
    while time.time() < deadline:
        improved = False
        # Try swaps for a random subset of blocks? Or all?
        # All is 50k * 50k = 2.5B. Too slow.
        # Sample?
        
        # Let's try random removals and repairs.
        # Pick random block A to remove.
        # If valid, we would have caught it in greedy deletion. So it's invalid.
        # Find the set of tsubs that become uncovered: U.
        # We need to add a block B that covers U.
        # If such B exists, swap A->B.
        
        # To find B, we can check all blocks? 50k checks.
        # Check: does B cover all tsubs in U?
        # U size is small (usually 1-5).
        
        # Let's iterate a random sample of candidate A's.
        if not current_indices: break
        
        sample_size = min(1000, len(current_indices))
        rng.shuffle(current_indices)
        
        for i in range(sample_size):
            idx_a = current_indices[i]
            tids_a = block_tsubs[idx_a]
            
            # Check if removing A makes any tsub uncovered
            uncovered_by_a = []
            for tid in tids_a:
                if cover_count[tid] == 1:
                    uncovered_by_a.append(tid)
            
            if not uncovered_by_a:
                # This shouldn't happen if greedy deletion converged, but just in case.
                for tid in tids_a:
                    cover_count[tid] -= 1
                current_indices[i] = None
                improved = True
                continue
                
            # We need a block B that covers all tids in uncovered_by_a.
            # And B != A.
            # And B is not already in the set? (If B is in the set, we can't add it, but we can swap? 
            # No, if B is in the set, it's already there. 
            # Wait, if B is in the set, it already covers its tsubs. 
            # If we remove A, and B is present, does B cover U? 
            # If B covers U, then A was redundant? 
            # No, A was the *only* one covering some tids in U. 
            # If B is present and covers U, then those tids would have count >= 2 (A and B). 
            # So count would not be 1. 
            # Therefore, if count is 1, no other block in the set covers that tid.
            # So B must be a block *not* in the current set.
            
            # Search for B
            found_b = -1
            # We can filter blocks by checking if they contain the first tid in U.
            # Precompute: for each tid, list of blocks containing it.
            # This is expensive to build (3876 * ~2500 blocks).
            # 3876 * 2500 = 9.7M entries. Memory ~ 80MB. OK.
            # But we already have block_masks.
            
            # Let's just iterate all blocks and check mask intersection.
            # mask_B & mask_U == mask_U ?
            # mask_U = 0
            # for tid in uncovered_by_a: mask_U |= (1 << tid)
            
            mask_U = 0
            for tid in uncovered_by_a:
                mask_U |= (1 << tid)
                
            # Iterate all blocks
            # To speed up, we can skip blocks that are in current set.
            # current_set = set(current_indices)
            
            # Optimization: 
            # If we precomputed `blocks_by_tid` (list of block indices for each tid),
            # we can intersect the lists for tids in U.
            
            # Let's build `blocks_by_tid` now.
            # It's a list of lists.
            # 3876 lists. Average length 50388 * 35 / 3876 ≈ 456.
            
            if not hasattr(construct, '_blocks_by_tid'):
                construct._blocks_by_tid = [[] for _ in range(num_tsubs)]
                for i, tids in enumerate(block_tsubs):
                    for tid in tids:
                        construct._blocks_by_tid[tid].append(i)
                        
            # Get candidates from the first tid in U
            if uncovered_by_a:
                candidates = construct._blocks_by_tid[uncovered_by_a[0]]
            else:
                candidates = []
                
            # Filter candidates
            for idx_b in candidates:
                if idx_b == idx_a: continue
                if idx_b in current_indices: continue # Must be outside set
                
                # Check if idx_b covers all U
                # mask_b = block_masks[idx_b]
                # if (mask_b & mask_U) == mask_U:
                #    found_b = idx_b
                #    break
                
                # Using mask is fast.
                if (block_masks[idx_b] & mask_U) == mask_U:
                    found_b = idx_b
                    break
                    
            if found_b != -1:
                # Perform Swap: Remove A, Add B
                # Update counts for A
                for tid in tids_a:
                    cover_count[tid] -= 1
                # Update counts for B
                tids_b = block_tsubs[found_b]
                for tid in tids_b:
                    cover_count[tid] += 1
                    
                # Update list
                current_indices[i] = found_b
                improved = True
                
                # We can continue looking for more swaps in this pass?
                # Or restart?
                # Restarting is safer to avoid interference.
                break
                
        if not improved:
            # No swap found in this sample.
            # Try a different sample? Or give up?
            # Let's try a few more samples.
            pass
            
    # Final cleanup
    # Remove any None if we did in-place updates? 
    # We did `current_indices[i] = found_b` or `None`.
    # In the swap loop, we break after first improvement.
    # So we might have multiple None if we did multiple passes?
    # No, we break the inner loop, but the outer `while time` continues.
    # If we break the inner loop with `improved=True`, we go back to `while time`.
    # We should clean up Nones at the end of each outer iteration?
    # Or just filter at the very end.
    
    # Let's filter Nones now.
    current_indices = [i for i in current_indices if i is not None]
    
    # Final Greedy Deletion to clean up any redundancy created by swaps
    while time.time() < deadline:
        changed = False
        rng.shuffle(current_indices)
        for i, idx in enumerate(current_indices):
            if time.time() > deadline: break
            tids = block_tsubs[idx]
            redundant = True
            for tid in tids:
                if cover_count[tid] == 1:
                    redundant = False
                    break
            if redundant:
                for tid in tids:
                    cover_count[tid] -= 1
                current_indices[i] = None
                changed = True
        if changed:
            current_indices = [i for i in current_indices if i is not None]
        else:
            break

    # Convert indices to blocks
    final_blocks = [list(all_blocks[i]) for i in current_indices]
    
    return final_blocks
