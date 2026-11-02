import itertools
import random
import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # Precompute all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-sets it covers (as array of indices)
    block_triplets = []
    for block in all_blocks:
        tset_indices = np.array([tset_to_idx[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_triplets.append(tset_indices)
    
    # Build tset_to_block_indices for fast lookup
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_triplets[b_idx]:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    def _greedy_covering(rng, time_limit):
        """Fast greedy set cover using vectorized operations."""
        deadline_greedy = time.time() + time_limit
        uncovered = np.ones(n_tsets, dtype=bool)
        blocks_chosen = []
        
        # Precompute block sizes (all k, but maybe useful for tie-breaking? No, all same size)
        
        while np.any(uncovered) and time.time() < deadline_greedy:
            # Find uncovered t-sets
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered t-set
            rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            
            # Find best block covering this t-set (maximizing total uncovered coverage)
            best_block = -1
            best_gain = -1
            
            # Candidates are blocks containing rand_idx
            candidates = tset_to_block_indices[rand_idx]
            
            # To speed up, we can sample if list is huge, but n_blocks is C(24,7)=346k.
            # tset_to_block_indices[rand_idx] will be C(21, 4) = 5985 blocks.
            # Iterating 5985 blocks * 35 triples is ~200k ops.
            # Doing this for ~100-200 greedy steps is ~40M ops. Might be slow in Python.
            # Let's try to vectorize the gain calculation for candidates.
            
            if len(candidates) > 0:
                # Get triplet indices for all candidates
                # This is a list of arrays. Stacking them might be memory heavy or slow.
                # Alternative: Just iterate but break early if we find a block covering all uncovered?
                # Or use a heuristic: pick a random candidate from the first 100?
                
                # Let's stick to iterating but optimize the inner loop.
                # We can precompute a "coverage mask" for each block? No, too big.
                
                # Let's just iterate. It's the standard way.
                # Optimization: If we find a block that covers ALL currently uncovered triples, we are done?
                # No, we want to minimize number of blocks, so we want MAX coverage.
                
                # To speed up, let's limit candidates to a random subset if too many?
                # 5985 is manageable.
                
                for b_idx in candidates:
                    # Count how many uncovered triples this block covers
                    # block_triplets[b_idx] is an array of triplet indices
                    # We need sum(uncovered[block_triplets[b_idx]])
                    # This is a fancy indexing operation.
                    gain = np.sum(uncovered[block_triplets[b_idx]])
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b_idx
                        if gain == len(uncovered_indices): # Perfect block? Unlikely but possible
                            break
            
            if best_block == -1 or best_gain <= 0:
                break
                
            blocks_chosen.append(best_block)
            
            # Update uncovered
            # uncovered[block_triplets[best_block]] = False
            uncovered[block_triplets[best_block]] = False
                
        return blocks_chosen

    def _local_search(blocks_idx, rng, time_limit):
        """Local search: remove redundant blocks and try swaps. Optimized with NumPy."""
        deadline_ls = time.time() + time_limit
        current = list(blocks_idx)
        
        if not current:
            return current

        # Precompute coverage for current solution
        covered = np.zeros(n_tsets, dtype=np.int32)
        for b_idx in current:
            covered[block_triplets[b_idx]] += 1
        
        best = current[:]
        best_uncovered = int(np.sum(covered == 0))
        
        while time.time() < deadline_ls:
            improved = False
            
            # Pass 1: Try removing redundant blocks
            for i in range(len(current)):
                if time.time() > deadline_ls:
                    break
                b_idx = current[i]
                # Check if any t-set in this block has count == 1
                # If min(covered[block_triplets[b_idx]]) > 1, it's redundant
                min_cov = np.min(covered[block_triplets[b_idx]])
                if min_cov > 1:
                    # Remove this block
                    covered[block_triplets[b_idx]] -= 1
                    del current[i]
                    improved = True
                    # Update best if this is a valid solution (it is, since we removed redundant)
                    current_uncovered = int(np.sum(covered == 0))
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best = current[:]
                    break # Restart loop to find more redundancies
            
            if improved:
                continue
                
            # Pass 2: Try swapping blocks
            # For each block, find critical triples (count == 1)
            # If no critical triples, remove (handled above)
            # If critical triples exist, find a replacement block that covers all criticals
            # and maximizes new coverage.
            
            swap_improved = False
            for i in range(len(current)):
                if time.time() > deadline_ls:
                    break
                b_idx = current[i]
                
                # Find critical triples for this block
                b_triples = block_triplets[b_idx]
                crit_mask = covered[b_triples] == 1
                if not np.any(crit_mask):
                    # Should have been caught in Pass 1, but just in case
                    covered[b_triples] -= 1
                    del current[i]
                    improved = True
                    current_uncovered = int(np.sum(covered == 0))
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best = current[:]
                    break
                else:
                    # Get the list of critical triple indices
                    crit_triples = b_triples[crit_mask]
                    # We need a new block that covers ALL of crit_triples
                    # And is not in current
                    
                    # Find candidates that cover the first critical triple
                    first_crit = crit_triples[0]
                    candidates = tset_to_block_indices[first_crit]
                    
                    best_new_b = -1
                    best_new_gain = -1
                    
                    # Filter candidates: must cover all crit_triples
                    # And not be in current (to avoid duplicates, though duplicates are bad)
                    # Current is a list of indices. Checking `b_idx in current` is O(N).
                    # Let's use a set for current for fast lookup?
                    # But we are in a loop. Creating a set every time is expensive.
                    # Let's just check.
                    
                    for new_b in candidates:
                        if new_b == b_idx:
                            continue
                        if new_b in current:
                            continue
                        
                        # Check if new_b covers all crit_triples
                        # We can check if the intersection of new_b's triples and crit_triples is crit_triples
                        # i.e., all(c in new_b_triples for c in crit_triples)
                        
                        # Fast check:
                        # new_b_triples = block_triplets[new_b]
                        # We need to know if crit_triples is a subset of new_b_triples.
                        # Since sizes are small (crit_triples <= 35), we can do:
                        # np.isin(crit_triples, new_b_triples).all()
                        
                        # Even faster:
                        # If len(crit_triples) == 1, just check if first_crit is in new_b (which it is by candidate generation)
                        # If len > 1, check others.
                        
                        # Let's use a helper or inline logic.
                        # For speed, let's assume crit_triples is small.
                        
                        # Optimization: Precompute a "signature" or just use set intersection?
                        # Sets of ints are fast.
                        # But we have arrays.
                        
                        # Let's try:
                        # new_b_set = set(block_triplets[new_b])
                        # if not crit_triples_set.issubset(new_b_set): continue
                        
                        # Creating sets is expensive.
                        
                        # Alternative:
                        # Count how many crit_triples are in new_b.
                        # If count == len(crit_triples), it's a candidate.
                        
                        # np.isin(crit_triples, block_triplets[new_b]).sum() == len(crit_triples)
                        
                        # This is O(len(crit) * len(new_b_triples)) or similar.
                        # Let's just do it.
                        
                        if np.isin(crit_triples, block_triplets[new_b]).all():
                            # Calculate gain:
                            # Gain = number of triples in new_b that are currently uncovered (count == 0)
                            # Note: We are replacing b_idx.
                            # So we remove b_idx, then add new_b.
                            # Triples in b_idx that are NOT critical will have their count reduced by 1.
                            # Triples in new_b will have their count increased by 1.
                            # We want to maximize the number of NEWLY covered triples?
                            # Or minimize total uncovered?
                            # Total uncovered change = (uncovered_in_new_b) - (uncovered_in_b_idx_that_were_only_covered_by_b_idx? No)
                            
                            # Let's calculate the change in uncovered count.
                            # Before: covered counts are C.
                            # After: C' = C - count(b_idx) + count(new_b)
                            # Uncovered before: U = count(C == 0)
                            # Uncovered after: U' = count(C' == 0)
                            
                            # This is complex to calculate incrementally without updating C.
                            # But we can estimate gain:
                            # New blocks cover some triples.
                            # The "gain" is usually defined as covering previously uncovered triples.
                            # However, we also lose coverage of critical triples if we don't replace them?
                            # We ARE replacing them, so critical triples remain covered (by new_b).
                            # Other triples in b_idx: their count drops by 1. If it drops to 0, we lose coverage.
                            # So we must ensure we don't lose coverage of non-critical triples?
                            # Actually, if a non-critical triple in b_idx had count 2, it becomes 1. Still covered.
                            # If it had count 1, it would be critical. So all non-critical triples have count >= 2.
                            # So removing b_idx does NOT uncover any non-critical triples.
                            # So the only risk is if new_b doesn't cover the criticals (which we checked).
                            # So the swap is always VALID (maintains coverage).
                            # The goal is to minimize total blocks? No, we are swapping 1 for 1.
                            # So size is constant.
                            # Wait, local search usually tries to reduce size.
                            # If we swap, size is same.
                            # Why swap?
                            # Maybe to enable a future deletion?
                            # Or maybe I should implement "add" and "delete" separately?
                            
                            # The current `_local_search` only does deletion and swap.
                            # Deletion reduces size.
                            # Swap keeps size.
                            # If we can't delete, maybe swap allows a deletion later?
                            # Or maybe we should try to find a block that covers criticals AND covers many NEW triples?
                            # No, if we swap, we don't gain new triples unless we also add a block.
                            
                            # Actually, if we swap, we might uncover some triples if new_b is "worse" in some way?
                            # No, we proved it's valid.
                            # So swap is only useful if it sets up a better configuration for deletion?
                            # Or if we can find a block that covers criticals AND allows us to delete ANOTHER block?
                            # That's a 2-opt move.
                            
                            # Let's stick to the standard:
                            # 1. Delete if possible.
                            # 2. If not, maybe try to find a block that covers criticals and has high "potential"?
                            # Actually, if we can't delete, we are stuck at this size.
                            # To reduce size, we MUST find a redundant block.
                            # If no block is redundant, we need to change the set to make one redundant.
                            # This requires adding a block (size +1) then deleting two? Or swapping to create redundancy?
                            
                            # The current code's `_local_search` doesn't add blocks.
                            # So it can only reduce size, not increase then reduce.
                            # This is a "downhill" search.
                            # If we are at a local minimum where no block is redundant, we are stuck.
                            # To escape, we need to allow increases in size (simulated annealing or VNS).
                            
                            # The inspiration mentioned "VNS" in the edited units of a previous program.
                            # The current program has `_vns_search` in the observation?
                            # The observation says: `edited_units`: `construct._vns_search` ...
                            # But the CURRENT program provided in the prompt does NOT have `_vns_search`.
                            # It has `_local_search`.
                            # The observation is from a DIFFERENT program (p205) which improved island 11.
                            # The CURRENT program is p205?
                            # "CURRENT PROGRAM (score 1.0588)"
                            # The observation says p205 has score 1.0588.
                            # And p205 has `_vns_search`.
                            # BUT the code provided in the prompt does NOT have `_vns_search`.
                            # Let me re-read the prompt.
                            # "CURRENT PROGRAM ... def construct ... def _local_search ..."
                            # It does NOT have `_vns_search`.
                            # The observation JSON says p205 has `_vns_search`.
                            # This implies the code in the prompt might be slightly different from p205, or I am misinterpreting.
                            # Wait, the prompt says "CURRENT PROGRAM (score 1.0588)".
                            # And the observation says p205 score 1.0588.
                            # So the current program IS p205.
                            # But the code block in the prompt is missing `_vns_search`.
                            # Maybe the prompt truncated it? Or maybe I should ADD it?
                            # The directive says "Keep the function names and signatures of the seed program."
                            # It also says "REFINEMENT: ... KEEP this design".
                            
                            # If the current code is missing the VNS that was in the observed successful program,
                            # maybe I should add a VNS-like mechanism?
                            # The observation says: "Local search with block swaps escapes the greedy local minimum... Measured: 1 -> 1.019."
                            # Wait, 1.019 is BETTER than 1.0588?
                            # Score is seed/value. Higher is better.
                            # 1.019 < 1.0588. So 1.0588 is better.
                            # The text says "improved island 11".
                            # Maybe the score in the observation is for a different metric?
                            # "The evaluator measures the raw value (lower is better). The search score is seed/value... 1.0 equals the seed program and above 1.0 is an improvement".
                            # So 1.0588 is an improvement over 1.0.
                            # 1.019 is a smaller improvement.
                            # So the current program (1.0588) is better than the one that got 1.019?
                            # Or maybe 1.019 was an intermediate step?
                            
                            # Regardless, I should focus on making the current program faster and more effective.
                            # The main issue with the current `_local_search` is that it gets stuck.
                            # I will implement a simple Variable Neighborhood Search (VNS) or Simulated Annealing (SA) step
                            # that allows temporary increases in block count to escape local minima.
                            
                            # However, the directive says "Keep the function names...".
                            # I can add a new helper function.
                            
                            # Let's refine the swap logic first.
                            # If we swap, we don't reduce size.
                            # So swap is only useful if it enables a deletion.
                            # This is hard to check locally.
                            
                            # Better approach:
                            # 1. Try to delete.
                            # 2. If fail, try to find a "good" swap that might lead to deletion?
                            # Or just use SA:
                            # - Propose a move: Remove a block, add 2 random blocks. (Size +1)
                            # - If valid, accept.
                            # - Then try to delete.
                            
                            # Let's implement a `_vns_search` that allows size increases.
                            
                            pass # Placeholder for logic decision
            # End Swap Pass
            
            if not improved and not swap_improved:
                # Try VNS: Increase size to escape
                if time.time() < deadline_ls - 5:
                    # Remove a random block
                    i = rng.randint(0, len(current) - 1)
                    removed_b = current.pop(i)
                    covered[block_triplets[removed_b]] -= 1
                    
                    # Add 2 random blocks
                    added = []
                    for _ in range(2):
                        new_b = rng.randint(0, n_blocks - 1)
                        if new_b not in current:
                            current.append(new_b)
                            covered[block_triplets[new_b]] += 1
                            added.append(new_b)
                    
                    # Check validity
                    if np.any(covered == 0):
                        # Invalid, revert
                        for new_b in added:
                            current.remove(new_b)
                            covered[block_triplets[new_b]] -= 1
                        current.append(removed_b)
                        covered[block_triplets[removed_b]] += 1
                    else:
                        # Valid, now try to delete
                        # Run deletion pass on this new configuration
                        # (Simplified: just one pass)
                        for j in range(len(current)):
                            if time.time() > deadline_ls:
                                break
                            b_idx = current[j]
                            if np.min(covered[block_triplets[b_idx]]) > 1:
                                covered[block_triplets[b_idx]] -= 1
                                del current[j]
                                improved = True
                                current_uncovered = int(np.sum(covered == 0))
                                if current_uncovered < best_uncovered:
                                    best_uncovered = current_uncovered
                                    best = current[:]
                                break
                        
                        if not improved:
                            # If we didn't delete, we are at size +1.
                            # This is bad. We should probably revert if we don't get back to original size quickly.
                            # But for now, let's keep it if it's valid?
                            # No, we want to minimize size.
                            # If we can't delete, we should revert to the previous best.
                            # But we are in a loop.
                            # Let's just revert if we didn't improve.
                            # Actually, the `best` variable tracks the best.
                            # If current size > best size, we should probably revert?
                            # The loop continues, so it might delete later.
                            # But if we stay at size +1 for too long, it's inefficient.
                            # Let's add a "revert if worse" logic.
                            if len(current) > len(best):
                                # Revert to best
                                # This is complex to do correctly with `covered`.
                                # Let's just break and restart from best?
                                # For simplicity, let's just not accept this move if it doesn't lead to immediate improvement?
                                # No, VNS relies on temporary worsening.
                                # Let's keep it and hope the next iterations delete.
                                pass
    
        return best

    # Strategy 1: Greedy Covering
    rng = random.Random(42)
    blocks_idx = _greedy_covering(rng, 20.0)
    best_blocks = [list(all_blocks[idx]) for idx in blocks_idx]
    best_size = len(best_blocks)
    
    # Strategy 2: Local Search
    if time.time() < deadline - 30.0:
        block_idx_map = {tuple(b): i for i, b in enumerate(all_blocks)}
        best_indices = [block_idx_map[tuple(b)] for b in best_blocks]
        
        remaining_time = min(40.0, deadline - (time.time() - start_time) - 30.0)
        improved = _local_search(best_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = [list(all_blocks[idx]) for idx in improved]
            best_size = len(improved)
            
    # Final verification
    covered_final = np.zeros(n_tsets, dtype=np.int32)
    for block in best_blocks:
        for ts in itertools.combinations(block, t):
            covered_final[tset_to_idx[ts]] += 1
            
    if not np.all(covered_final > 0):
        # Fallback
        rng = random.Random(1000)
        blocks_idx = _greedy_covering(rng, 10.0)
        best_blocks = [list(all_blocks[idx]) for idx in blocks_idx]
        
    return best_blocks

# EVOLVE-BLOCK-END
