import itertools
import random
import time
import numpy as np

# Precomputation
V = 19
K = 8
T = 5

# Map 5-subsets to indices
all_5_subs = list(itertools.combinations(range(V), T))
sub_index = {s: i for i, s in enumerate(all_5_subs)}
N_SUBS = len(all_5_subs) # 11628

# Precompute coverage for all 8-blocks
# block_id -> list of sub_indices
all_8_blocks = list(itertools.combinations(range(V), K))
block_coverage = []
for b in all_8_blocks:
    cov = [sub_index[s] for s in itertools.combinations(b, T)]
    block_coverage.append(cov)

# Map from 5-subset to list of block indices containing it
# sub_idx -> list of block_idxs
sub_to_blocks = [[] for _ in range(N_SUBS)]
for b_idx, b in enumerate(all_8_blocks):
    for s_idx in block_coverage[b_idx]:
        sub_to_blocks[s_idx].append(b_idx)

def calc_variance(counts):
    # counts is a list or array of integers
    # Variance = E[X^2] - (E[X])^2
    # For speed, we can use numpy if available, but standard lib is safer for small lists?
    # N_SUBS is 11628, so numpy is much faster for repeated calls.
    c = np.array(counts, dtype=np.float64)
    mean = c.mean()
    var = np.mean((c - mean) ** 2)
    return var

def spectral_polish(blocks, deadline):
    """
    Attempt to reduce the variance of the coverage counts.
    A more uniform coverage often allows for better future pruning or indicates a denser packing.
    """
    if not blocks:
        return blocks
    
    # Initialize counts
    counts = [0] * N_SUBS
    for b_idx in blocks:
        for s_idx in block_coverage[b_idx]:
            counts[s_idx] += 1
    
    current_var = calc_variance(counts)
    
    # We perform multiple passes. In each pass, we try to improve variance.
    # If we improve variance, we reset the pass counter (or just keep going until time limit).
    # Since we want to keep the number of blocks the same (or reduce it if variance reduction allows redundancy),
    # we only swap blocks.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(blocks)
        
        for i, b_idx in enumerate(blocks[:]):
            if time.time() > deadline - 5.0:
                break
                
            # Find unique subsets of b_idx (count == 1)
            unique_subs = []
            for s_idx in block_coverage[b_idx]:
                if counts[s_idx] == 1:
                    unique_subs.append(s_idx)
            
            # If there are no unique subsets, this block is redundant (or was handled before).
            # If there are unique subsets, any replacement MUST cover them.
            # If unique_subs is empty, we can swap with ANY block that reduces variance.
            
            # Determine candidates
            if unique_subs:
                # Intersection of blocks covering all unique subsets
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(sub_to_blocks[s_idx])
                    else:
                        candidates &= set(sub_to_blocks[s_idx])
                
                if not candidates:
                    continue
                candidates.discard(b_idx)
                if not candidates:
                    continue
            else:
                # No unique subsets, so we can swap with any other block
                # To limit search space, we might restrict to blocks that are "far" or random.
                # For now, let's sample a few random blocks from the universe?
                # Or just try to swap with a block that has low coverage overlap?
                # Actually, if no unique subsets, the block is redundant.
                # The previous local_search should have removed it.
                # If it wasn't removed, maybe it became redundant after a swap?
                # In that case, we can just remove it?
                # But spectral_polish is for swapping. Let's stick to swaps to maintain block count stability during this phase.
                # If it's truly redundant, the next pass of local_search (or a final prune) will remove it.
                # To be safe, if no unique subs, we can try to swap with a random block to break symmetry.
                # But let's just skip if no unique subs to avoid unnecessary computation, 
                # assuming the previous local_search did its job.
                # However, if we *want* to explore, we can pick a random candidate.
                # Let's just skip for now to focus on "tight" blocks.
                continue

            # Get list of candidates
            cand_list = list(candidates)
            
            # Evaluate best candidate based on variance reduction
            best_candidate = -1
            best_var = current_var
            
            # To speed up, we can sample if the list is too large
            if len(cand_list) > 50:
                cand_list = random.sample(cand_list, 50)
            
            for new_b_idx in cand_list:
                # Calculate new variance efficiently?
                # We need to update counts for b_idx and new_b_idx.
                # This is O(56) per candidate.
                # We can do a "dry run" update on a copy or just calculate the delta.
                
                # Delta in counts:
                # Subtract b_idx, Add new_b_idx
                # We can calculate the new variance without full array update if we track sums?
                # Var = E[X^2] - E[X]^2
                # SumSq changes by: -SumSq(b) + SumSq(new)
                # Sum changes by: -Sum(b) + Sum(new)
                # But X is the count vector.
                # Let S1 = sum(counts), S2 = sum(counts^2)
                # Removing b:
                #   S1 -= sum(b_counts)  (which is 56, constant for all blocks? No, counts are 0/1 in coverage, but we are removing 1 from specific indices)
                #   Actually, for each s in b, counts[s] decreases by 1.
                #   Delta S1 = -56
                #   Delta S2 = sum_{s in b} (counts[s] - 1)^2 - counts[s]^2 = sum_{s in b} (counts[s]^2 - 2counts[s] + 1 - counts[s]^2) = sum_{s in b} (1 - 2counts[s])
                # Adding new_b:
                #   Delta S1 += 56
                #   Delta S2 += sum_{s in new_b} (counts[s] + 1)^2 - counts[s]^2 = sum_{s in new_b} (1 + 2counts[s])
                
                # So Net Delta S1 = 0.
                # Net Delta S2 = sum_{s in new_b} (1 + 2counts[s]) - sum_{s in b} (1 - 2counts[s])
                #              = 56 + 2*sum_{s in new_b} counts[s] - (56 - 2*sum_{s in b} counts[s])
                #              = 2 * (sum_{s in new_b} counts[s] + sum_{s in b} counts[s])
                
                # Wait, this is for the *change* in S2.
                # New S2 = Old S2 + Delta S2.
                # New Var = (New S2 / N) - (S1 / N)^2. Since S1 is constant, minimizing Var is equivalent to minimizing S2.
                
                # So we just need to minimize S2, which is equivalent to minimizing:
                # sum_{s in new_b} counts[s] + sum_{s in b} counts[s]
                
                # Let's verify.
                # S2 = sum c_i^2
                # c_i' = c_i - 1_{i in b} + 1_{i in new_b}
                # c_i'^2 = c_i^2 - 2c_i 1_{i in b} + 1_{i in b} + 2c_i 1_{i in new_b} + 1_{i in new_b}
                # Sum c_i'^2 = S2 - 2 sum_{i in b} c_i + 56 + 2 sum_{i in new_b} c_i + 56
                # Delta S2 = 112 + 2 sum_{i in new_b} c_i - 2 sum_{i in b} c_i
                # We want to minimize Delta S2, so minimize (sum_{i in new_b} c_i - sum_{i in b} c_i).
                
                score_new = 0
                for s_idx in block_coverage[new_b_idx]:
                    score_new += counts[s_idx]
                
                score_old = 0
                for s_idx in block_coverage[b_idx]:
                    score_old += counts[s_idx]
                    
                delta_s2 = 112 + 2 * (score_new - score_old)
                
                # We want delta_s2 < 0
                if delta_s2 < 0:
                    # This swap reduces variance
                    # We pick the one with most negative delta_s2
                    if best_candidate == -1 or delta_s2 < best_var: # Reusing best_var as best_delta
                        best_candidate = new_b_idx
                        best_var = delta_s2
            
            if best_candidate != -1:
                # Perform swap
                # Update counts
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
                for s_idx in block_coverage[best_candidate]:
                    counts[s_idx] += 1
                
                # Update blocks list
                blocks[i] = best_candidate
                improved = True
                # Update current_var? Not strictly necessary for next iteration's delta calc,
                # as we recalculate score_old/new from counts.
                # But we should update current_var if we use it for anything else.
                # We don't use current_var in the inner loop anymore, only delta_s2.
                # So we can skip recalculating full variance.
                
                # Early exit if we found a huge improvement? No, keep going.
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Greedy Construction
    def greedy_construct():
        uncovered = [True] * N_SUBS
        blocks = []
        # Randomize order of subsets to break ties? 
        # Standard greedy: pick first uncovered, pick best block containing it.
        while True:
            # Find an uncovered subset
            try:
                # To speed up, we can just iterate, but checking all is O(N_SUBS)
                # N_SUBS is 11628, so it's fine.
                # Optimization: keep a list of uncovered indices?
                # Simpler: just find the first True
                target_idx = -1
                for i in range(N_SUBS):
                    if uncovered[i]:
                        target_idx = i
                        break
                if target_idx == -1:
                    break
            except IndexError:
                break
            
            # Get all blocks covering this subset
            candidates = sub_to_blocks[target_idx]
            
            # Pick the block that covers the most uncovered subsets
            best_block = -1
            max_gain = -1
            for b_idx in candidates:
                gain = 0
                for s_idx in block_coverage[b_idx]:
                    if uncovered[s_idx]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
            
            # Add the block
            blocks.append(best_block)
            for s_idx in block_coverage[best_block]:
                uncovered[s_idx] = False
        
        return blocks

    # Initial Solution
    current_blocks = greedy_construct()
    
    # Local Search
    def count_coverage(blocks):
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        return counts

    def is_redundant(b_idx, counts):
        for s_idx in block_coverage[b_idx]:
            if counts[s_idx] <= 1:
                return False
        return True

    def local_search(blocks):
        # Convert to set for faster membership? 
        # We need to track counts dynamically.
        counts = count_coverage(blocks)
        block_set = set(blocks)
        
        improved = True
        while improved:
            improved = False
            # Pass 1: Remove redundant blocks
            # Iterate in random order to avoid bias
            random.shuffle(blocks)
            for b_idx in blocks[:]: # iterate over a copy
                if is_redundant(b_idx, counts):
                    # Remove
                    for s_idx in block_coverage[b_idx]:
                        counts[s_idx] -= 1
                    blocks.remove(b_idx)
                    block_set.remove(b_idx)
                    improved = True
            
            # Pass 2: Try to replace a block with a better one
            # For each block, find its unique subsets (count == 1)
            # If a block has no unique subsets, it's redundant (handled above)
            # So for remaining blocks, they have at least one unique subset.
            # We try to find a replacement block B' such that B' covers all unique subsets of B.
            # If such B' exists and B' != B, we can swap B with B'.
            # This might make other blocks redundant.
            
            # Optimization: Only try this if time allows
            if time.time() > deadline - 10:
                break

            for b_idx in blocks[:]:
                # Find unique subsets
                unique_subs = []
                for s_idx in block_coverage[b_idx]:
                    if counts[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    continue
                
                # Find a block B' that covers all unique_subs
                # Candidates must contain all unique_subs
                # Intersection of sub_to_blocks for each unique_sub
                # Start with the smallest list
                candidates = None
                for s_idx in unique_subs:
                    if candidates is None:
                        candidates = set(sub_to_blocks[s_idx])
                    else:
                        candidates &= set(sub_to_blocks[s_idx])
                
                if not candidates:
                    continue
                
                # Exclude current block
                candidates.discard(b_idx)
                
                if not candidates:
                    continue
                
                # Pick one candidate (random or best)
                # "Best" might be one that covers more *other* subsets to help future removals?
                # For now, just pick one randomly or the first one.
                # Let's try the one that covers the most total subsets (not just unique)
                # Actually, any valid replacement is a step towards potentially removing others.
                # Let's just pick one.
                new_b_idx = next(iter(candidates))
                
                # Perform Swap
                # Remove b_idx
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
                # Add new_b_idx
                for s_idx in block_coverage[new_b_idx]:
                    counts[s_idx] += 1
                
                blocks.remove(b_idx)
                blocks.append(new_b_idx)
                block_set.discard(b_idx)
                block_set.add(new_b_idx)
                improved = True
                break # Restart loop after swap to re-evaluate redundancies
            
            if time.time() > deadline - 5:
                break

        return blocks

    # Run local search
    current_blocks = local_search(current_blocks)
    
    # Spectral Polish: Reduce variance to enable further improvements or tighten the solution
    if time.time() < deadline - 10:
        current_blocks = spectral_polish(current_blocks, deadline)
        
    # Final Prune: After spectral polish, some blocks might have become redundant
    # Run a quick redundancy check and removal
    def final_prune(blocks):
        if not blocks:
            return blocks
        counts = [0] * N_SUBS
        for b_idx in blocks:
            for s_idx in block_coverage[b_idx]:
                counts[s_idx] += 1
        
        # Remove redundant blocks
        new_blocks = []
        for b_idx in blocks:
            # Check if block is redundant
            # A block is redundant if all its subsets are covered by at least one OTHER block.
            # This means counts[s_idx] > 1 for all s_idx in the block.
            # If any counts[s_idx] <= 1, the block is essential (not redundant).
            is_redundant = True
            for s_idx in block_coverage[b_idx]:
                if counts[s_idx] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Block is redundant, so we do NOT add it to new_blocks.
                # We must update counts to reflect its removal for subsequent checks.
                for s_idx in block_coverage[b_idx]:
                    counts[s_idx] -= 1
            else:
                # Block is essential, so we keep it.
                new_blocks.append(b_idx)
                
        return new_blocks

    current_blocks = final_prune(current_blocks)
    
    # Convert indices back to blocks
    # The evaluator expects list of lists of ints
    result = [list(all_8_blocks[b_idx]) for b_idx in current_blocks]
    
    return result
