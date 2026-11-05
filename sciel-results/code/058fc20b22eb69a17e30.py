import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage structure
    # block_covered_indices[i] = list of t-subset indices covered by block i
    block_covered_indices = []
    # subset_to_blocks[j] = list of block indices that cover t-subset j
    subset_to_blocks = [[] for _ in range(n_subsets)]
    
    # Frequency of each t-subset (how many blocks cover it)
    subset_freq = np.zeros(n_subsets, dtype=np.int32)
    
    for b_idx, b in enumerate(all_blocks):
        idxs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_covered_indices.append(idxs)
        for s_idx in idxs:
            subset_to_blocks[s_idx].append(b_idx)
            subset_freq[s_idx] += 1
    
    # Dual Weights: 1/frequency. Rare subsets get high weight.
    # Avoid division by zero (though freq should be > 0 for valid v,k,t)
    dual_weights = np.zeros(n_subsets, dtype=np.float64)
    for i in range(n_subsets):
        if subset_freq[i] > 0:
            dual_weights[i] = 1.0 / subset_freq[i]
        else:
            dual_weights[i] = 0.0

    # Build sparse-ish coverage matrix for vectorized gain calculation
    # Using a dense matrix is feasible for n_blocks * n_subsets ~ 50k * 3876 ~ 200M elements? 
    # 200M floats is ~1.6GB. Too big for memory. 
    # We must use a sparse representation or efficient iteration.
    # However, NumPy operations on lists are slow.
    # Let's use a different approach for the "Pricing" step:
    # Since n_blocks is ~50k and each block covers C(7,4)=35 subsets,
    # we can compute gains by iterating blocks and summing weights of uncovered subsets.
    # To speed this up, we maintain an array of "active" weights (0 if covered, weight if uncovered).
    
    def dual_greedy():
        """
        Greedy set cover using dual weights.
        At each step, pick the block that maximizes sum(dual_weights[s] for s in block if s is uncovered).
        """
        covered = np.zeros(n_subsets, dtype=bool)
        # active_weights[s] = dual_weights[s] if not covered[s] else 0
        active_weights = dual_weights.copy()
        
        chosen_blocks = []
        
        while True:
            if time.time() > deadline - 10:
                break
            
            # Check if all covered
            if not np.any(~covered):
                break
            
            # Find best block
            best_block = -1
            best_gain = 0.0
            
            # Iterate through all blocks to find max gain
            # Optimization: We can skip blocks that are already chosen? No, we don't reuse blocks in set cover.
            # But we don't have a fast way to skip.
            # 50k blocks * 35 subsets = 1.75M operations per step.
            # If we need ~100-150 blocks, that's ~250M operations. In Python, this is too slow.
            
            # Alternative: Use the coverage matrix if it fits in memory or use a smarter heuristic.
            # Let's try to vectorize the gain calculation using the precomputed indices.
            
            # We can compute gains for all blocks at once if we have a matrix.
            # Let's try to build a sparse matrix representation or use a trick.
            # Trick: Precompute for each block the list of indices.
            # We can use numpy to sum active_weights[block_covered_indices[i]] for all i.
            
            # This is still a loop in Python if we do it naively.
            # Let's use a "batch" approach or just accept the loop but optimize it.
            
            # Actually, for v=19, k=7, t=4:
            # n_blocks = C(19,7) = 50388
            # n_subsets = C(19,4) = 3876
            # A dense matrix of 50388 x 3876 is ~195 million elements.
            # As int8, that's 195 MB. This is acceptable!
            
            # We didn't build it in the global scope to save time/memory if not needed, 
            # but for this specific size, it's worth it for speed.
            
            # Let's build it once inside construct if not cached? 
            # Better: Build it now.
            
            # We'll use a global-like cache in the function scope for this run.
            # But building it takes time.
            
            # Let's stick to the loop but make it as fast as possible.
            # Or, we can use the fact that we only care about *uncovered* subsets.
            
            # Let's try a different strategy: 
            # 1. Start with random or standard greedy to get a rough solution.
            # 2. Use local search to improve.
            # The "Dual Greedy" is the inspiration. To make it fast, we need the matrix.
            
            # Let's build the matrix once.
            if not hasattr(dual_greedy, 'cov_matrix'):
                # Build coverage matrix
                # cov_matrix[i, j] = 1 if block i covers subset j
                # Using uint8
                dual_greedy.cov_matrix = np.zeros((n_blocks, n_subsets), dtype=np.uint8)
                for i, idxs in enumerate(block_covered_indices):
                    dual_greedy.cov_matrix[i, idxs] = 1
                # Store indices for fast update? No, matrix multiplication is fast.
            
            cov_matrix = dual_greedy.cov_matrix
            
            # Gains for all blocks: cov_matrix @ active_weights
            # cov_matrix is (n_blocks, n_subsets)
            # active_weights is (n_subsets,)
            # Result is (n_blocks,)
            raw_gains = cov_matrix @ active_weights
            
            # Hypergraph-coloring-inspired: normalize by sqrt of "degree" 
            # (number of uncovered subsets this block intersects) to prefer
            # blocks that concentrate gain on rare subsets rather than spreading
            # thin. degree = number of currently-uncovered subsets in the block.
            degrees = (cov_matrix @ (~covered).astype(np.float64))
            # Avoid division by zero; where degree==0, gain is also 0
            with np.errstate(divide='ignore', invalid='ignore'):
                gains = np.where(degrees > 0, raw_gains / np.sqrt(degrees), 0.0)
            
            best_block = int(np.argmax(gains))
            best_gain = gains[best_block]
            
            if best_gain <= 0:
                break
            
            chosen_blocks.append(best_block)
            
            # Update covered and active_weights
            # newly_covered = cov_matrix[best_block]
            # We need to set active_weights[s] = 0 for s in block
            # And covered[s] = 1
            
            # Vectorized update
            # Get indices of subsets covered by this block
            # We can use the precomputed list for speed in updating the boolean array
            idxs = block_covered_indices[best_block]
            covered[idxs] = True
            active_weights[idxs] = 0.0
            
            # Check termination
            if not np.any(~covered):
                break

        return chosen_blocks

    def local_search(blocks, max_time=10):
        """
        Local search:
        1. Remove redundant blocks.
        2. Try 2-for-1 swaps (replace 2 blocks with 1).
        3. Try 1-for-1 swaps (replace 1 block with 1 block that covers its unique parts and maybe more).
        """
        current_blocks = list(blocks)
        start_ls = time.time()
        
        # Maintain coverage count array
        # count_arr[s] = number of blocks covering subset s
        count_arr = np.zeros(n_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            idxs = block_covered_indices[b_idx]
            count_arr[idxs] += 1
        
        # Helper to check if a block is redundant
        # A block is redundant if for all s in block, count_arr[s] > 1
        # i.e., min(count_arr[idxs]) > 1
        
        improved = True
        while improved and time.time() < start_ls + max_time:
            improved = False
            n_blocks_cur = len(current_blocks)
            
            # 1. Remove redundant blocks
            # Iterate backwards to allow popping
            for i in range(n_blocks_cur - 1, -1, -1):
                if time.time() > start_ls + max_time:
                    break
                b_idx = current_blocks[i]
                idxs = block_covered_indices[b_idx]
                # Check if any subset in this block has count == 1
                # If min count > 1, it's redundant
                if np.min(count_arr[idxs]) > 1:
                    current_blocks.pop(i)
                    count_arr[idxs] -= 1
                    improved = True
                    break # Restart scan after removal
            
            # 2. Try 1-for-1 replacement
            # Pick a block i. If there exists a block k (not in current set)
            # such that every subset uniquely covered by i (count==1 and in i)
            # is also covered by k, then we can replace i with k.
            if not improved and len(current_blocks) > 1:
                n_trials_11 = min(15, len(current_blocks))
                for _ in range(n_trials_11):
                    if time.time() > start_ls + max_time:
                        break
                    i = random.randint(0, len(current_blocks) - 1)
                    b_i = current_blocks[i]
                    idxs_i = block_covered_indices[b_i]
                    
                    # Find subsets uniquely covered by b_i
                    unique_to_i = [s for s in idxs_i if count_arr[s] == 1]
                    
                    if not unique_to_i:
                        # b_i is redundant in a stronger sense; just remove it
                        current_blocks.pop(i)
                        count_arr[idxs_i] -= 1
                        improved = True
                        break
                    
                    # Find candidate blocks k covering all unique_to_i
                    cands = set(subset_to_blocks[unique_to_i[0]])
                    for s in unique_to_i[1:]:
                        cands &= set(subset_to_blocks[s])
                        if not cands:
                            break
                    
                    if cands:
                        # Remove b_i from candidates (we want a different block)
                        cands.discard(b_i)
                        if cands:
                            # Pick a candidate that maximizes additional coverage
                            best_k = -1
                            best_extra = -1
                            for k in list(cands)[:20]:  # Limit candidates for speed
                                extra = np.sum(count_arr[block_covered_indices[k]] == 0)
                                if extra > best_extra:
                                    best_extra = extra
                                    best_k = k
                            if best_k != -1:
                                current_blocks[i] = best_k
                                count_arr[idxs_i] -= 1
                                count_arr[block_covered_indices[best_k]] += 1
                                improved = True
                                break

            # 3. Try 2-for-1 replacement
            # Pick two blocks i, j. If there exists a block k such that 
            # (Cover(i) U Cover(j)) \ (Cover(i) ∩ Cover(j)) is covered by k, 
            # and k is not i or j, and we can remove i and j and add k.
            # More simply: 
            # Unique_i = {s in i | count[s]==1}
            # Unique_j = {s in j | count[s]==1}
            # Shared_ij = {s in i ∩ j | count[s]==2}
            # At_risk = Unique_i U Unique_j U Shared_ij
            # If there is a block k covering all At_risk, we can replace i,j with k.
            
            if not improved and len(current_blocks) > 2:
                # Sample a few pairs
                n_trials = min(20, len(current_blocks))
                for _ in range(n_trials):
                    if time.time() > start_ls + max_time:
                        break
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    idxs_i = block_covered_indices[b_i]
                    idxs_j = block_covered_indices[b_j]
                    
                    # Find at-risk subsets
                    # We need to check counts relative to current state
                    # count_arr reflects the state WITH i and j present.
                    
                    # Subsets in i with count 1 are unique to i (among all blocks)
                    # Subsets in j with count 1 are unique to j
                    # Subsets in both i and j with count 2 are unique to the pair (i,j)
                    
                    # Efficient check:
                    # We can't easily get "in both" without intersection logic.
                    # Let's use sets for this specific check, it's small (35 elements).
                    set_i = set(idxs_i)
                    set_j = set(idxs_j)
                    inter_ij = set_i & set_j
                    union_ij = set_i | set_j
                    
                    at_risk = []
                    for s in union_ij:
                        c = count_arr[s]
                        if c == 1:
                            # Unique to i or j
                            at_risk.append(s)
                        elif c == 2 and s in inter_ij:
                            # Covered by i and j, but no one else
                            at_risk.append(s)
                    
                    if not at_risk:
                        # Safe to remove both? 
                        # Wait, if at_risk is empty, it means every subset in i or j is covered by >=2 blocks, 
                        # and if covered by 2, it's not just i and j.
                        # So we can remove both.
                        if i > j: i, j = j, i
                        current_blocks.pop(j)
                        current_blocks.pop(i)
                        count_arr[idxs_i] -= 1
                        count_arr[idxs_j] -= 1
                        improved = True
                        break
                    else:
                        # Try to find a block k covering all at_risk
                        # Start with blocks covering the first at-risk subset
                        if at_risk:
                            cands = set(subset_to_blocks[at_risk[0]])
                            for s in at_risk[1:]:
                                cands &= set(subset_to_blocks[s])
                                if not cands:
                                    break
                            if cands:
                                # Remove i and j from candidates
                                cands.discard(b_i)
                                cands.discard(b_j)
                                if cands:
                                    best_k = random.choice(list(cands))
                                    # Verify best_k covers all at_risk (should be true by construction)
                                    # Check if adding k and removing i,j is valid
                                    # We already know k covers at_risk.
                                    # We need to ensure no OTHER subset becomes uncovered.
                                    # Since we only remove i and j, only subsets in i or j are at risk.
                                    # We checked at_risk. So it's valid.
                                    
                                    if i > j: i, j = j, i
                                    current_blocks.pop(j)
                                    current_blocks.pop(i)
                                    current_blocks.append(best_k)
                                    count_arr[idxs_i] -= 1
                                    count_arr[idxs_j] -= 1
                                    count_arr[block_covered_indices[best_k]] += 1
                                    improved = True
                                    break

        return current_blocks

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # 1. Initial Dual Greedy
    init_blocks = dual_greedy()
    improved = local_search(init_blocks, max_time=15)
    if len(improved) < best_size:
        best_size = len(improved)
        best_solution = improved[:]
        
    if time.time() > deadline - 30:
        return [list(all_blocks[idx]) for idx in best_solution]
        
    # 2. Iterated Local Search
    current_blocks = best_solution[:]
    
    while time.time() < deadline - 10:
        # Perturb: remove 3-5 random blocks
        n_remove = random.randint(3, 5)
        if len(current_blocks) > n_remove:
            to_remove = set(random.sample(range(len(current_blocks)), n_remove))
            current_blocks = [b for i, b in enumerate(current_blocks) if i not in to_remove]
            
        # Repair with Dual Greedy (only for uncovered parts)
        # Calculate current coverage
        covered = np.zeros(n_subsets, dtype=bool)
        for b_idx in current_blocks:
            covered[block_covered_indices[b_idx]] = True
        
        # If not fully covered, run a "repair" greedy
        # We can reuse the dual_greedy logic but starting from current state?
        # Easier: Just run standard greedy on uncovered parts, or re-run dual greedy on the whole set?
        # Re-running dual greedy is expensive. 
        # Let's do a quick repair:
        while not np.all(covered):
            if time.time() > deadline - 5:
                break
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            # Pick most constrained uncovered subset
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered_indices])
            min_count = counts.min()
            candidates = uncovered_indices[counts == min_count]
            target_subset = random.choice(candidates.tolist())
            candidate_blocks = subset_to_blocks[target_subset]
            
            # Pick block with max gain among candidates
            best_gain = -1
            best_block = -1
            for b_idx in candidate_blocks:
                gain = np.sum(~covered[block_covered_indices[b_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block != -1:
                current_blocks.append(best_block)
                covered[block_covered_indices[best_block]] = True
            else:
                break
                
        # Local Search
        improved = local_search(current_blocks, max_time=5)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_solution = improved[:]
            current_blocks = best_solution[:]
        else:
            current_blocks = improved[:]

    # Final validation and cleanup
    final_covered = np.zeros(n_subsets, dtype=bool)
    for idx in best_solution:
        final_covered[block_covered_indices[idx]] = True
    
    if not np.all(final_covered):
        # Emergency repair
        while not np.all(final_covered):
            if time.time() > deadline - 2:
                break
            uncovered = np.where(~final_covered)[0]
            if len(uncovered) == 0:
                break
            counts = np.array([len(subset_to_blocks[i]) for i in uncovered])
            min_count = counts.min()
            candidates = uncovered[counts == min_count]
            target = random.choice(candidates.tolist())
            for b_idx in subset_to_blocks[target]:
                if b_idx not in best_solution:
                    best_solution.append(b_idx)
                    final_covered[block_covered_indices[b_idx]] = True
                    break
            else:
                best_solution.append(random.randint(0, n_blocks - 1))
                final_covered = np.zeros(n_subsets, dtype=bool)
                for idx in best_solution:
                    final_covered[block_covered_indices[idx]] = True

    # Remove duplicates
    best_solution = list(dict.fromkeys(best_solution))

    return [list(all_blocks[idx]) for idx in best_solution]

if __name__ == "__main__":
    # Self-test
    blocks = construct(19, 7, 4)
    print(f"Blocks found: {len(blocks)}")
    # Verify
    t_subsets = list(itertools.combinations(range(19), 4))
    block_sets = [set(b) for b in blocks]
    all_covered = True
    for sub in t_subsets:
        sub_set = set(sub)
        if not any(sub_set.issubset(bs) for bs in block_sets):
            all_covered = False
            break
    print(f"Valid: {all_covered}")
