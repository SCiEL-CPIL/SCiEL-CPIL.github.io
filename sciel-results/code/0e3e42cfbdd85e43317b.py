import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Compute masks for each block
    # Using Python integers for bitmasks (num_t_subsets <= 3876 for 19,4)
    block_masks = [0] * num_blocks
    # Also store the list of subset indices for each block for faster iteration later
    block_subsets = [[] for _ in range(num_blocks)]
    
    for i, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs
        
    # Inverted index: subset_idx -> list of block indices
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    def count_bits(x):
        return x.bit_count()

    def greedy_construct(randomize=False):
        """
        Greedy set cover.
        If randomize is True, adds slight noise to break ties and explore different solutions.
        """
        uncovered = (1 << num_t_subsets) - 1
        chosen = []
        used = [False] * num_blocks
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            
            # Optimization: Only consider blocks that cover at least one uncovered subset?
            # With 50k blocks, scanning all is ~50k * 3876/64 ops? No, bitwise AND is fast.
            # But Python loop over 50k is slow (~0.05s per iteration? 50k * 0.1us = 5ms).
            # 150 blocks * 5ms = 0.75s. Acceptable.
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = count_bits(block_masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and randomize:
                    # Random tie-breaking
                    if random.random() < 0.5:
                        best_idx = i

            if best_idx == -1:
                break # Should not happen if uncovered != 0 and valid blocks exist
                
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        """
        Removes blocks that are not essential.
        Returns the pruned list of indices.
        """
        if not chosen_indices:
            return []
        
        # Count coverage of each subset
        counts = [0] * num_t_subsets
        chosen_set = set(chosen_indices)
        
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        # Iteratively remove non-essential blocks
        changed = True
        current_list = list(chosen_indices)
        
        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                # Check if block idx is essential
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block idx
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            
            current_list = new_list
        
        return current_list

    def try_swap_improvement(chosen_indices, max_swaps=10):
        """
        Attempts to improve the solution by swapping blocks.
        Returns the new list if improved, else the original.
        """
        current_solution = list(chosen_indices)
        improved = True
        while improved:
            improved = False
            # Compute counts for current solution
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Identify essential subsets for each block
            # We try to swap each block
            for i, idx in enumerate(current_solution):
                # Find essential subsets for block idx
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    # Should have been pruned, but just in case
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break # Restart loop as list changed
                
                # Find candidates that cover ALL essential_subs
                # Pick the first essential sub to get a candidate pool
                pool = subset_to_blocks[essential_subs[0]]
                
                best_candidate = -1
                best_score = -1
                
                for cand_idx in pool:
                    if cand_idx in current_solution:
                        continue
                    
                    # Check if cand covers all essential_subs
                    covers_all = True
                    cand_mask = block_masks[cand_idx]
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: How much "redundancy" does this candidate add?
                    # i.e., how many subsets in cand are already covered by others (count > 1 after removing idx)?
                    # If we swap idx with cand:
                    # New counts = old counts - idx + cand
                    # For a subset s in cand:
                    # If s in idx: count_new = count_old - 1 + 1 = count_old (if s in both)
                    # If s not in idx: count_new = count_old + 1
                    # We want to maximize the number of subsets in cand that are ALREADY covered (count_old > 1).
                    # This increases the chance that other blocks become removable.
                    
                    redundancy = 0
                    cand_subs = block_subsets[cand_idx]
                    idx_subs = set(block_subsets[idx]) # Cache this?
                    
                    for s in cand_subs:
                        if s in idx_subs:
                            # Covered by idx (1) and cand (1).
                            # Old count was C.
                            # New count: C - 1 (remove idx) + 1 (add cand) = C.
                            # If C > 1, it was redundant before and remains redundant.
                            # If C == 1, it was essential for idx, becomes essential for cand.
                            if counts[s] > 1:
                                redundancy += 1
                        else:
                            # Not in idx.
                            # Old count C.
                            # New count C + 1.
                            # If C >= 1, it becomes more redundant (or stays covered).
                            if counts[s] >= 1:
                                redundancy += 1
                    
                    if redundancy > best_score:
                        best_score = redundancy
                        best_candidate = cand_idx
                
                if best_candidate != -1:
                    # Perform swap
                    current_solution[i] = best_candidate
                    # We don't immediately prune here, but the next iteration of 'while improved'
                    # will recompute counts and prune.
                    # To ensure we don't loop forever without improvement, we rely on 'prune_solution'
                    # reducing the size.
                    # However, a swap that doesn't allow immediate removal might be a dead end.
                    # We only accept the swap if it leads to a smaller solution after pruning.
                    
                    # Temporarily apply swap
                    old_idx = idx
                    new_idx = best_candidate
                    
                    # Check if this swap allows removal of other blocks
                    # We can just call prune_solution on the new list and see if it's smaller.
                    # But that's expensive.
                    # Alternative: Just do the swap, then prune at the end of the outer loop?
                    # No, we need to know if it's an improvement.
                    
                    # Let's assume the swap is good if it increases redundancy significantly.
                    # To be safe, we perform the swap, prune, and check size.
                    
                    test_solution = list(current_solution)
                    pruned_test = prune_solution(test_solution)
                    
                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        break # Restart scanning with new solution
                    # Else, revert (do nothing, continue to next block)
            
        return current_solution

    # Main Loop: Try multiple random greedy constructions
    best_solution = None
    best_len = float('inf')
    
    # Initial greedy (deterministic)
    init_sol = greedy_construct(randomize=False)
    pruned_init = prune_solution(init_sol)
    if len(pruned_init) < best_len:
        best_len = len(pruned_init)
        best_solution = pruned_init
    
    # Try randomized greedy + swap improvements
    # Time budget check
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    # Estimate time for one greedy+prune+swap cycle.
    # Greedy: ~150 iterations * 50k checks.
    # Let's run as many cycles as possible.
    
    cycles = 0
    while time.time() < deadline - 5.0: # Keep 5s for final safety
        # Randomized Greedy
        sol = greedy_construct(randomize=True)
        sol = prune_solution(sol)
        
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
        
        # Try to improve with swaps
        # Only if we have time and the solution isn't already perfect (lower bound is 131, best known 151)
        # If we are already at 151, swap might help.
        
        # Limit swap attempts to avoid long hangs
        improved_sol = try_swap_improvement(sol, max_swaps=5)
        
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_solution = improved_sol
        
        cycles += 1
        # If we hit 131, we are done (lower bound)
        if best_len <= 131:
            break

    # Convert indices to lists of points
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

# Quick test
if __name__ == "__main__":
    # Test with v=19, k=7, t=4
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
    # Verify coverage
    v, k, t = 19, 7, 4
    all_t = list(itertools.combinations(range(v), t))
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    is_valid = all(t_sub in covered for t_sub in all_t)
    print(f"Valid: {is_valid}")
