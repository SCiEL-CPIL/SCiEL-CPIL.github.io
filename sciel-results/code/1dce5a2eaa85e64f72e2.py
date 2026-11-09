import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Bitmasks for each block
    # Python big ints
    block_masks = [0] * num_blocks
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

    # Inverted index
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    def greedy_construct(start_set=None, randomize=False):
        """
        Greedy set cover.
        If start_set is provided, starts with those blocks already chosen and their subsets covered.
        """
        if start_set is None:
            uncovered = (1 << num_t_subsets) - 1
            chosen = []
            used = [False] * num_blocks
        else:
            uncovered = (1 << num_t_subsets) - 1
            chosen = list(start_set)
            used = [False] * num_blocks
            for idx in chosen:
                used[idx] = True
                uncovered &= ~block_masks[idx]
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            
            # Optimization: Skip used blocks
            # If randomize, add noise?
            for i in range(num_blocks):
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and randomize:
                    if random.random() < 0.5:
                        best_idx = i

            if best_idx == -1:
                break # Should not happen
                
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        """
        Removes redundant blocks.
        """
        if not chosen_indices:
            return []
        
        counts = [0] * num_t_subsets
        for idx in chosen_indices:
            for s_idx in block_subsets[idx]:
                counts[s_idx] += 1
        
        changed = True
        current_list = list(chosen_indices)
        
        while changed:
            changed = False
            new_list = []
            for idx in current_list:
                essential = False
                for s_idx in block_subsets[idx]:
                    if counts[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for s_idx in block_subsets[idx]:
                        counts[s_idx] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            current_list = new_list
            
        return current_list

    def local_search(chosen_indices, time_limit=5.0):
        """
        1-Swap Local Search.
        """
        current_solution = list(chosen_indices)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            
            # Prune first
            pruned = prune_solution(current_solution)
            if len(pruned) < len(current_solution):
                current_solution = pruned
                improved = True
                continue
            
            # Compute counts
            counts = [0] * num_t_subsets
            for idx in current_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Try swaps
            for i, idx in enumerate(current_solution):
                # Find essential subsets
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    # Redundant, should be caught by prune, but remove anyway
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break
                
                # Find candidates covering all essential_subs
                # Use the first essential sub to filter
                pool = subset_to_blocks[essential_subs[0]]
                
                best_candidate = -1
                best_score = -1
                
                # Check candidates
                for cand_idx in pool:
                    if cand_idx in current_solution:
                        continue
                    
                    # Check if covers all essential
                    covers_all = True
                    cand_mask = block_masks[cand_idx]
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: Maximize redundancy of new block
                    # Count how many subsets in cand are already covered (count > 1)
                    # Note: We are removing idx, so counts for idx's subsets decrease by 1.
                    # But for simplicity, just count current redundancy.
                    redundancy = 0
                    for s in block_subsets[cand_idx]:
                        if counts[s] > 1:
                            redundancy += 1
                    
                    if redundancy > best_score:
                        best_score = redundancy
                        best_candidate = cand_idx
                
                if best_candidate != -1:
                    # Perform swap
                    # Check if it improves size after prune
                    test_sol = list(current_solution)
                    test_sol[i] = best_candidate
                    pruned_test = prune_solution(test_sol)
                    
                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        break # Restart scan
                    # Else, do not swap (would not reduce size immediately)
                    # We could accept neutral swaps to explore, but risky with time.
            
        return current_solution

    # Main Loop
    best_solution = None
    best_len = float('inf')
    
    # 1. Initial Deterministic Greedy
    sol = greedy_construct()
    sol = local_search(sol, 5.0)
    best_solution = sol
    best_len = len(sol)
    
    # 2. Iterated Local Search with Kicks
    while time.time() < deadline - 10.0:
        # Create a kick from the best solution
        # Pick a random block to remove
        if best_len > 0:
            kick_idx = random.randint(0, len(best_solution) - 1)
            removed_block = best_solution[kick_idx]
            
            # Remove it
            new_base = [b for i, b in enumerate(best_solution) if i != kick_idx]
            
            # Repair with Greedy (starting from new_base)
            # This is faster than full greedy because many subsets are already covered
            sol = greedy_construct(start_set=new_base, randomize=True)
            
            # Local Search
            sol = local_search(sol, 5.0)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                
            # Optional: If no improvement for a while, try a full restart?
            # Or just keep kicking.
            
    # Convert to lists
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
