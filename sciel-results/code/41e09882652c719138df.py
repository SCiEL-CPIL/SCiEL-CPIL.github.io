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
    
    # Build block_masks as list of Python big ints (fast for bit operations)
    # Also build block_subsets for each block
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

    # Inverted index: for each t-subset, which blocks contain it
    subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for idx in block_subsets[i]:
            subset_to_blocks[idx].append(i)

    # Precompute which blocks cover each subset (for fast lookup in local search)
    # Convert subset_to_blocks to sets for O(1) membership
    subset_to_blocks_set = [set(lst) for lst in subset_to_blocks]
    
    # Precompute rarity of each subset (number of blocks containing it)
    # This can be used for weighted greedy if needed, but let's stick to standard gain first.
    
    def greedy_construct(start_set=None, randomize=False, seed=None):
        """
        Greedy set cover with NumPy-accelerated gain computation.
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
        
        # Track which blocks are still candidates
        # Instead of a list with .remove(), we can just iterate and check used[]
        # But iterating 50k items every step is okay.
        # Let's keep the list for efficiency in iteration if we filter it.
        # available = [i for i in range(num_blocks) if not used[i]]
        # But .remove is O(N).
        # Let's just iterate over all blocks and check used[] in the loop.
        # This avoids the O(N) remove cost and keeps the loop tight.
        
        while uncovered != 0:
            if time.time() > deadline - 5.0:
                break
            
            # Compute gains for all available blocks
            best_gain = -1
            best_idx = -1
            best_idx2 = -1  # second best for randomization
            
            if randomize:
                # Use a subset of candidates for speed?
                # If we iterate all, it's 50k iterations.
                # Let's stick to all for correctness, it's fast enough in C.
                candidates = range(num_blocks)
            else:
                candidates = range(num_blocks)
            
            for i in candidates:
                if used[i]:
                    continue
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    if randomize and best_gain > 0 and random.random() < 0.3:
                        # Keep some diversity
                        best_idx2 = best_idx
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if randomize:
                        if best_idx2 == -1 or random.random() < 0.5:
                            best_idx2 = i
                    else:
                        # Tie-break by index for determinism
                        pass
            
            if best_idx == -1:
                break
            
            # For randomization, occasionally pick second-best
            if randomize and best_idx2 != -1 and random.random() < 0.2:
                best_idx = best_idx2
            
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~block_masks[best_idx]
            
        return chosen

    def prune_solution(chosen_indices):
        """
        Removes redundant blocks by iteratively removing any block
        whose all subsets are covered by other chosen blocks.
        """
        if not chosen_indices:
            return []
        
        # Build count array
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
        1-Swap Local Search with aggressive pruning.
        Optimized to limit candidate pool size.
        """
        current_solution = list(chosen_indices)
        start_ls = time.time()
        
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and time.time() - start_ls < time_limit and iteration < max_iterations:
            improved = False
            iteration += 1
            
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
            
            # Try swaps: for each block, try to replace it with a better one
            best_swap = None
            best_swap_score = 0  # We want to maximize "freedom" - subsets that become non-essential
            
            for i, idx in enumerate(current_solution):
                # Find essential subsets (only covered by this block)
                essential_subs = [s for s in block_subsets[idx] if counts[s] == 1]
                
                if not essential_subs:
                    # Redundant block - just remove it
                    for s in block_subsets[idx]:
                        counts[s] -= 1
                    current_solution.pop(i)
                    improved = True
                    break
                
                # Find candidates that cover all essential subsets
                # Use intersection of subset_to_blocks for all essential subsets
                if len(essential_subs) == 1:
                    pool = subset_to_blocks[essential_subs[0]]
                else:
                    # Intersect
                    pool = list(subset_to_blocks_set[essential_subs[0]])
                    for s in essential_subs[1:]:
                        # Optimize intersection
                        if len(pool) > len(subset_to_blocks[s]):
                            # Swap to iterate over smaller
                            pool = [b for b in subset_to_blocks[s] if b in subset_to_blocks_set[essential_subs[0]]]
                        else:
                            pool = [b for b in pool if b in subset_to_blocks_set[s]]
                
                if not pool:
                    continue
                
                # Limit pool size to speed up search
                MAX_POOL_SIZE = 50
                if len(pool) > MAX_POOL_SIZE:
                    # Sample from pool
                    pool = random.sample(pool, MAX_POOL_SIZE)
                
                # Score candidates: prefer blocks that cover many already-covered subsets
                # (i.e., high redundancy), and prefer blocks that are "different" from current
                best_cand = -1
                best_cand_score = -1
                
                for cand_idx in pool:
                    if cand_idx == idx:
                        continue
                    if cand_idx in current_solution:
                        continue
                    
                    # Verify covers all essential
                    cand_mask = block_masks[cand_idx]
                    covers_all = True
                    for s in essential_subs:
                        if not (cand_mask & (1 << s)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Score: number of subsets in cand that are currently covered by >1 block
                    # After removing idx, these will still be covered
                    redundancy = 0
                    for s in block_subsets[cand_idx]:
                        # After removing idx, count[s] decreases by 1 if s is in idx
                        new_count = counts[s] - (1 if s in block_subsets[idx] else 0)
                        if new_count >= 2:
                            redundancy += 1
                    
                    if redundancy > best_cand_score:
                        best_cand_score = redundancy
                        best_cand = cand_idx
                
                if best_cand != -1 and best_cand_score > 0:
                    # Test the swap
                    test_sol = list(current_solution)
                    test_sol[i] = best_cand
                    pruned_test = prune_solution(test_sol)
                    
                    if len(pruned_test) < len(current_solution):
                        current_solution = pruned_test
                        improved = True
                        break
                    elif len(pruned_test) == len(current_solution) and random.random() < 0.1:
                        # Accept neutral swap occasionally for exploration
                        current_solution = pruned_test
                        improved = True
                        break
            
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
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() < deadline - 10.0:
        if no_improve_count > max_no_improve:
            # Try a full restart with randomization
            sol = greedy_construct(randomize=True)
            sol = local_search(sol, 3.0)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
        
        # Create a kick from the best solution
        if best_len > 0:
            # Strategy: remove a block with lowest "redundancy" (most essential)
            # Compute counts
            counts = [0] * num_t_subsets
            for idx in best_solution:
                for s_idx in block_subsets[idx]:
                    counts[s_idx] += 1
            
            # Find block with minimum essential subsets (easiest to remove)
            min_essential = float('inf')
            kick_idx = 0
            for i, idx in enumerate(best_solution):
                essential_count = sum(1 for s in block_subsets[idx] if counts[s] == 1)
                if essential_count < min_essential:
                    min_essential = essential_count
                    kick_idx = i
            
            # Sometimes use random kick instead
            if random.random() < 0.3:
                kick_idx = random.randint(0, len(best_solution) - 1)
            
            # NEW: Double Kick Strategy
            # Occasionally remove 2 blocks to allow a larger rearrangement
            if random.random() < 0.2 and len(best_solution) > 2:
                kick_idx2 = random.randint(0, len(best_solution) - 1)
                if kick_idx2 == kick_idx:
                    kick_idx2 = (kick_idx2 + 1) % len(best_solution)
                new_base = [b for i, b in enumerate(best_solution) if i != kick_idx and i != kick_idx2]
            else:
                new_base = [b for i, b in enumerate(best_solution) if i != kick_idx]
            
            # Repair with Greedy
            sol = greedy_construct(start_set=new_base, randomize=True)
            
            # Local Search
            sol = local_search(sol, 4.0)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Final polish: one more round of local search on best solution
    if best_solution and time.time() < deadline - 3.0:
        best_solution = local_search(best_solution, 3.0)
    
    # Convert to lists
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks

if __name__ == "__main__":
    blocks = construct(19, 7, 4)
    print(f"Number of blocks: {len(blocks)}")
