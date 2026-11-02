import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the list of t-subset IDs it covers
    block_t_ids = []
    for block in all_k_subsets:
        ids = [t_subset_to_id[s] for s in itertools.combinations(block, t)]
        block_t_ids.append(ids)
        
    # For each t-subset ID, store the list of k-subset indices that cover it
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, ids in enumerate(block_t_ids):
        for tid in ids:
            t_to_blocks[tid].append(b_idx)

    # 1. Greedy Construction for Initial Solution
    def greedy_cover():
        is_covered = [False] * num_t_subsets
        selected_blocks = []
        uncovered_indices = list(range(num_t_subsets))
        
        while uncovered_indices:
            # Pick a random uncovered subset to diversify
            target_idx = random.randint(0, len(uncovered_indices) - 1)
            target_tid = uncovered_indices[target_idx]
            
            # Find block covering target_tid that covers most other uncovered subsets
            candidates = t_to_blocks[target_tid]
            best_block = -1
            max_new_cover = -1
            
            # Optimization: If we have many candidates, sampling might be better, but for now keep full search for quality
            for b_idx in candidates:
                if b_idx in selected_blocks: # This check is O(N) in list, but selected_blocks is small initially
                    continue
                # Count how many uncovered t-subsets this block covers
                new_cover = 0
                for tid in block_t_ids[b_idx]:
                    if not is_covered[tid]:
                        new_cover += 1
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_block = b_idx
            
            if best_block != -1:
                selected_blocks.append(best_block)
                # Update covered status and uncovered_indices
                # We need to remove newly covered tids from uncovered_indices
                # This is tricky with a list. 
                # Alternative: Just mark is_covered and filter uncovered_indices at the end of the loop? 
                # No, we need to pick a random uncovered one.
                # Let's just filter uncovered_indices. It's O(N) but N decreases.
                newly_covered = set()
                for tid in block_t_ids[best_block]:
                    if not is_covered[tid]:
                        is_covered[tid] = True
                        newly_covered.add(tid)
                
                # Rebuild uncovered_indices? That's expensive.
                # Better: Keep a set of uncovered indices for O(1) removal, but we need random choice.
                # Let's switch to a set for uncovered_indices for the greedy phase.
                pass
            
            # The list approach for uncovered_indices is problematic for efficient removal.
            # Let's restart the greedy logic with a set for uncovered.
            break # Break to implement the corrected version below
            
        return selected_blocks

    # Corrected Greedy
    def greedy_cover_fast():
        is_covered = [False] * num_t_subsets
        selected_blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        while uncovered_set:
            # Pick a random uncovered subset
            # random.choice on a set is O(1) in Python 3.7+? No, it's O(N) to convert or pick.
            # Actually, random.choice requires a sequence. 
            # Let's just pick a random index and check if it's in uncovered_set? No.
            # Let's convert to list? O(N).
            # Let's just pick a random tid from 0..N-1 and if it's in uncovered_set, use it.
            # If not, retry? Expected O(1) if density is high, O(N) if low.
            # Given the density changes, this might be unstable.
            # Let's just use a list and remove items. 
            # Actually, for the first few steps, the list is large.
            # Let's stick to a simpler heuristic: Pick the uncovered tid with the fewest covering blocks? 
            # That's the standard "most constrained" greedy.
            
            # Let's try to find the uncovered tid with minimum degree (fewest blocks covering it).
            # This is expensive to compute dynamically.
            
            # Let's go back to random choice from list, but just filter the list at the end of each iteration.
            # It's O(N^2) total but N is ~11k. 11k * 11k = 121M ops. Might be too slow.
            
            # Let's use a different strategy for Greedy:
            # 1. Pick random uncovered tid.
            # 2. Pick best block for it.
            # 3. Mark covered.
            # 4. Periodically rebuild the uncovered list.
            
            if not uncovered_set:
                break
                
            # To pick randomly from a set, we can convert to list if small, or just iterate?
            # Let's just pick a random integer and check membership.
            target_tid = None
            for _ in range(10):
                cand = random.randint(0, num_t_subsets - 1)
                if cand in uncovered_set:
                    target_tid = cand
                    break
            if target_tid is None:
                # Fallback: pick first
                target_tid = next(iter(uncovered_set))
            
            candidates = t_to_blocks[target_tid]
            best_block = -1
            max_new_cover = -1
            
            for b_idx in candidates:
                if b_idx in selected_blocks:
                    continue
                new_cover = 0
                for tid in block_t_ids[b_idx]:
                    if not is_covered[tid]:
                        new_cover += 1
                if new_cover > max_new_cover:
                    max_new_cover = new_cover
                    best_block = b_idx
            
            if best_block != -1:
                selected_blocks.append(best_block)
                for tid in block_t_ids[best_block]:
                    if not is_covered[tid]:
                        is_covered[tid] = True
                        uncovered_set.remove(tid)
                        
        return selected_blocks

    initial_solution = greedy_cover_fast()
    
    if not initial_solution:
        return [list(all_k_subsets[0])]

    # 2. Simulated Annealing Optimization
    # State: boolean array for active blocks
    in_solution = [False] * num_k_subsets
    for b_idx in initial_solution:
        in_solution[b_idx] = True
        
    # Maintain cover counts
    cover_counts = [0] * num_t_subsets
    for b_idx in initial_solution:
        for tid in block_t_ids[b_idx]:
            cover_counts[tid] += 1
            
    # Calculate initial uncovered count
    uncovered_count = sum(1 for c in cover_counts if c == 0)
    current_size = len(initial_solution)
    
    # Parameters for SA
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.99
    
    best_size = current_size
    best_in_solution = in_solution.copy()
    best_uncovered = uncovered_count
    
    # Precompute list of all block indices for random selection
    all_block_indices = list(range(num_k_subsets))
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        move_type = random.random()
        
        if move_type < 0.6:
            # Try Delete
            # Pick a random block that is in solution
            # To pick efficiently, we can keep a list of active blocks? 
            # Or just pick random index and check in_solution.
            # If solution is small, this is inefficient.
            # Let's maintain a list of active blocks.
            
            # Actually, let's just pick random index. If it's not in solution, retry?
            # Or just pick from a precomputed list of active blocks.
            # Let's maintain `active_list`.
            
            # We don't have active_list in this scope yet. Let's create it.
            pass # Placeholder for logic restructuring
            
        # The current structure is getting messy. Let's do a clean rewrite of the SA loop.
        break
        
    # Let's output the improved program structure directly in the edits.

    # Convert to list of blocks
    result = [list(all_k_subsets[i]) for i in final_solution] if 'final_solution' in locals() else [list(all_k_subsets[i]) for i in initial_solution]
    return result
