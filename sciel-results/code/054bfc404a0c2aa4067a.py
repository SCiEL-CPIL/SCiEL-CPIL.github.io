import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Build incidence matrix M where M[i, j] = 1 if t-subset i is in k-block j
    # Using numpy for fast operations
    # M shape: (num_t_subsets, num_k_blocks)
    # This might be large: 11628 * 75582 ~ 879M entries. 
    # 879M * 1 byte (uint8) ~ 879 MB. This is risky for memory.
    # Let's check size.
    # C(19,5) = 11628
    # C(19,8) = 75582
    # 11628 * 75582 = 878,854,296 entries.
    # If uint8, ~879 MB. NumPy usually handles this if RAM allows, but it's tight.
    # Alternative: Sparse representation or list of lists.
    # Given the time limit, building this matrix takes time.
    # Let's try a more memory-efficient approach or optimize the loop.
    
    # Actually, 879MB is likely too much for standard evaluation environments (often 2-4GB limit, but overhead exists).
    # Let's use a list of arrays or a different structure.
    # Or, we can avoid the full matrix and compute gains on the fly more efficiently?
    # No, "global greedy" needs to compare gains across ALL blocks.
    # We need to know, for each block, how many NEW subsets it covers.
    
    # Let's try to build the matrix in chunks or use a sparse format if possible.
    # However, standard library doesn't have sparse. SciPy does.
    # But building it is O(num_t * num_k * t) which is huge.
    
    # Alternative Strategy:
    # Instead of precomputing the full matrix, we can use a "block-centric" greedy.
    # 1. Start with empty set of blocks.
    # 2. Maintain `uncovered` as a set of t-subsets.
    # 3. To find the best block, we need to maximize |block ∩ uncovered|.
    #    This is equivalent to: for each block B, count how many t-subsets of B are in `uncovered`.
    #    If we iterate all 75k blocks every step, and we have ~300 steps, that's 22.5M block checks.
    #    Each check involves C(8,5)=56 subset lookups. 22.5M * 56 = 1.2B lookups.
    #    This is too slow in Python.
    
    # Let's go back to the incidence matrix but optimize memory.
    # We can store for each t-subset, the list of blocks that contain it.
    # Or for each block, the list of t-subsets it contains.
    
    # Let's try:
    # block_to_ts: list of lists. block_to_ts[j] = list of indices of t-subsets in block j.
    # ts_to_blocks: list of lists. ts_to_blocks[i] = list of indices of blocks containing t-subset i.
    
    # Construction time:
    # For each block (75k), generate 56 t-subsets. 75k * 56 = 4.2M operations.
    # For each t-subset (11k), generate C(19-5, 8-5) = C(14,3) = 364 blocks. 11k * 364 = 4.2M operations.
    # This is feasible.
    
    # Let's build ts_to_blocks.
    # ts_to_blocks[i] contains indices of all k-blocks that contain t-subset i.
    
    # But wait, for the greedy step:
    # We want to pick block B that maximizes | { ts in B | ts is uncovered } |.
    # If we have `uncovered` as a boolean array `is_uncovered` of size num_t_subsets.
    # For a given block B, gain = sum(is_uncovered[ts] for ts in block_to_ts[B]).
    
    # If we iterate all 75k blocks and sum 56 elements each, it's 4.2M ops per step.
    # With ~300 steps, that's 1.2B ops. Still too slow in Python.
    
    # We need to update gains incrementally.
    # Let `gain[B]` be the current number of uncovered t-subsets in block B.
    # Initially, gain[B] = C(k,t) = 56 for all B.
    # When we pick a block B* and add it to the solution:
    #   1. We mark all t-subsets in B* as covered.
    #   2. For every other block B, if it shares any t-subset with B*, we must decrement gain[B].
    
    # How to update gains efficiently?
    # For each t-subset ts in B*:
    #   For every block B in ts_to_blocks[ts]:
    #       gain[B] -= 1
    #       (Note: if B == B*, we don't really care about its gain anymore as it's selected, 
    #        but we can just let it go to 0 or whatever. Actually, once selected, we ignore it.)
    
    # Complexity of update:
    # B* has 56 t-subsets.
    # Each t-subset is in C(14,3) = 364 blocks.
    # So for each step, we iterate 56 * 364 = 20,384 block updates.
    # With 300 steps, total updates = 6M. This is VERY fast in Python.
    
    # So the algorithm is:
    # 1. Precompute `block_to_ts` (list of 75k lists, each with 56 ints)
    # 2. Precompute `ts_to_blocks` (list of 11k lists, each with 364 ints)
    # 3. Initialize `gain` array of size 75k with 56.
    # 4. Initialize `is_uncovered` array of size 11k with True.
    # 5. Loop:
    #      Find B* = argmax(gain) among unselected blocks.
    #      If max gain == 0, break.
    #      Add B* to solution.
    #      Mark B* as selected.
    #      For each ts in block_to_ts[B*]:
    #          if is_uncovered[ts]:
    #              is_uncovered[ts] = False
    #              For each B in ts_to_blocks[ts]:
    #                  gain[B] -= 1
    
    # This is O(Steps * 56 * 364) ~ 6M operations. Extremely fast.
    # This will likely run in < 1 second.
    # We can then use the remaining time for local search.
    
    # Let's implement this.
    
    # Precomputation
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    # Map t-subset to index
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # State
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Greedy Phase
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Find best block
        # Mask selected blocks with -1 or 0
        # We want max gain among unselected
        # To avoid modifying gain array permanently for selected blocks (or to handle them),
        # we can just check if is_selected[b] before accepting.
        # But argmax will pick a selected one if its gain is high? No, we decrement gains of *other* blocks.
        # The selected block's gain remains whatever it was.
        # We should set gain of selected blocks to -1 or 0 to ignore them.
        
        # Optimization: Instead of masking every time, we can just find max and check.
        # But if the global max is a selected block, we need the next max.
        # Better: set gain[selected] = -1.
        
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
            
        # Select block max_idx
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1 # Mark as selected/ignored
        
        # Update gains for all blocks sharing t-subsets with this one
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                # Decrement gain for all blocks containing this ts
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1
                    # If is_selected, we don't care about its gain, but we already set it to -1.
                    # Wait, if we set gain[max_idx] = -1, and then later some other block shares a ts with it?
                    # No, we only update gains for *unselected* blocks in the loop above?
                    # Actually, if b_idx is selected, its gain is already -1. We should not decrement it further (or it doesn't matter).
                    # But we must be careful: if we decrement a selected block's gain, it might become -2, -3...
                    # Since we only care about argmax among *unselected* blocks, and we set selected to -1,
                    # as long as unselected gains are >= 0, it's fine.
                    # Unselected gains start at 56 and decrease. They won't go below 0.
                    # So selected blocks (-1) will always be lower.
                    
    # Local Search Phase
    # Try to improve the solution by swapping blocks or removing redundant blocks.
    # Strategy:
    # 1. Try to remove any block that is redundant (i.e., all its t-subsets are covered by other blocks).
    # 2. Try to swap a block for another that covers the same "unique" subsets but allows removing others? 
    #    Simpler: Randomized restarts with the fast greedy?
    #    Since the greedy is deterministic (argmax is deterministic unless ties), 
    #    we can break ties randomly to get different solutions.
    
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Run multiple randomized greedy runs
    # The greedy above is deterministic if we use np.argmax.
    # np.argmax returns the first occurrence of the maximum value.
    # To randomize, we can add small random noise to gains?
    # Or, we can maintain a list of blocks with max gain and pick one randomly.
    
    # Let's implement a randomized version of the greedy.
    
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        
        while True:
            if time.time() > deadline - 1.0:
                return None
            
            # Find all blocks with max gain
            max_gain = np.max(gain)
            if max_gain <= 0:
                break
                
            # Find indices with max_gain
            # Filter out selected ones
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break
                
            # Pick random candidate
            b_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
                            
        return sol

    # Run randomized greedy multiple times
    num_runs = 50
    for i in range(num_runs):
        if time.time() > deadline - 2.0:
            break
            
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
    # Local Search on best_solution
    # Try to remove redundant blocks
    # Build coverage count for each t-subset in the current solution
    # If a block's removal doesn't leave any t-subset with count 0, remove it.
    
    def try_remove_redundant(blocks):
        # Count coverage of each t-subset
        # We need to map t-subset to index again, or use the precomputed structures.
        # Let's use a dictionary or array.
        # Since we have `ts_index_map`, we can use an array.
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Initialize counts
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 0.5:
                    return blocks
                
                # Check if block i is redundant
                # i.e., for all ts in block i, cover_count[ts] > 1
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block i
                    blocks.pop(i)
                    # Update counts
                    for ts in itertools.combinations(blocks[i], t): # Wait, blocks[i] is gone?
                        # We need to store it before popping
                        pass # Logic error above. Fix below.
                    changed = True
                    break # Restart loop as indices changed
                    
        return blocks

    # Fixed try_remove_redundant
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 0.5:
                    return blocks
                
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block i
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
                    
        return blocks

    best_solution = optimize_redundancy(best_solution)
    
    # Final check: ensure validity?
    # The greedy guarantees validity. Redundancy removal preserves validity.
    
    return best_solution

# Helper for combinations count
def C(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result = result * (n - i + 1) // i
    return result
