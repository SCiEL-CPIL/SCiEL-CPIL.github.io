import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    k_to_idx = {ks: i for i, ks in enumerate(all_k_subsets)}

    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    
    # cover_matrix[i] is the list of t-subset indices covered by all_k_subsets[i]
    # We store this as a numpy array of shape (n_k, c_k_t) for fast vectorized ops
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, ks in enumerate(all_k_subsets):
        for j, ts in enumerate(itertools.combinations(ks, t)):
            cover_matrix[i, j] = t_to_idx[ts]

    best_blocks = None
    best_len = float('inf')

    def fast_greedy_cover(seed):
        """
        Vectorized greedy set cover.
        At each step, pick the block that covers the most currently uncovered t-subsets.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Gains: number of uncovered t-subsets covered by each block
            # uncovered[cover_matrix] has shape (n_k, c_k_t)
            # Summing along axis 1 gives gains for each block
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[rng.randint(0, len(candidates) - 1)]
            
            blocks.append(all_k_subsets[chosen_idx])
            uncovered[cover_matrix[chosen_idx]] = False
            
        return blocks

    def local_search_remove_redundant(blocks, time_limit):
        """
        Removes redundant blocks. A block is redundant if all its t-subsets
        are covered by at least one other block in the set.
        """
        if not blocks:
            return blocks
        
        # Map block tuple to its index in all_k_subsets for fast lookup
        # This is O(1) per lookup if we have the dict, but creating it is O(N_k)
        # We already have k_to_idx.
        
        # Build coverage count array: coverage[i] = number of blocks in 'blocks' covering t-subset i
        coverage = np.zeros(n_t, dtype=np.int32)
        
        # To make removal efficient, we need to update coverage when a block is removed.
        # We maintain the list of blocks.
        
        changed = True
        current_blocks = list(blocks)
        
        while changed:
            if time.time() > time_limit:
                break
                
            # Rebuild coverage map from scratch?
            # If N_blocks is small (e.g., < 100), and N_t is 2024, C(k,t) is 35.
            # Rebuilding is O(N_blocks * C(k,t)) = 100 * 35 = 3500 ops. Very fast.
            coverage = np.zeros(n_t, dtype=np.int32)
            for b in current_blocks:
                idx = k_to_idx[b]
                coverage[cover_matrix[idx]] += 1
            
            changed = False
            to_remove = []
            
            for i, b in enumerate(current_blocks):
                idx = k_to_idx[b]
                # Check if any t-subset covered by b is covered by exactly 1 block (itself)
                # If coverage[cover_matrix[idx]] has any 1, then b is NOT redundant.
                # If all are >= 2, then b is redundant.
                if not np.any(coverage[cover_matrix[idx]] == 1):
                    to_remove.append(i)
            
            if to_remove:
                # Remove all redundant blocks at once to speed up convergence
                remove_set = set(to_remove)
                current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
                changed = True
                
        return current_blocks

    def local_search_swap(blocks, time_limit):
        """
        Attempts to swap a block with another block that covers its unique subsets
        and potentially allows for further reduction or maintains coverage with same size.
        This is a heuristic to escape local minima where removal is not possible.
        """
        if not blocks:
            return blocks
            
        current_blocks = list(blocks)
        
        # We will iterate a few times
        for _ in range(5):
            if time.time() > time_limit:
                break
                
            # Build coverage map
            coverage = np.zeros(n_t, dtype=np.int32)
            for b in current_blocks:
                idx = k_to_idx[b]
                coverage[cover_matrix[idx]] += 1
                
            improved = False
            
            # For each block, find its unique t-subsets
            # unique_mask[i] is True if coverage[i] == 1
            unique_mask = (coverage == 1)
            
            for i, b in enumerate(current_blocks):
                if time.time() > time_limit:
                    break
                    
                idx = k_to_idx[b]
                # Get indices of t-subsets covered by b
                b_subsets = cover_matrix[idx]
                
                # Check if any of b's subsets are unique to b
                # unique_in_b = b_subsets[unique_mask[b_subsets]]
                
                # If no unique subsets, it was already removed in the previous step,
                # but maybe it became redundant due to previous swaps?
                # Actually, if it has no unique subsets, it's redundant.
                # The removal step should have handled this.
                # So here we assume b has at least one unique subset.
                
                # We want to find a replacement block b' not in current_blocks
                # such that b' covers all unique subsets of b.
                
                # Let U be the set of unique t-subsets of b.
                # We need a block b' such that U is a subset of subsets(b').
                
                # Optimization:
                # Instead of checking all N_k blocks, we can filter.
                # But N_k is 346k. Checking 346k blocks * 35 subsets is ~12M ops per block.
                # With ~80 blocks, that's 1B ops. Too slow in Python loop.
                
                # Vectorized approach for swap:
                # For each candidate block j, does it cover all unique subsets of i?
                # This is hard to vectorize directly across all i and j without huge memory.
                
                # Alternative heuristic:
                # Pick one unique subset s of b.
                # Find all blocks that cover s.
                # Check if any of them cover ALL unique subsets of b.
                
                # Let's try a simpler, faster heuristic:
                # Just try to replace b with a random block that covers its unique subsets?
                # Or iterate through candidates that cover the "hardest" unique subset?
                
                # Given time constraints, let's stick to a simpler swap:
                # If we find a block that covers the unique subsets, swap it.
                
                # To speed this up, we can precompute for each t-subset, the list of blocks covering it.
                # This is the transpose of the incidence matrix.
                # Since we didn't precompute this, we might be slow.
                
                # Let's check if we have time to do a more expensive search.
                # If not, skip swap.
                
                # For now, skip complex swaps to ensure stability and speed.
                # The removal step is the main driver of improvement.
                
            if not improved:
                break
                
        return current_blocks

    # Main Loop
    restart_count = 0
    # We want to run as many restarts as possible.
    # Each restart takes some time.
    # Greedy is O(N_blocks * N_k * C(k,t)).
    # With vectorization, it's fast.
    
    while time.time() < deadline - 5:
        # Use different seeds for diversity
        seed = restart_count
        
        # 1. Greedy Construction
        blocks = fast_greedy_cover(seed)
        
        if not blocks:
            continue
            
        # 2. Local Search: Remove Redundant
        # Allocate most of the time to the best candidate so far?
        # Or just do a fixed amount per restart?
        # Let's do a quick removal pass for every restart.
        # And a deeper one for the best ones.
        
        # Quick removal (single pass or few passes)
        pruned_blocks = local_search_remove_redundant(blocks, time.time() + 2) # 2s budget
        
        if len(pruned_blocks) < best_len:
            best_len = len(pruned_blocks)
            best_blocks = pruned_blocks[:]
            
            # If we found a new best, try to polish it further with more time
            # but we must be careful not to exceed the global deadline.
            # We can save the best blocks and continue restarting,
            # then do a final polish on the very best at the end.
            
        restart_count += 1
        
        # Check if we are running out of time
        if time.time() > deadline - 10:
            break

    # Final Polish
    if best_blocks is not None:
        # Spend remaining time trying to improve the best solution found
        remaining_time = deadline - time.time()
        if remaining_time > 1:
            # Try more aggressive local search on the best blocks
            # 1. Remove redundant again (in case previous passes were too short)
            polished = local_search_remove_redundant(best_blocks, deadline - 0.5)
            
            if len(polished) < best_len:
                best_len = len(polished)
                best_blocks = polished
                
            # 2. Try Swap-based local search if time permits
            # This is expensive, so only do it if we have > 5s left
            if time.time() < deadline - 5:
                # Implement a simple swap search
                # For each block, try to replace it with a block that covers its unique subsets
                # and results in a smaller set after removal.
                
                # We'll implement a basic version:
                # For each block b in best_blocks:
                #   Find unique subsets U of b
                #   For each candidate block c not in best_blocks:
                #     If c covers U:
                #       new_blocks = (best_blocks - {b}) + {c}
                #       new_blocks = remove_redundant(new_blocks)
                #       If len(new_blocks) < len(best_blocks):
                #          best_blocks = new_blocks
                #          break
                
                # This is O(N_blocks * N_k * Cost(Remove)).
                # Too slow if N_k is large.
                # We will skip this complex swap for now to ensure robustness,
                # relying on the quality of the greedy + removal.
                pass

    return [list(b) for b in best_blocks] if best_blocks is not None else []
