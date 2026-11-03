import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    # Generate all possible k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Generate all possible t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Map: block_index -> list of tsubset_indices it covers
    # Map: tsubset_index -> list of block_indices that cover it
    block_tsubs = [[] for _ in range(n_blocks)]
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    
    # Optimization: Precompute combinations for each block once
    # Note: C(7,4) = 35 subsets per block. 50388 * 35 ~ 1.7M operations. Fast.
    for bi, block in enumerate(all_blocks):
        # Calculate t-subsets for this block
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
            
    # Convert block_tsubs to numpy arrays for faster indexing if needed, 
    # but lists are fine for the local search logic which relies on set intersections.
    
    best_solution = None
    best_size = float('inf')
    
    # We will run a few "restarts" or "attempts" if time permits.
    # Attempt 1: Deterministic Greedy
    # Attempt 2: Greedy with random tie-breaking (if time allows)
    
    num_attempts = 3
    for attempt in range(num_attempts):
        if time.time() > deadline - 20:
            break
            
        rng = np.random.default_rng(42 + attempt)
        
        # --- Greedy Phase ---
        solution_blocks = []
        in_solution = np.zeros(n_blocks, dtype=bool)
        coverage_counts = np.zeros(n_tsubsets, dtype=np.int32)
        uncovered_mask = np.ones(n_tsubsets, dtype=bool)
        
        # Precompute block_tsubs as numpy arrays for the greedy phase speed
        # We store them as a list of lists, but for the greedy step, we need to count
        # how many uncovered subsets each block covers.
        
        # To speed up greedy, we can maintain a gain array.
        # However, updating the gain array for all blocks after adding one block is O(N_blocks * C(k,t)).
        # C(7,4)=35. 50000 * 35 = 1.75M ops per step.
        # If we need ~150 blocks, that's 260M ops. Might be too slow in Python loops.
        
        # Alternative: Vectorized Greedy
        # For each step, compute gain for all blocks:
        # gain[j] = sum(uncovered_mask[block_tsubs[j]])
        
        # Let's try to be efficient.
        # Precompute block_tsubs as a numpy array of shape (n_blocks, C(k,t)) with padding?
        # No, lengths are fixed (35). So we can make a (n_blocks, 35) int array.
        
        block_tsubs_arr = np.zeros((n_blocks, 35), dtype=np.int32)
        for bi in range(n_blocks):
            block_tsubs_arr[bi] = block_tsubs[bi]
            
        while np.any(uncovered_mask):
            if time.time() > deadline - 15:
                break
                
            # Calculate gain for all blocks
            # uncovered_mask is (n_tsubsets,) bool
            # block_tsubs_arr is (n_blocks, 35) int
            # We want to sum uncovered_mask[indices] for each row.
            # This is effectively a matrix multiplication or advanced indexing.
            # vals = uncovered_mask[block_tsubs_arr] -> shape (n_blocks, 35) bool
            # gains = vals.sum(axis=1)
            
            # This creates a large temporary array (50k * 35 booleans). ~1.7MB. Fast.
            vals = uncovered_mask[block_tsubs_arr]
            gains = vals.sum(axis=1)
            
            best_idx = np.argmax(gains)
            
            if attempt > 0 and gains[best_idx] > 0:
                # Random tie-breaking for exploration
                max_gain = gains[best_idx]
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 1:
                    best_idx = rng.choice(candidates)
                    
            if gains[best_idx] == 0:
                break
                
            solution_blocks.append(best_idx)
            in_solution[best_idx] = True
            
            # Update coverage
            t_idxs = block_tsubs[best_idx]
            coverage_counts[t_idxs] += 1
            uncovered_mask[t_idxs] = False
            
            if time.time() > deadline - 10:
                break
        
        # If we didn't finish covering, skip this attempt
        if np.any(uncovered_mask):
            continue
            
        # --- Local Search Phase ---
        # We have a valid solution. Now refine it.
        current_solution = solution_blocks.copy()
        current_len = len(current_solution)
        
        # Rebuild in_solution and coverage_counts for the local search context
        # (They are already correct from the greedy phase, but let's ensure consistency)
        # in_solution and coverage_counts are already set.
        
        improved = True
        no_improvement_count = 0
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Pass 1: Remove redundant blocks
            # We iterate through the current solution.
            # To make this efficient, we check if a block is essential.
            
            # Shuffle to avoid bias
            sol_copy = current_solution[:]
            random.shuffle(sol_copy)
            
            for b_idx in sol_copy:
                if time.time() > deadline - 5:
                    break
                    
                t_idxs = block_tsubs[b_idx]
                # Check if any subset in this block is covered ONLY by this block
                # i.e., coverage_counts[t_idxs] == 1
                if np.any(coverage_counts[t_idxs] == 1):
                    continue
                    
                # It is safe to remove
                coverage_counts[t_idxs] -= 1
                in_solution[b_idx] = False
                current_solution.remove(b_idx)
                improved = True
                no_improvement_count = 0
                break # Restart loop with new state
            
            if improved:
                if len(current_solution) < best_size:
                    best_size = len(current_solution)
                    best_solution = current_solution.copy()
                continue
                
            # Pass 2: Swap
            # Try to swap a block A with a block B
            # A is in solution, B is not.
            # Conditions:
            # 1. Removing A leaves some subsets with count 0 (Essentials).
            # 2. B must cover ALL essentials.
            # 3. B must not be in solution.
            
            sol_copy2 = current_solution[:]
            random.shuffle(sol_copy2)
            
            swap_done = False
            for A_idx in sol_copy2:
                if time.time() > deadline - 5:
                    break
                    
                t_idxs_A = block_tsubs[A_idx]
                essential_mask = (coverage_counts[t_idxs_A] == 1)
                essential_t_idxs = t_idxs_A[essential_mask]
                
                if len(essential_t_idxs) == 0:
                    # This block is redundant, should have been caught in Pass 1
                    # But if it wasn't (due to order), we can just remove it?
                    # Pass 1 handles removal. If we get here, it means it's essential.
                    # Wait, if essential_mask is all False, then no subset is exclusively covered by A.
                    # So A is redundant.
                    # Why didn't Pass 1 remove it? Maybe it did, and we broke.
                    # If we are in Pass 2, Pass 1 found no removable blocks.
                    # So essential_mask must be True for at least one?
                    # Actually, if coverage_counts[t_idxs_A] has no 1s, then A is removable.
                    # So if we are here, A has at least one essential subset.
                    pass
                
                if len(essential_t_idxs) == 0:
                    continue
                
                # Find candidates B that cover all essential_t_idxs
                # Intersection of tsub_blocks[e] for e in essential_t_idxs
                candidates = None
                for e_idx in essential_t_idxs:
                    blocks_covering_e = tsub_blocks[e_idx]
                    if candidates is None:
                        candidates = set(blocks_covering_e)
                    else:
                        candidates.intersection_update(blocks_covering_e)
                        
                if not candidates:
                    continue
                
                # Filter out blocks already in solution
                candidates = [b for b in candidates if not in_solution[b]]
                
                if not candidates:
                    continue
                
                # Evaluate candidates
                # We want to pick a B that minimizes "waste" or maximizes future flexibility?
                # For now, just pick the first valid one or a random one.
                # To be smarter, we could count how many *new* subsets B covers that are currently covered only by A?
                # No, B must cover essentials. The rest is bonus.
                # Let's just pick a random candidate to explore different paths.
                
                random.shuffle(candidates)
                B_idx = candidates[0]
                
                # Perform Swap
                # Remove A
                coverage_counts[t_idxs_A] -= 1
                in_solution[A_idx] = False
                current_solution.remove(A_idx)
                
                # Add B
                t_idxs_B = block_tsubs[B_idx]
                coverage_counts[t_idxs_B] += 1
                in_solution[B_idx] = True
                current_solution.append(B_idx)
                
                improved = True
                no_improvement_count = 0
                swap_done = True
                break # Restart loop with new state
                
            if not swap_done and not improved:
                no_improvement_count += 1
                if no_improvement_count > 5:
                    break
        
        if len(current_solution) < best_size:
            best_size = len(current_solution)
            best_solution = current_solution.copy()
            
    # Final Deletion Pass on the best solution found
    if best_solution is not None and time.time() < deadline - 2:
        # Rebuild state for the best solution
        cover_count_final = np.zeros(n_tsubsets, dtype=np.int32)
        in_sol_final = np.zeros(n_blocks, dtype=bool)
        for b_idx in best_solution:
            cover_count_final[block_tsubs[b_idx]] += 1
            in_sol_final[b_idx] = True
            
        changed = True
        current = list(best_solution)
        
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            # Shuffle to ensure we don't get stuck in a specific order
            random.shuffle(current)
            for b_idx in current:
                if time.time() > deadline - 1:
                    break
                t_idxs = block_tsubs[b_idx]
                if np.any(cover_count_final[t_idxs] == 1):
                    new_solution.append(b_idx)
                else:
                    # Remove
                    cover_count_final[t_idxs] -= 1
                    in_sol_final[b_idx] = False
                    changed = True
            current = new_solution
        best_solution = current
        best_size = len(best_solution)

    if best_solution is None:
        # Fallback: Return an empty list or a trivial solution if something went wrong
        # But the greedy should always find a solution for C(19,7,4)
        return []

    return [list(all_blocks[idx]) for idx in best_solution]
