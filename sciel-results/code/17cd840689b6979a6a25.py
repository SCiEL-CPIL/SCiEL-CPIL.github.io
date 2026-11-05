import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # M[i, j] = 1 if block i covers tsub j
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    # Precompute tsub indices for each block to fill M
    # Also store block tuples for output
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Helper to get block list from indices
    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Phase 1: Vectorized Greedy Construction
    # This is fast and provides a strong initial solution
    if time.monotonic() < deadline - 5.0:
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 10.0:
                break
            
            # Calculate gain for each block: number of uncovered tsubs it covers
            gains = M @ uncovered
            
            # Find block with max gain
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and counts
            row = M[best_idx]
            covered_indices = np.where(row == 1)[0]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
            best_counts = counts.copy()
            best_selected_mask = selected_mask.copy()

    # Phase 2: Local Search Polish with ILS
    # We operate on the best solution found so far
    if best_solution is not None:
        current_indices = best_solution[:]
        current_counts = best_counts.copy()
        current_mask = best_selected_mask.copy()
        
        improved = True
        stagnation_counter = 0
        max_stagnation = 50 # If no improvement for 50 iterations, perturb
        
        while time.monotonic() < deadline - 2.0:
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            # If stagnating, perform a perturbation (ILS)
            if stagnation_counter >= max_stagnation and len(current_indices) > 0:
                # Remove 5-10 random blocks
                n_remove = min(5 + rng.randint(0, 5), len(current_indices))
                to_remove = rng.sample(current_indices, n_remove)
                for b_idx in to_remove:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    row = M[b_idx]
                    covered_indices = np.where(row == 1)[0]
                    current_counts[covered_indices] -= 1
                
                # Greedily re-add to cover gaps
                uncovered_mask = current_counts == 0
                while np.any(uncovered_mask):
                    if time.monotonic() > deadline - 5.0:
                        break
                    gains = M @ uncovered_mask
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    current_mask[best_idx] = True
                    current_indices.append(best_idx)
                    row = M[best_idx]
                    covered_indices = np.where(row == 1)[0]
                    current_counts[covered_indices] += 1
                    uncovered_mask = current_counts == 0
                    
                # Reset stagnation
                stagnation_counter = 0
                improved = True
                continue

            improved = False
            current_indices_set = set(current_indices)
            
            # 1. Remove Redundant Blocks
            # A block is redundant if all tsubs it covers have count > 1
            redundant_found = False
            for b_idx in current_indices:
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                if np.all(current_counts[covered_indices] > 1):
                    # Remove it
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    improved = True
                    redundant_found = True
                    # Restart the loop over indices to catch new redundancies
                    # We can't easily break and restart a for loop over a modifying list
                    # So we set a flag and break to re-evaluate
                    break
            
            if redundant_found:
                continue

            # 2. Try Swaps
            # For each block, check if it can be swapped with another to maintain coverage
            # and potentially create redundancies
            # Optimization: Only try swaps for a random subset of blocks to save time?
            # Or iterate through all. 140 blocks * 50k checks is ~7M ops, fast enough in numpy?
            # The check is: M[:, critical_indices].all(axis=1)
            
            swap_done = False
            # Shuffle indices to avoid bias
            idx_list = current_indices[:]
            rng.shuffle(idx_list)
            
            for b_idx in idx_list:
                if time.monotonic() > deadline - 2.0:
                    break
                
                row = M[b_idx]
                covered_indices = np.where(row == 1)[0]
                # Critical tsubs: covered ONLY by this block
                critical_mask = current_counts == 1
                critical_indices = covered_indices[critical_mask[covered_indices]]
                
                if len(critical_indices) == 0:
                    # This is redundant, should have been caught above
                    # But maybe counts changed? Just in case.
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    improved = True
                    swap_done = True
                    break
                
                if len(critical_indices) == 1:
                    # Only one critical tsub. We need a block that covers it.
                    # Any block covering it will work for validity.
                    # We want one that is NOT in current_indices.
                    # To be efficient, we can look at tsub_blocks_map? 
                    # We didn't build tsub_blocks_map explicitly, but we can use M.
                    # M[:, critical_indices[0]] == 1
                    candidates_mask = M[:, critical_indices[0]] == 1
                    candidates_mask &= ~current_mask
                    candidates = np.where(candidates_mask)[0]
                    
                    if len(candidates) > 0:
                        # Pick a candidate. 
                        # Heuristic: Pick one that covers the most "count==1" tsubs (other than the critical one)
                        # to potentially create more redundancies?
                        # Or just pick random.
                        # Let's pick the one that minimizes the "cost" (number of tsubs it covers that are already well-covered)
                        # Actually, simpler: Pick random candidate to diversify.
                        c_idx = candidates[rng.randrange(len(candidates))]
                        
                        # Perform swap
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        current_counts[covered_indices] -= 1
                        
                        current_mask[c_idx] = True
                        current_indices.append(c_idx)
                        new_row = M[c_idx]
                        new_covered = np.where(new_row == 1)[0]
                        current_counts[new_covered] += 1
                        
                        improved = True
                        swap_done = True
                        break
                else:
                    # Multiple critical tsubs.
                    # We need a block that covers ALL of them.
                    # M[:, critical_indices].all(axis=1)
                    # This creates a boolean vector of length num_kblocks
                    candidates_mask = M[:, critical_indices].all(axis=1)
                    candidates_mask &= ~current_mask
                    candidates = np.where(candidates_mask)[0]
                    
                    if len(candidates) > 0:
                        c_idx = candidates[rng.randrange(len(candidates))]
                        
                        # Perform swap
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        current_counts[covered_indices] -= 1
                        
                        current_mask[c_idx] = True
                        current_indices.append(c_idx)
                        new_row = M[c_idx]
                        new_covered = np.where(new_row == 1)[0]
                        current_counts[new_covered] += 1
                        
                        improved = True
                        swap_done = True
                        break
            
            if swap_done:
                continue
                
            # If we get here, no improvements found in this iteration
            # The outer while loop will increment stagnation_counter

    # Final Result
    if best_solution is None:
        return []
        
    return get_blocks(best_solution)
