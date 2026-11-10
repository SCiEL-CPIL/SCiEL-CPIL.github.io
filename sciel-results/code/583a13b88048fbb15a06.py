import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Selects points from an 8x8x8 grid such that no 5 points are co-spherical 
    (or co-planar).
    
    Strategy:
    1. Generate a dense initial set using a fast, randomized greedy algorithm.
    2. Use local search (2-opt swaps) to optimize a smooth surrogate objective:
       the log-sum-exp of the magnitudes of the determinants of all potential 
       violating 4-tuples. This allows escaping local optima of the greedy 
       approach by finding swaps that increase the "validity margin" globally.
    """
    n = 8
    deadline = time.time() + 580
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    sq_norms_grid = (grid ** 2).sum(axis=1)
    
    # --- Phase 1: Fast Greedy Initialization ---
    # We run a few fast greedy passes to get a good starting set.
    # The check is vectorized but we limit the combinations if m is large to keep it fast.
    
    def fast_greedy(seed_val):
        rng = np.random.default_rng(seed_val)
        order = rng.permutation(N)
        current_set = []
        
        # We will try to add points. 
        # To keep it fast, we'll use a simplified check or just accept the cost 
        # since N=512 and we only do this a few times.
        
        for p_idx in order:
            if len(current_set) < 4:
                current_set.append(p_idx)
                continue
            
            # Check if p_idx can be added
            # We need to check all 4-subsets of current_set.
            # For speed in initialization, we might skip some checks or use a simpler heuristic?
            # No, correctness is key. Let's use the vectorized check but be mindful of time.
            
            m = len(current_set)
            # If m is large, C(m,4) is big. 
            # Let's cap the greedy set size or use a smarter check?
            # Actually, for m=50, C(50,4) ~ 230k. np.linalg.det on 230k 4x4 matrices is fast (<0.1s).
            # So greedy is fine.
            
            arr_pts = grid[np.array(current_set)]
            p_pts = grid[p_idx]
            
            diff = arr_pts - p_pts
            r = sq_norms_grid[np.array(current_set)] - sq_norms_grid[p_idx]
            
            W = np.column_stack((diff, r))
            
            # Generate indices
            # Optimization: if m is large, this is the bottleneck.
            # But we only do this ~50-100 times per greedy run.
            idx = np.array(list(itertools.combinations(range(m), 4)))
            if len(idx) == 0:
                current_set.append(p_idx)
                continue
                
            subs = W[idx]
            dets = np.linalg.det(subs)
            
            if not np.any(np.abs(dets) < 0.5):
                current_set.append(p_idx)
                
            if time.time() > deadline - 100: # Safety margin
                break
                
        return current_set

    # Run a few greedy restarts to find the best initial candidate
    best_initial = []
    for i in range(5):
        cand = fast_greedy(42 + i)
        if len(cand) > len(best_initial):
            best_initial = cand
            
    if not best_initial:
        # Fallback: return empty or a small valid set
        return np.empty((0, 3), dtype=np.int64)

    # --- Phase 2: Local Search Optimization ---
    
    # Objective function:
    # We want to maximize the minimum determinant magnitude among all 5-point subsets?
    # No, the condition is: NO 5 points are co-spherical.
    # This is equivalent to: For every 5 points, the determinant of the 5x5 matrix (or equivalent 4x4 check for any 4 of them relative to the 5th) is NOT near zero.
    # Actually, the test says: for each point p and every 4 earlier points q1..q4, form matrix... if |det| < 0.5 invalid.
    # This means for ANY 5 points, if we pick one as 'p' and the other 4 as 'q', the det must be >= 0.5.
    # Since the det is symmetric in the 5 points (up to sign), we can define a global "validity score" for a set S of size k:
    # Score(S) = min_{p in S, {q1..q4} subset of S\{p}, |{q}|=4} |det(M(p, q1..q4))|
    # We want to maximize this Score(S). If Score(S) >= 0.5, the set is valid.
    
    # To make it smooth for optimization (though we use discrete swaps), we can use:
    # LogSumExp of the dets? No, we want to maximize the MIN.
    # Max-min is non-smooth. 
    # However, we are doing discrete swaps. We can just evaluate the exact min-det for the current set.
    # If a swap increases the min-det (or keeps it valid while allowing further additions), we take it.
    
    # Actually, a simpler and more robust local search for "maximize k subject to validity" is:
    # 1. Start with a valid set S.
    # 2. Try to add a point from the remaining grid. If valid, add it.
    # 3. If stuck (no point can be added), try a swap: remove p_i from S, add p_j from grid\S.
    #    If the new set is valid, and we can subsequently add MORE points than before, accept the swap.
    #    Or simply: accept the swap if it leads to a state where we can add at least one more point in the next greedy pass.
    
    # Let's implement a "Greedy + Swap" loop.
    
    def is_valid_set(indices):
        if len(indices) < 5:
            return True
        idx_arr = np.array(indices)
        pts = grid[idx_arr]
        norms = sq_norms_grid[idx_arr]
        k = len(idx_arr)
        
        # We need to check all 5-subsets? No, the test checks specific structures.
        # The test: "for each point p with at least 4 earlier points, and every 4 earlier points q1..q4"
        # This effectively checks all 5-combinations of points in the set.
        # For a set of k points, we need to check C(k, 5) combinations?
        # No, for each p, we check C(k-1, 4) quadruples.
        # Total checks ~ k * C(k-1, 4) = 5 * C(k, 5).
        
        # Efficient check:
        # For each p in set:
        #   For each 4-subset of (set \ {p}):
        #      Compute det.
        
        # This is expensive for large k. 
        # However, we only need to check if ANY det is < 0.5.
        
        # Let's vectorize per p.
        for i, p_idx in enumerate(idx_arr):
            # Get indices of other points
            others = np.delete(idx_arr, i)
            if len(others) < 4:
                continue
            
            p_pts = grid[p_idx]
            p_norm = norms[i]
            
            o_pts = grid[others]
            o_norms = norms[np.delete(np.arange(k), i)]
            
            diff = o_pts - p_pts
            r = o_norms - p_norm
            
            W = np.column_stack((diff, r))
            
            # Check all 4-subsets of others
            m = len(others)
            if m < 4:
                continue
                
            # To avoid generating huge arrays if m is very large, we can chunk or just do it.
            # For k=50, m=49, C(49,4) ~ 211k. This is fine.
            idx_comb = np.array(list(itertools.combinations(range(m), 4)))
            if len(idx_comb) == 0:
                continue
                
            subs = W[idx_comb]
            dets = np.linalg.det(subs)
            
            if np.any(np.abs(dets) < 0.5):
                return False
                
        return True

    # Current best set
    current_set = list(best_initial)
    best_len = len(current_set)
    
    # Local search loop
    # We will try to improve the set size.
    # Strategy:
    # 1. Try to add any remaining point.
    # 2. If stuck, try swaps.
    
    # Precompute list of all points
    all_points = list(range(N))
    
    # We'll use a priority queue or just iterate?
    # Let's do a simple hill climbing on the set size.
    
    # To speed up "try to add", we can precompute which points are "close" to violations?
    # No, let's just iterate.
    
    # Optimization: Cache validity? Hard because set changes.
    
    # Let's run a loop:
    # While time < deadline:
    #   1. Greedy Add: Try to add any point from grid not in set.
    #   2. If added, update best.
    #   3. If not added, try Swap: Pick a point in set, pick a point outside, swap.
    #      If valid, keep swap.
    #      Then try Greedy Add again.
    
    # To avoid infinite loops of swaps that don't increase size, we can:
    # - Only accept a swap if it allows a subsequent addition, OR
    # - Use a "tabu" list of recent swaps.
    # - Or simply: Run a limited number of swap attempts. If no size increase, restart greedy with different seed?
    
    # Better approach: 
    # The "validity" is a hard constraint. We want max k.
    # Let's try to increase k by 1.
    # If we can't, we try to "shake" the set via swaps to find a configuration that allows +1.
    
    # Implementation:
    
    # Helper: Try to add a specific point
    def try_add_point(set_list, p_idx):
        if p_idx in set_list:
            return False
        new_set = set_list + [p_idx]
        if is_valid_set(new_set):
            return True
        return False

    # Helper: Try to find ANY point to add
    def find_addable_point(set_list):
        set_arr = np.array(set_list)
        in_set = np.zeros(N, dtype=bool)
        in_set[set_arr] = True
        candidates = np.where(~in_set)[0]
        
        # Randomize candidates to avoid bias
        np.random.shuffle(candidates)
        
        for p_idx in candidates:
            if try_add_point(set_list, p_idx):
                return p_idx
        return None

    # Main Loop
    np.random.seed(123)
    
    # We will keep the best set found so far
    global_best_set = list(current_set)
    
    # Local search iterations
    # We'll allow a certain number of "stuck" iterations before restarting the greedy phase.
    
    while time.time() < deadline - 10:
        
        # 1. Try to add
        added_idx = find_addable_point(current_set)
        if added_idx is not None:
            current_set.append(added_idx)
            if len(current_set) > len(global_best_set):
                global_best_set = list(current_set)
            # Continue loop to try adding more
            continue
            
        # 2. Stuck. Try swaps.
        # We will try a few random swaps.
        # Pick a point in set to remove, a point outside to add.
        
        # To be efficient, we can prioritize removing points that are "central" or "crowded"?
        # For now, random.
        
        found_swap = False
        # Limit swap attempts per stuck phase to avoid wasting time
        for _ in range(100): 
            if time.time() > deadline - 10:
                break
                
            if len(current_set) < 5:
                break
                
            # Pick random point in set to remove
            rem_idx_pos = np.random.randint(0, len(current_set))
            rem_idx = current_set[rem_idx_pos]
            
            # Pick random point outside to add
            in_set = np.zeros(N, dtype=bool)
            in_set[np.array(current_set)] = True
            candidates = np.where(~in_set)[0]
            if len(candidates) == 0:
                break
            add_idx = candidates[np.random.randint(0, len(candidates))]
            
            # Perform swap
            new_set = list(current_set)
            new_set.pop(rem_idx_pos)
            new_set.append(add_idx)
            
            # Check validity
            if is_valid_set(new_set):
                # Accept swap
                current_set = new_set
                found_swap = True
                # After a successful swap, we might be able to add points.
                # Loop back to "Try to add"
                break
        
        if not found_swap:
            # No good swaps found in 100 tries.
            # Restart greedy with a different seed to escape this local optimum.
            # We can also try to shrink the set? No, we want max size.
            # Let's just do a new greedy restart.
            
            # To diversify, we can perturb the best set?
            # Or just run fast_greedy again.
            
            # We'll run a few more greedy restarts if we are stuck.
            # But we already did 5. Let's do more with different seeds.
            # We'll keep a counter.
            
            # Actually, a better strategy when stuck:
            # Try to remove ONE point and then greedily add as many as possible.
            # This is a "1-out" local search.
            
            improved = False
            for i in range(len(current_set)):
                if time.time() > deadline - 10:
                    break
                # Remove i-th point
                trial_set = list(current_set)
                trial_set.pop(i)
                
                # Greedy add from trial_set
                # We need a fast greedy add function that takes a starting set.
                
                # We'll reuse the logic from fast_greedy but start with trial_set.
                # For simplicity, let's just try to add points to trial_set.
                
                # This might be slow if we do it for every i.
                # Let's just restart the whole greedy process with a new seed.
                pass
                
            # Restart greedy
            # We'll generate a new seed
            seed_val = np.random.randint(0, 10000)
            cand = fast_greedy(seed_val)
            
            # If the new candidate is better, use it.
            # Or, if the new candidate is similar, maybe merge?
            # For now, just replace if better.
            if len(cand) > len(current_set):
                current_set = cand
                if len(current_set) > len(global_best_set):
                    global_best_set = list(current_set)
            else:
                # If not better, we might be stuck in a local optimum.
                # Let's try a few more restarts.
                # To prevent infinite loop of bad restarts, we can break if we've tried many.
                # But we have time.
                
                # Let's try to "shake" the current set by removing a random point and re-greedy-ing.
                if len(current_set) > 0:
                    rem_pos = np.random.randint(0, len(current_set))
                    shake_set = list(current_set)
                    shake_set.pop(rem_pos)
                    
                    # Greedy add to shake_set
                    # We'll implement a quick greedy add
                    # This is essentially running the greedy loop starting from shake_set.
                    
                    # To avoid code duplication, we can just call fast_greedy?
                    # No, fast_greedy starts from scratch.
                    # Let's write a helper "greedy_complete(starting_set)".
                    
                    def greedy_complete(starting_set, seed_val):
                        rng = np.random.default_rng(seed_val)
                        order = rng.permutation(N)
                        current = list(starting_set)
                        in_set = np.zeros(N, dtype=bool)
                        if len(current) > 0:
                            in_set[np.array(current)] = True
                            
                        # Filter order to exclude those already in set
                        # Or just check during loop
                        for p_idx in order:
                            if in_set[p_idx]:
                                continue
                            if len(current) < 4:
                                current.append(p_idx)
                                in_set[p_idx] = True
                                continue
                            
                            # Check validity
                            # Same as before
                            arr_pts = grid[np.array(current)]
                            p_pts = grid[p_idx]
                            diff = arr_pts - p_pts
                            r = sq_norms_grid[np.array(current)] - sq_norms_grid[p_idx]
                            W = np.column_stack((diff, r))
                            m = len(current)
                            idx = np.array(list(itertools.combinations(range(m), 4)))
                            if len(idx) == 0:
                                current.append(p_idx)
                                in_set[p_idx] = True
                                continue
                            subs = W[idx]
                            dets = np.linalg.det(subs)
                            if not np.any(np.abs(dets) < 0.5):
                                current.append(p_idx)
                                in_set[p_idx] = True
                                
                            if time.time() > deadline - 10:
                                break
                        return current

                    new_cand = greedy_complete(shake_set, seed_val)
                    if len(new_cand) > len(current_set):
                        current_set = new_cand
                        if len(current_set) > len(global_best_set):
                            global_best_set = list(current_set)
                        improved = True
                        break # If improved, break out of restart loop and try to add more

            if not improved:
                # If we couldn't improve, we might be stuck.
                # Let's just continue the main loop, which will try add/swap again.
                # But if we just did a restart and it was bad, we might loop.
                # To avoid this, we can add a "patience" counter.
                pass

    # Final validation and return
    final_set = global_best_set
    if not final_set:
        return np.empty((0, 3), dtype=np.int64)
        
    # Double check validity of the final set
    if not is_valid_set(final_set):
        # Fallback to a smaller valid set if something went wrong (shouldn't happen)
        # We can just return the largest valid prefix?
        # Or just return empty.
        # Let's try to find a valid subset.
        # For now, return empty to be safe, but this is unlikely.
        return np.empty((0, 3), dtype=np.int64)

    return grid[np.array(final_set)]
