import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)

    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # Using a sparse-like approach or dense uint8 matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    # Greedy construction
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected_indices = []
    
    # Precompute block gains for initial greedy step? 
    # Just do standard greedy.
    remaining_time_limit = deadline - 60.0  # Leave 60s for ILS
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
        gains = Cover @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    best_solution = selected_indices.copy()
    best_size = len(best_solution)

    # Iterated Local Search
    current_solution = selected_indices.copy()
    
    # Helper to compute cover counts
    def get_cover_counts(sol_indices):
        if not sol_indices:
            return np.zeros(n_tsubs, dtype=np.int32)
        return np.sum(Cover[sol_indices], axis=0)

    # Local Optimizer
    def local_optimize(sol_indices, time_limit):
        sol = sol_indices.copy()
        cover_counts = get_cover_counts(sol)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            new_sol = []
            
            # Pass 1: Remove redundant blocks
            for idx in sol:
                # Check if block idx covers any t-subset with count 1
                # If not, it is redundant
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_sol.append(idx)
                else:
                    cover_counts -= Cover[idx]
                    improved = True
            
            sol = new_sol
            if time.time() > time_limit:
                break

            # Pass 2: Try replacements to reduce criticality or enable removal
            # This is heuristic. We look for blocks that are "critical" (have unique t-subsets)
            # and try to replace them with blocks that cover those criticals but are "better" 
            # in some sense (e.g., cover more low-count t-subsets).
            
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                new_sol = []
                for idx in sol:
                    critical_mask = (Cover[idx] == 1) & (cover_counts == 1)
                    critical_tsubs = np.where(critical_mask)[0]
                    
                    if len(critical_tsubs) == 0:
                        # Should have been removed in Pass 1, but just in case
                        cover_counts -= Cover[idx]
                        improved = True
                        continue
                    
                    # Find alternatives that cover all critical t-subsets
                    # candidate_scores[i] = number of critical tsubs covered by block i
                    candidate_scores = np.sum(Cover[:, critical_tsubs], axis=1)
                    valid_candidates = candidate_scores == len(critical_tsubs)
                    valid_candidates[idx] = False # Exclude self
                    
                    if not np.any(valid_candidates):
                        new_sol.append(idx)
                        continue
                    
                    # Among valid candidates, pick one that maximizes coverage of low-count t-subsets
                    # Low count means high potential for future redundancy
                    low_count_mask = cover_counts <= 2
                    candidate_low_score = np.sum(Cover * low_count_mask, axis=1)
                    
                    valid_indices = np.where(valid_candidates)[0]
                    best_alt = valid_indices[np.argmax(candidate_low_score[valid_indices])]
                    
                    # If best_alt is the same as idx, no change
                    if best_alt == idx:
                        new_sol.append(idx)
                        continue
                        
                    # Replace
                    cover_counts -= Cover[idx]
                    cover_counts += Cover[best_alt]
                    new_sol.append(best_alt)
                    changed = True
                    improved = True
                
                sol = new_sol
                
        return sol

    # ILS Loop
    current_solution = local_optimize(current_solution, deadline - 10.0)
    best_solution = current_solution
    best_size = len(best_solution)
    
    while time.time() < deadline - 5.0:
        # Perturbation: Randomly swap a few blocks or remove/add
        # Simple perturbation: Pick a random block, try to replace it with a random valid one
        # Or just restart greedy from a slightly different state?
        # Let's try a "kick": Remove 2 random blocks, then re-greedy the uncovered parts?
        # Too expensive.
        
        # Better kick: Randomly replace 1-3 blocks with random alternatives that maintain validity?
        # Hard to maintain validity easily.
        
        # Alternative: Just run local_optimize again with a different random seed for tie-breaking?
        # The current local_optimize is deterministic.
        
        # Let's implement a random restart of the greedy phase with a small bias?
        # Or, perturb the current solution by swapping one block for a random other block,
        # and if it becomes invalid, repair it with greedy.
        
        sol = current_solution.copy()
        if len(sol) == 0:
            break
            
        # Pick a random index to perturb
        idx_to_perturb = random.randint(0, len(sol) - 1)
        old_block = sol[idx_to_perturb]
        
        # Find a random alternative block
        # To ensure validity, we can just remove it and re-greedy the uncovered t-subsets.
        # This might increase size, but ILS can handle that if we keep the best.
        
        # Remove old block
        sol.remove(old_block)
        
        # Calculate uncovered
        cover_counts = get_cover_counts(sol)
        uncovered_mask = (cover_counts == 0)
        
        if np.any(uncovered_mask):
            # Re-greedy only for uncovered
            # This is expensive if we do it fully.
            # We can just add blocks to cover uncovered.
            uncovered_indices = np.where(uncovered_mask)[0]
            # This is complex to do efficiently without rebuilding the matrix.
            # Given time constraints, a simpler perturbation is better.
            
            # Let's try a different perturbation: 
            # Randomly swap two blocks in the solution.
            # If the solution remains valid, keep it.
            # Check validity is O(N * k * t) which is small.
            
            if len(sol) < 2:
                # Can't swap
                current_solution = sol # It got smaller? No, we just removed one. 
                # If it's invalid, we must revert.
                # Let's check validity of sol (without old_block)
                # If valid, we improved!
                if not np.any(uncovered_mask):
                    best_solution = sol
                    best_size = len(sol)
                    current_solution = sol
                    continue
                else:
                    # Revert
                    sol.append(old_block)
                    current_solution = sol
                    continue
            
            i, j = random.sample(range(len(sol)), 2)
            sol[i], sol[j] = sol[j], sol[i]
            
            # Check validity
            cover_counts_check = get_cover_counts(sol)
            if np.any(cover_counts_check == 0):
                # Invalid, revert
                sol[i], sol[j] = sol[j], sol[i]
            else:
                current_solution = sol
                # Try to optimize this new valid solution
                current_solution = local_optimize(current_solution, deadline - 10.0)
                if len(current_solution) < best_size:
                    best_solution = current_solution
                    best_size = len(current_solution)
                    
    return [all_kblocks[i] for i in best_solution]

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    t0 = time.time()
    result = construct(v, k, t)
    t1 = time.time()
    print(f"C({v},{k},{t}): {len(result)} blocks, time {t1 - t0:.1f}s")
    # Verify
    all_tsubs = set(itertools.combinations(range(v), t))
    covered = set()
    for b in result:
        assert len(b) == k and len(set(b)) == k
        covered.update(itertools.combinations(b, t))
    assert covered == all_tsubs, "Invalid covering!"
    print("Valid!")
