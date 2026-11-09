import numpy as np
import time
import random

def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard: 330s limit. We will use ~300s for search.
    start_time = time.time()
    deadline = start_time + 300.0
    
    # 1. Initialize heuristic with a reasonable baseline (tight fit preference)
    pair_sums = d[:, None] + d[None, :]
    valid = pair_sums <= capacity
    eta = np.full((n, n), 1e-9, dtype=float)
    waste = capacity - pair_sums
    waste_valid = waste[valid]
    eta[valid] = 1.0 / (1.0 + waste_valid / 5.0) ** 2
    
    # Ensure initial validity
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # 2. Simulation function
    def simulate(eta_matrix, seed_offset=0):
        """
        Runs the ant colony optimization using the given heuristic matrix.
        Returns the fewest bins used by any ant.
        """
        rng = np.random.RandomState(42 + seed_offset) # Fixed base seed for reproducibility in search
        
        pheromone = np.ones((n, n))
        best_bins = float('inf')
        
        # Pre-compute valid fits for speed
        # Note: The problem says "placing item j right after item i". 
        # The ant picks j given current i.
        # We need to know if j fits in the bin containing i? 
        # Actually, the problem states: "places a random fitting item, then repeatedly picks a fitting item with probability proportional to pheromone[current, j] * heuristic[current, j]. When none fits, it opens a new bin."
        # This implies the "current" item is the last placed item in the bin.
        # The condition "fitting" usually means size_j + current_bin_load <= capacity.
        # However, the heuristic is eta[i][j].
        # Let's re-read carefully: "attractiveness for placing item j right after item i in the same bin".
        # The ant state is (current_item, bin_load).
        # If we place j after i, the new load is bin_load + size_j.
        # The constraint is bin_load + size_j <= capacity.
        
        for _ in range(30): # 30 iterations
            iter_best = float('inf')
            for _ in range(20): # 20 ants
                # Ant starts
                bins_used = 0
                placed = np.zeros(n, dtype=bool)
                current_bin_load = 0
                current_item = -1
                
                # 1. Open a bin, place a random fitting item
                available = np.where((~placed) & (d <= capacity))[0]
                if len(available) == 0:
                    return float('inf')
                
                first_idx = rng.choice(available)
                placed[first_idx] = True
                current_bin_load = d[first_idx]
                current_item = first_idx
                bins_used = 1
                
                # 2. Repeatedly pick a fitting item
                while True:
                    # Find candidates j that fit in current bin
                    # Condition: current_bin_load + d[j] <= capacity
                    # And j is not placed
                    fits_mask = (~placed) & (current_bin_load + d <= capacity)
                    
                    # Exclude current_item? No, items are distinct indices.
                    # The matrix is item-by-item.
                    # We pick j from available items that fit.
                    candidates = np.where(fits_mask)[0]
                    
                    if len(candidates) == 0:
                        # Open new bin
                        remaining = np.where(~placed)[0]
                        if len(remaining) == 0:
                            break
                        # Pick a random fitting item for the new bin? 
                        # "When none fits, it opens a new bin."
                        # Then what? It must place an item in the new bin.
                        # The rule says "places a random fitting item" for the first bin.
                        # For subsequent bins, it likely follows the same logic or just picks any fitting item.
                        # "Each ant opens a bin, places a random fitting item..."
                        # This phrasing usually applies to the start of a bin.
                        # Let's assume we pick a random fitting item for the new bin too.
                        new_candidates = np.where((~placed) & (d <= capacity))[0]
                        if len(new_candidates) == 0:
                            break
                        next_idx = rng.choice(new_candidates)
                        placed[next_idx] = True
                        current_bin_load = d[next_idx]
                        current_item = next_idx
                        bins_used += 1
                        continue
                    
                    # Calculate probabilities
                    # prob[j] = pheromone[current_item, j] * eta[current_item, j]
                    # Note: pheromone is updated per iteration, not per step.
                    # "After each iteration, pheromone is multiplied by 0.9 and every entry gains 1/(fewest bins of that iteration)."
                    # So pheromone is static during an ant's run?
                    # "Ant colony: 20 ants for 30 iterations... After each iteration, pheromone is multiplied by 0.9..."
                    # Yes, pheromone is constant for all ants in a specific iteration.
                    
                    p_vals = pheromone[current_item, candidates] * eta_matrix[current_item, candidates]
                    p_vals = np.maximum(p_vals, 1e-9) # Ensure positive
                    total_p = np.sum(p_vals)
                    if total_p == 0:
                        break
                    probs = p_vals / total_p
                    
                    next_idx = candidates[rng.choice(len(candidates), p=probs)]
                    placed[next_idx] = True
                    current_bin_load += d[next_idx]
                    current_item = next_idx
                
            # End of ant
            if bins_used < iter_best:
                iter_best = bins_used
        
        # End of iteration
        if iter_best < best_bins:
            best_bins = iter_best
        
        # Update Pheromone
        pheromone *= 0.9
        pheromone += 1.0 / iter_best
        
        return best_bins

    # 3. Local Search / Simulated Annealing
    best_score = simulate(eta)
    current_score = best_score
    current_eta = eta.copy()
    
    # We will perturb the eta matrix and see if it improves the simulation score.
    # Since simulation is expensive (20 ants * 30 iters * 120 items), we need to be efficient.
    # One simulation run:
    # 30 iters * 20 ants.
    # Each ant places 120 items.
    # 30 * 20 * 120 = 72,000 steps.
    # Each step involves vector operations on n=120.
    # This should be fast enough to run thousands of simulations in 300s?
    # Let's estimate: 72k steps * ~100 ops/step = 7.2M ops.
    # Python overhead is high. Maybe 0.05s per simulation?
    # 300s / 0.05s = 6000 simulations.
    
    # Strategy: 
    # 1. Start with a good baseline.
    # 2. Randomly perturb a subset of eta values (e.g., scale them or add noise).
    # 3. If it improves, accept. If it worsens, accept with probability exp(-delta/T).
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # To make it faster, we can reduce the number of ants/iterations in the search phase?
    # No, the score depends on the full 20x30 run. We must simulate the full run to get the true score.
    # However, we can use a "quick check" with fewer ants to filter out bad moves?
    # Risky, but might be effective. Let's stick to full simulation for robustness first.
    
    # Optimization: Vectorize the ant simulation if possible?
    # Hard to vectorize across ants because of the sequential placement and branching.
    # But we can optimize the inner loop.
    
    # Let's try to optimize the simulate function first.
    # Use local variables for speed.
    
    def simulate_fast(eta_matrix):
        rng = np.random.RandomState(42) # Fixed seed for consistency in search
        pheromone = np.ones((n, n))
        best_bins = float('inf')
        
        for _ in range(30):
            iter_best = float('inf')
            for _ in range(20):
                bins_used = 0
                placed = np.zeros(n, dtype=bool)
                current_bin_load = 0.0
                current_item = -1
                
                # Initial placement
                available = np.where((~placed) & (d <= capacity))[0]
                if len(available) == 0:
                    return float('inf')
                first_idx = rng.choice(available)
                placed[first_idx] = True
                current_bin_load = d[first_idx]
                current_item = first_idx
                bins_used = 1
                
                while True:
                    # Check remaining items
                    if np.all(placed):
                        break
                    
                    # Find fitting items
                    # d[j] <= capacity - current_bin_load
                    max_size = capacity - current_bin_load
                    fits_mask = (~placed) & (d <= max_size)
                    candidates = np.where(fits_mask)[0]
                    
                    if len(candidates) == 0:
                        # New bin
                        remaining = np.where(~placed)[0]
                        if len(remaining) == 0:
                            break
                        # Random fitting item for new bin
                        # All remaining items fit in a new bin (since d <= 99 < 150)
                        next_idx = rng.choice(remaining)
                        placed[next_idx] = True
                        current_bin_load = d[next_idx]
                        current_item = next_idx
                        bins_used += 1
                        continue
                    
                    # Probabilities
                    # p = tau[current, j] * eta[current, j]
                    p_vals = pheromone[current_item, candidates] * eta_matrix[current_item, candidates]
                    # Clip to 1e-9
                    p_vals = np.maximum(p_vals, 1e-9)
                    total_p = np.sum(p_vals)
                    if total_p < 1e-15:
                        # Fallback to random
                        next_idx = rng.choice(candidates)
                    else:
                        probs = p_vals / total_p
                        next_idx = candidates[rng.choice(len(candidates), p=probs)]
                    
                    placed[next_idx] = True
                    current_bin_load += d[next_idx]
                    current_item = next_idx
                
            if bins_used < iter_best:
                iter_best = bins_used
            
            if iter_best < best_bins:
                best_bins = iter_best
            
            pheromone *= 0.9
            pheromone += 1.0 / iter_best
        
        return best_bins

    # Re-evaluate baseline with fast simulator
    current_score = simulate_fast(current_eta)
    best_score = current_score
    
    # Local Search
    # We will perturb eta values.
    # Since eta is positive, we can log-transform it for search, or just multiply by random factors.
    # Let's use multiplicative noise.
    
    n_search_steps = 0
    while time.time() < deadline:
        n_search_steps += 1
        
        # Create a perturbed version
        # Pick a random subset of indices to perturb
        # Or perturb all?
        # Let's perturb all valid entries with small multiplicative noise.
        # To allow exploration, we can also try "resetting" some entries to baseline.
        
        new_eta = current_eta.copy()
        
        # Randomly select 10% of valid entries to perturb significantly
        valid_indices = np.where(valid)
        if len(valid_indices[0]) > 0:
            num_perturb = max(1, int(0.1 * len(valid_indices[0])))
            idx_choice = rng.choice(len(valid_indices[0]), num_perturb, replace=False)
            
            for idx in idx_choice:
                i, j = valid_indices[0][idx], valid_indices[1][idx]
                # Multiplicative noise
                factor = 10 ** rng.uniform(-0.5, 0.5) # 0.316 to 3.16
                new_eta[i, j] = current_eta[i, j] * factor
                # Ensure positive and finite
                if new_eta[i, j] <= 0 or not np.isfinite(new_eta[i, j]):
                    new_eta[i, j] = 1e-9
        
        # Evaluate
        new_score = simulate_fast(new_eta)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            current_eta = new_eta
            if new_score < best_score:
                best_score = new_score
                # Optional: restart with lower temperature if we find a new best?
                # Or just keep cooling.
        else:
            # Simulated annealing acceptance
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_score = new_score
                    current_eta = new_eta
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            # Optional: Restart with higher T if stuck?
            # Let's just keep going.
            
    # Return the best eta found
    # We should return the eta that produced best_score, not current_eta if current is worse.
    # We didn't track the best_eta separately. Let's fix that.
    
    # Wait, I need to track best_eta.
    # Let's restart the search logic properly to track best_eta.
    
    # --- Corrected Search Loop ---
    
    # Reset state
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Baseline
    eta_base = np.full((n, n), 1e-9, dtype=float)
    waste_valid = (capacity - pair_sums)[valid]
    eta_base[valid] = 1.0 / (1.0 + waste_valid / 5.0) ** 2
    eta_base[~np.isfinite(eta_base)] = 1e-9
    eta_base[eta_base <= 0] = 1e-9
    
    best_eta = eta_base.copy()
    best_score = simulate_fast(best_eta)
    current_eta = eta_base.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    # Use a separate RNG for search to not interfere with simulation RNG?
    # Simulation uses fixed seed 42.
    # Search RNG can be random.
    search_rng = np.random.RandomState(123)
    
    while time.time() < deadline:
        # Perturb
        new_eta = current_eta.copy()
        
        # Perturb a random subset of valid entries
        valid_flat_indices = np.where(valid.ravel())[0]
        if len(valid_flat_indices) > 0:
            num_perturb = max(1, int(0.05 * len(valid_flat_indices))) # 5%
            idx_choice = search_rng.choice(len(valid_flat_indices), num_perturb, replace=False)
            
            for idx in idx_choice:
                flat_idx = valid_flat_indices[idx]
                # Multiplicative noise
                factor = 10 ** search_rng.uniform(-0.5, 0.5)
                new_eta.flat[flat_idx] = current_eta.flat[flat_idx] * factor
                if new_eta.flat[flat_idx] <= 0 or not np.isfinite(new_eta.flat[flat_idx]):
                    new_eta.flat[flat_idx] = 1e-9
        
        # Evaluate
        new_score = simulate_fast(new_eta)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            current_eta = new_eta
            if new_score < best_score:
                best_score = new_score
                best_eta = new_eta.copy()
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if search_rng.random() < prob:
                    current_score = new_score
                    current_eta = new_eta
        
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_eta
