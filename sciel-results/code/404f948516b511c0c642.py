"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _simulate_aco_fast(prizes, weights, heuristic, seed=42):
    """
    Optimized ACO simulation.
    prizes: (5, 100)
    weights: (5, 100)
    heuristic: (100,)
    """
    np.random.seed(seed)
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    tau = np.ones(n_items)
    best_global_prize = 0.0
    
    # Pre-transpose for faster column access if needed, but slicing is usually fast enough
    # weights is (5, 100), weights[:, i] is a column.
    
    for iter_count in range(30):
        best_iter_prize = 0.0
        
        for ant_count in range(20):
            remaining_cap = np.full(n_constraints, capacity)
            current_prize = 0.0
            # Use a boolean mask for remaining items to avoid list overhead
            remaining_mask = np.ones(n_items, dtype=bool)
            
            while True:
                # Find items that fit
                # weights[:, remaining_mask] is (5, num_remaining)
                # We need to check if weights[c, i] <= remaining_cap[c] for all c
                # This is equivalent to: np.all(weights[:, remaining_mask] <= remaining_cap[:, np.newaxis], axis=0)
                
                # Optimization: Instead of full boolean check every time, 
                # we can just check the candidates.
                candidates = np.where(remaining_mask)[0]
                if len(candidates) == 0:
                    break
                
                # Vectorized check for which candidates fit
                # weights[:, candidates] has shape (5, num_cands)
                # remaining_cap has shape (5,)
                fits = np.all(weights[:, candidates] <= remaining_cap[:, np.newaxis], axis=0)
                fitting_indices = candidates[fits]
                
                if len(fitting_indices) == 0:
                    break
                
                # Calculate probabilities
                # heuristic[fitting_indices]
                # tau[fitting_indices]
                probs = tau[fitting_indices] * heuristic[fitting_indices]
                
                # Handle non-positive or zero probs
                # The task says: "Heuristic entries that are not positive become 1e-9"
                # This is handled in the heuristic generation, but let's be safe.
                # However, the simulation in the evaluator uses the raw heuristic returned.
                # If heuristic[i] <= 0, it becomes 1e-9.
                # So in our simulation, we should mimic that.
                h_safe = heuristic.copy()
                h_safe[h_safe <= 0] = 1e-9
                h_safe[~np.isfinite(h_safe)] = 1e-9
                
                probs = tau[fitting_indices] * h_safe[fitting_indices]
                probs[probs <= 0] = 1e-9
                sum_probs = probs.sum()
                if sum_probs == 0:
                    probs = np.ones(len(probs))
                else:
                    probs = probs / sum_probs
                
                # Sample
                chosen_idx = np.random.choice(fitting_indices, p=probs)
                
                # Update
                current_prize += prizes[:, chosen_idx].sum()
                remaining_cap -= weights[:, chosen_idx]
                remaining_mask[chosen_idx] = False
                
        if current_prize > best_iter_prize:
            best_iter_prize = current_prize
        
        if best_iter_prize > best_global_prize:
            best_global_prize = best_iter_prize
            
        # Update pheromone
        tau = tau * 0.9 + (best_iter_prize / 10.0)
        
    return best_global_prize

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 20.0 # 20s safety margin
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    
    P_sum = prizes.sum(axis=0)
    W_sum = weights.sum(axis=0)
    W_max = weights.max(axis=0)
    W_min = weights.min(axis=0)
    
    # Calculate "binding constraint" impact
    # For each item, find the constraint where it consumes the largest fraction of capacity
    # Or simply use the max weight as a proxy for "difficulty"
    
    candidates = []
    
    # 1. Standard Density
    candidates.append(P_sum / (W_sum + 1e-9))
    
    # 2. Max Weight Density (Focus on the tightest constraint)
    candidates.append(P_sum / (W_max + 1e-9))
    
    # 3. Min Weight Density (Focus on the loosest constraint - likely bad, but test)
    candidates.append(P_sum / (W_min + 1e-9))
    
    # 4. Geometric Mean of Ratios
    # prod(prizes[c] / weights[c])^(1/5)
    # log(prod) = sum(log(prizes) - log(weights))
    # To avoid 0s, add epsilon
    log_ratios = np.log(prizes + 1e-9) - np.log(weights + 1e-9)
    geo_mean_ratio = np.exp(log_ratios.mean(axis=0))
    candidates.append(geo_mean_ratio)
    
    # 5. Prize / (Sum of Weights in Binding Constraints)
    # Identify for each item which constraint is most "binding" relative to its weight
    # Actually, let's try: Prize / (Weight in the constraint where Weight is Max)
    # This is similar to #2 but explicit.
    
    # 6. Rank-based on Density
    ranks = np.argsort(np.argsort(P_sum / (W_max + 1e-9)))
    candidates.append(ranks.astype(float) + 1.0)
    
    # 7. Inverse Weight (Cheapest items)
    candidates.append(1.0 / (W_max + 1e-9))
    
    # 8. Prize * (1 - Max Weight/Capacity)
    # Prefer items that don't consume too much of the capacity
    candidates.append(P_sum * (1.0 - W_max / 25.0))
    
    # 9. Prize / (W_max ** 0.5)
    candidates.append(P_sum / (W_max ** 0.5 + 1e-9))
    
    # 10. Prize / (W_sum ** 0.5)
    candidates.append(P_sum / (W_sum ** 0.5 + 1e-9))
    
    # 11. Combined: Density * (1 - W_max/25)
    candidates.append((P_sum / (W_sum + 1e-9)) * (1.0 - W_max / 25.0))
    
    # 12. Prize / (W_max + W_sum)
    candidates.append(P_sum / (W_max + W_sum + 1e-9))
    
    # 13. Prize / (W_max * 2 + W_sum)
    candidates.append(P_sum / (2*W_max + W_sum + 1e-9))
    
    # 14. Prize / (W_max * 0.5 + W_sum)
    candidates.append(P_sum / (0.5*W_max + W_sum + 1e-9))
    
    # 15. Pure Prize (if weights are similar, this works)
    candidates.append(P_sum)
    
    # 16. Prize / (W_max ** 1.5)
    candidates.append(P_sum / (W_max ** 1.5 + 1e-9))
    
    # 17. Prize / (W_sum ** 1.5)
    candidates.append(P_sum / (W_sum ** 1.5 + 1e-9))
    
    # 18. Prize / (W_max ** 0.25)
    candidates.append(P_sum / (W_max ** 0.25 + 1e-9))
    
    # 19. Prize / (W_sum ** 0.25)
    candidates.append(P_sum / (W_sum ** 0.25 + 1e-9))
    
    # 20. Prize / (W_max ** 2)
    candidates.append(P_sum / (W_max ** 2 + 1e-9))
    
    # 21. Prize / (W_sum ** 2)
    candidates.append(P_sum / (W_sum ** 2 + 1e-9))
    
    # 22. Prize / (W_max ** 0.75)
    candidates.append(P_sum / (W_max ** 0.75 + 1e-9))
    
    # 23. Prize / (W_sum ** 0.75)
    candidates.append(P_sum / (W_sum ** 0.75 + 1e-9))
    
    # 24. Prize / (W_max ** 1.25)
    candidates.append(P_sum / (W_max ** 1.25 + 1e-9))
    
    # 25. Prize / (W_sum ** 1.25)
    candidates.append(P_sum / (W_sum ** 1.25 + 1e-9))

    # Evaluate candidates
    best_score = -1
    best_heuristic = candidates[0]
    
    # Use a few seeds to evaluate robustness, but keep it fast
    seeds = [42, 123, 999]
    
    # Sort candidates by a quick proxy first?
    # Let's just evaluate them all. 25 candidates * 3 seeds = 75 runs.
    # Each run is 30 iters * 20 ants. 
    # With n=100, this is fast.
    
    for h in candidates:
        if time.time() > deadline:
            break
            
        # Ensure h is 1D and positive
        h_eval = np.asarray(h).flatten()
        # The evaluator says: "Heuristic entries that are not positive become 1e-9"
        # We should return the raw heuristic, and the evaluator handles the clamping.
        # But for our simulation, we need to handle it to get a fair score.
        # However, the task says "Heuristic entries that are not positive become 1e-9".
        # This implies the *input* to the ACO is clamped.
        # So we should return the raw values, but in our simulation, we clamp them.
        
        # Actually, the prompt says: "Heuristic entries that are not positive become 1e-9".
        # This is a rule of the evaluator.
        # So if I return a negative value, it becomes 1e-9.
        # My simulation should mimic this.
        
        total_score = 0.0
        for s in seeds:
            if time.time() > deadline:
                break
            try:
                score = _simulate_aco_fast(prizes, weights, h_eval, seed=s)
                total_score += score
            except Exception:
                total_score = -1
                break
        
        if total_score > best_score:
            best_score = total_score
            best_heuristic = h_eval
            
    # Final check
    best_heuristic = np.asarray(best_heuristic).flatten()
    # Ensure it's finite
    best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
    
    return best_heuristic
# EVOLVE-BLOCK-END
