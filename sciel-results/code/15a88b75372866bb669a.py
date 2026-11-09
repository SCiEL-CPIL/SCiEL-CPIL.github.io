"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _simulate_aco(prizes, weights, heuristic, n_ants=20, n_iters=30, seed=42):
    """Run a simplified ACO simulation to estimate the quality of a heuristic."""
    rng = np.random.default_rng(seed)
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Heuristic: non-positive becomes 1e-9
    h = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Pheromone starts at 1
    pheromone = np.ones(n_items)
    
    best_total_prize = 0.0
    
    for it in range(n_iters):
        iter_best = 0.0
        
        for ant in range(n_ants):
            remaining_cap = np.full(n_constraints, capacity)
            total_prize = 0.0
            selected = set()
            
            # Greedy with stochastic selection
            while True:
                # Check which items still fit
                fits = np.ones(n_items, dtype=bool)
                for i in range(n_items):
                    if i in selected:
                        fits[i] = False
                    else:
                        for c in range(n_constraints):
                            if weights[c, i] > remaining_cap[c]:
                                fits[i] = False
                                break
                
                if not np.any(fits):
                    break
                
                # Probability proportional to pheromone * heuristic
                probs = pheromone * h
                probs[~fits] = 0.0
                total_prob = probs.sum()
                if total_prob <= 0:
                    break
                
                probs /= total_prob
                
                # Stochastic selection with some greediness
                if rng.random() < 0.3:
                    # Greedy: pick best prize/weight ratio among fitting items
                    fitting_indices = np.where(fits)[0]
                    ratios = prizes[0, fitting_indices] / (weights.sum(axis=0)[fitting_indices] + 1e-9)
                    best_idx = fitting_indices[np.argmax(ratios)]
                else:
                    best_idx = rng.choice(n_items, p=probs)
                
                selected.add(best_idx)
                total_prize += prizes[0, best_idx]
                remaining_cap -= weights[:, best_idx]
            
            if total_prize > iter_best:
                iter_best = total_prize
        
        # Update pheromone
        pheromone *= 0.9
        pheromone += iter_best / 10.0
        
        if iter_best > best_total_prize:
            best_total_prize = iter_best
    
    return best_total_prize


def heuristics(prizes, weights):
    """Compute attractiveness scores for items in the ACO knapsack problem."""
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s limit
    
    n_constraints, n_items = weights.shape
    
    # Generate multiple diverse heuristic candidates
    candidates = []
    
    # 1. Prize-to-total-weight ratio (baseline)
    total_weight = weights.sum(axis=0)
    candidates.append(prizes[0] / (total_weight + 1e-9))
    
    # 2. Prize-to-max-weight ratio
    max_weight = weights.max(axis=0)
    candidates.append(prizes[0] / (max_weight + 1e-9))
    
    # 3. Prize-to-min-weight ratio
    min_weight = weights.min(axis=0)
    candidates.append(prizes[0] / (min_weight + 1e-9))
    
    # 4. Pure prize
    candidates.append(prizes[0].copy())
    
    # 5. Prize-to-geometric-mean-weight
    geo_mean_weight = np.exp(np.mean(np.log(weights + 1e-9), axis=0))
    candidates.append(prizes[0] / (geo_mean_weight + 1e-9))
    
    # 6. Prize-to-std-weight (penalize high variance)
    std_weight = weights.std(axis=0)
    candidates.append(prizes[0] / (std_weight + 1e-9))
    
    # 7. Prize * (1 / (1 + total_weight))
    candidates.append(prizes[0] / (1.0 + total_weight))
    
    # 8. Prize / (1 + max_weight)
    candidates.append(prizes[0] / (1.0 + max_weight))
    
    # 9. Normalized prize
    norm_prize = (prizes[0] - prizes[0].min()) / (prizes[0].max() - prizes[0].min() + 1e-9)
    candidates.append(norm_prize)
    
    # 10. Prize-to-weight with per-constraint normalization
    per_constraint_ratio = np.zeros(n_items)
    for c in range(n_constraints):
        per_constraint_ratio += prizes[c] / (weights[c] + 1e-9)
    candidates.append(per_constraint_ratio / n_constraints)
    
    # 11. Min over constraints of prize/weight
    min_ratio = np.ones(n_items) * np.inf
    for c in range(n_constraints):
        min_ratio = np.minimum(min_ratio, prizes[c] / (weights[c] + 1e-9))
    candidates.append(min_ratio)
    
    # 12. Geometric mean of per-constraint ratios
    log_ratio_sum = np.zeros(n_items)
    for c in range(n_constraints):
        log_ratio_sum += np.log(prizes[c] / (weights[c] + 1e-9) + 1e-9)
    candidates.append(np.exp(log_ratio_sum / n_constraints))
    
    # 13. Prize-to-weight with capacity awareness
    # Items that are "tight" relative to capacity get bonus
    tightness = total_weight / capacity  # how much of capacity this item uses
    candidates.append(prizes[0] / (total_weight + 1e-9) * (1.0 + 0.5 * (1.0 - tightness)))
    
    # 14. Prize^alpha / weight^beta with different exponents
    candidates.append(prizes[0] ** 1.5 / (total_weight + 1e-9))
    candidates.append(prizes[0] / (total_weight + 1e-9) ** 1.5)
    candidates.append(prizes[0] ** 0.5 / (total_weight + 1e-9))
    
    # 15. Composite: blend of prize ratio and constraint balance
    balance = 1.0 - std_weight / (max_weight + 1e-9)  # higher if weights are balanced
    candidates.append(prizes[0] / (total_weight + 1e-9) * (1.0 + balance))
    
    # Evaluate candidates using reduced-fidelity ACO simulation
    best_score = -1.0
    best_candidate = candidates[0]
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        
        # Use reduced fidelity for speed: fewer ants and iterations
        score = _simulate_aco(prizes, weights, cand, n_ants=10, n_iters=15, seed=42)
        
        if score > best_score:
            best_score = score
            best_candidate = cand
    
    # If we have time, do a full-fidelity evaluation of the best candidate
    if time.time() < deadline - 5.0:
        full_score = _simulate_aco(prizes, weights, best_candidate, n_ants=20, n_iters=30, seed=42)
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(best_candidate) & (best_candidate > 0), best_candidate, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
