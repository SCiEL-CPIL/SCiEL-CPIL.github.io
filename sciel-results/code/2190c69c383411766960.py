import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    Uses a local mini-ACO simulation to select the best heuristic from a set of candidates.
    """
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    eps = 1e-9
    
    # Candidate Heuristics
    # 1. Baseline: Prize / Sum(Weights)
    h1 = prizes / (weights.sum(axis=0) + eps)
    
    # 2. Bottleneck: Prize / Max(Weight)
    h2 = prizes / (weights.max(axis=0) + eps)
    
    # 3. L2 Norm: Prize / Norm(Weights)
    h3 = prizes / (np.sqrt((weights**2).sum(axis=0)) + eps)
    
    # 4. Geometric Mean: Prize / Geomean(Weights)
    # Geomean = exp(mean(log(w)))
    # Handle zeros with eps
    log_w = np.log(weights + eps)
    geomean_w = np.exp(log_w.mean(axis=0))
    h4 = prizes / (geomean_w + eps)
    
    # 5. Weighted Sum by Constraint Variance? Or just a different power.
    # Let's try Prize / (Sum(Weights) ** 0.5)
    h5 = prizes / (np.sqrt(weights.sum(axis=0)) + eps)
    
    candidates = [h1, h2, h3, h4, h5]
    
    def mini_aco_score(prizes, weights, heuristic, seed=42):
        np.random.seed(seed)
        n_items = weights.shape[1]
        tau = np.ones(n_items)
        best_total = 0.0
        n_ants = 10
        n_iters = 5
        
        for _ in range(n_iters):
            iteration_best = 0.0
            for _ in range(n_ants):
                rem_cap = np.full(n_constraints, capacity)
                total_prize = 0.0
                available = np.arange(n_items)
                
                while len(available) > 0:
                    # Check which items fit
                    # weights[available] is shape (C, N_avail)
                    # rem_cap is shape (C,)
                    # We need weights[available][j, i] <= rem_cap[j] for all j
                    fits = np.all(weights[:, available] <= rem_cap[:, np.newaxis], axis=0)
                    candidates_idx = available[fits]
                    
                    if len(candidates_idx) == 0:
                        break
                    
                    # Calculate probabilities
                    # heuristic[candidates_idx] is shape (N_cand,)
                    # tau[candidates_idx] is shape (N_cand,)
                    probs = tau[candidates_idx] * heuristic[candidates_idx]
                    sum_probs = np.sum(probs)
                    if sum_probs < eps:
                        break
                    probs /= sum_probs
                    
                    # Select item
                    choice_idx = np.random.choice(candidates_idx, p=probs)
                    
                    # Update state
                    rem_cap -= weights[:, choice_idx]
                    total_prize += prizes[choice_idx]
                    available = available[available != choice_idx]
                
            if total_prize > iteration_best:
                iteration_best = total_prize
            
            if iteration_best > best_total:
                best_total = iteration_best
                
            # Update pheromone
            tau = tau * 0.9 + iteration_best / 10.0
            
        return best_total

    best_score = -1.0
    best_heuristic = h1
    
    for h in candidates:
        # Ensure heuristic is positive and finite
        h_clean = np.where(np.isfinite(h) & (h > 0), h, eps)
        score = mini_aco_score(prizes, weights, h_clean)
        if score > best_score:
            best_score = score
            best_heuristic = h_clean
            
    return best_heuristic
