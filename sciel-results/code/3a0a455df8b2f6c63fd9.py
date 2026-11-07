"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are expected to be shape (5, 100)
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    
    # 1. Calculate Dominance Relationships
    # An item i dominates item j if:
    # prizes[:, i] >= prizes[:, j] AND weights[:, i] <= weights[:, j]
    # AND (prizes[:, i] > prizes[:, j] OR weights[:, i] < weights[:, j])
    
    # To speed up, we can use broadcasting or loops. N=100 is small.
    # Let's create a matrix of dominance counts.
    domination_count = np.zeros(n_items, dtype=int)
    
    # Pre-compute some comparisons
    # prizes is (C, N), weights is (C, N)
    
    # For each item i, count how many items j it dominates? 
    # Actually, we need for each item i, how many items j dominate i.
    # Let's iterate.
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            
            # Check if j dominates i
            # j has >= prize and <= weight in all constraints
            prize_ge = np.all(prizes[:, j] >= prizes[:, i])
            weight_le = np.all(weights[:, j] <= weights[:, i])
            
            if prize_ge and weight_le:
                # Check strict inequality in at least one
                # If they are identical, they don't strictly dominate each other in standard Pareto, 
                # but for heuristic purposes, if they are identical, they are equivalent.
                # Usually, if p_j == p_i and w_j == w_i, neither dominates the other.
                if np.any(prizes[:, j] > prizes[:, i]) or np.any(weights[:, j] < weights[:, i]):
                    domination_count[i] += 1
                    
    # 2. Assign Ranks (Fronts)
    # Rank 1: domination_count == 0
    # Rank 2: domination_count > 0, but all dominators are Rank 1
    # ...
    
    ranks = np.zeros(n_items, dtype=int)
    assigned = np.zeros(n_items, dtype=bool)
    
    # Simple iterative front assignment
    current_front = 1
    # We need to know who dominates whom to remove dominators.
    # Let's build a list of dominators for each item.
    dominators = [[] for _ in range(n_items)]
    
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            prize_ge = np.all(prizes[:, j] >= prizes[:, i])
            weight_le = np.all(weights[:, j] <= weights[:, i])
            if prize_ge and weight_le:
                if np.any(prizes[:, j] > prizes[:, i]) or np.any(weights[:, j] < weights[:, i]):
                    dominators[i].append(j)
                    
    # Now assign ranks
    while not np.all(assigned):
        for i in range(n_items):
            if assigned[i]:
                continue
            # If all dominators are assigned, i is in the next front
            if all(assigned[j] for j in dominators[i]):
                ranks[i] = current_front
                assigned[i] = True
        current_front += 1
        if current_front > n_items: break # Safety

    # 3. Calculate Heuristic Score
    # Base score from rank: Lower rank is better.
    # rank_score = 1 / (rank + 1)
    
    # Secondary score: Density
    epsilon = 1e-9
    densities = prizes / (weights + epsilon)
    # Sum densities across constraints
    total_density = densities.sum(axis=0)
    
    # Normalize density to [0, 1] roughly? Or just use raw?
    # Raw density can vary widely. Let's normalize by max density.
    max_density = total_density.max()
    if max_density > 0:
        norm_density = total_density / max_density
    else:
        norm_density = np.zeros(n_items)
        
    # Combine: 
    # Items in Front 1 (rank 1) should have high score.
    # Items in Front 2 lower, etc.
    # Within a front, higher density is better.
    
    # Score = RankWeight * (1/rank) + DensityWeight * norm_density
    # Let's use a strong preference for lower rank.
    rank_score = 1.0 / (ranks + 1.0)
    
    # To ensure Front 1 items always beat Front 2 items, the rank component must dominate the density component.
    # Max rank_score for Front 1 is 1.0. Min for Front 1 is 1.0.
    # Max rank_score for Front 2 is 0.5.
    # If density is in [0,1], we can do:
    # Score = 10 * rank_score + norm_density
    # This ensures Front 1 (10..11) > Front 2 (5..6) > Front 3 (3.3..4.3)
    
    scores = 10.0 * rank_score + norm_density
    
    # Ensure positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
