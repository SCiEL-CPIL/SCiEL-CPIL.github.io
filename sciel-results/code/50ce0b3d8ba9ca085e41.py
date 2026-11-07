"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Uses a scarcity-weighted efficiency metric with smooth bottleneck penalty.
    1. Estimate scarcity of each constraint based on total weight demand 
       relative to the fixed capacity (25).
    2. Compute a composite "cost" for each item, weighted by constraint scarcity.
    3. Calculate base efficiency as Prize / Composite Cost.
    4. Apply a soft-min bottleneck penalty via log-sum-exp to smoothly penalize 
       items that are inefficient in the tightest constraints.
    5. Rank-normalize the final scores to be robust to outliers and ensure 
       positive values.
    """
    eps = 1e-9
    capacity = 25.0
    
    # Handle potential shape mismatch defensively
    if prizes.shape != (100,):
        if prizes.shape == (5, 100):
            prizes = prizes.sum(axis=0)
        else:
            prizes = prizes.flatten()
            
    if weights.shape != (5, 100):
        if weights.shape == (100, 5):
            weights = weights.T
        else:
            weights = weights.reshape((5, 100))
            
    n_constraints = 5
    n_items = 100
    
    # 1. Scarcity (5,)
    total_weights_per_constraint = weights.sum(axis=1)
    scarcity = total_weights_per_constraint / capacity
    scarcity = np.maximum(scarcity, eps)
    
    # 2. Composite Cost (100,)
    composite_costs = np.dot(scarcity, weights)
    
    # 3. Base efficiency (100,)
    base_efficiencies = prizes / (composite_costs + eps)
    
    # 4. Per-constraint efficiency ratios (5, 100)
    # For each constraint k and item i: prize_i / weight_ki
    # This captures how well an item performs in each specific constraint
    per_constraint_eff = np.zeros((n_constraints, n_items))
    for k in range(n_constraints):
        per_constraint_eff[k] = prizes / (weights[k] + eps)
    
    # 5. Soft-min bottleneck penalty via log-sum-exp
    # We want to penalize items that have a low minimum efficiency across constraints
    # Use log-sum-exp to create a smooth approximation of the minimum
    # The minimum of x is approximated by -log-sum-exp(-x / temperature) * temperature
    # Lower temperature -> closer to true minimum
    temperature = 0.1
    
    # Work with negative efficiencies for the soft-min
    neg_eff = -per_constraint_eff
    
    # Subtract max for numerical stability
    neg_eff_max = np.max(neg_eff, axis=0, keepdims=True)  # (1, 100)
    neg_eff_shifted = neg_eff - neg_eff_max  # (5, 100)
    
    # Log-sum-exp along constraint axis
    lse = np.log(np.sum(np.exp(neg_eff_shifted / temperature), axis=0)) + neg_eff_max[0]
    
    # Soft-min is the negative of this
    soft_min_eff = -lse  # (100,)
    
    # 6. Combine base efficiency with soft-min bottleneck penalty
    # Weight the two components
    w_base = 0.7
    w_bottleneck = 0.3
    
    # Normalize both components to [0, 1] range for fair combination
    base_min = np.min(base_efficiencies)
    base_max = np.max(base_efficiencies)
    base_norm = (base_efficiencies - base_min) / (base_max - base_min + eps)
    
    sm_min = np.min(soft_min_eff)
    sm_max = np.max(soft_min_eff)
    sm_norm = (soft_min_eff - sm_min) / (sm_max - sm_min + eps)
    
    combined = w_base * base_norm + w_bottleneck * sm_norm
    
    # 7. Rank Normalize
    ranks = np.argsort(np.argsort(combined)).astype(float) + 1.0
    scores = ranks / (ranks.mean() + eps)
    
    # Ensure positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
