import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Marginal Opportunity Cost heuristic.
    
    The heuristic estimates the value of an item relative to the opportunity cost
    of the resources it consumes. The opportunity cost for a resource is estimated
    by the maximum efficiency (prize/weight) observed for that resource across all items.
    """
    # Ensure inputs are float arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate efficiency for each item in each constraint
    # Avoid division by zero
    eps = 1e-9
    efficiencies = prizes / (weights + eps)
    
    # Find the maximum efficiency for each constraint (resource)
    # This represents the "best alternative" or shadow price for each resource
    max_eff_per_constraint = np.max(efficiencies, axis=1)
    
    # Calculate the opportunity cost for each item
    # For each item j, the cost is sum over constraints k of weight[k,j] * max_eff[k]
    # weights is (5, 100), max_eff_per_constraint is (5,)
    # We need to multiply each row of weights by the corresponding max_eff and sum along axis 0
    opportunity_costs = np.sum(weights * max_eff_per_constraint[:, np.newaxis], axis=0)
    
    # Calculate total prize for each item
    total_prizes = np.sum(prizes, axis=0)
    
    # Heuristic score: Total Prize / Opportunity Cost
    # This rewards items that have high prize relative to the "value" of the resources they take up
    scores = total_prizes / (opportunity_costs + eps)
    
    # Ensure all scores are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
