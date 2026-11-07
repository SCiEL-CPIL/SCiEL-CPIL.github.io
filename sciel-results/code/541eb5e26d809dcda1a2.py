import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Competitive Exclusion / Opportunity Cost
    The heuristic is the total prize of the item divided by its relative resource consumption.
    Relative resource consumption is the sum, over all constraints, of the item's weight
    normalized by the total weight of all items in that constraint. This estimates the
    opportunity cost of selecting the item (how much of the available resource pool it uses).
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Handle potential shape mismatches or empty inputs gracefully
    if prizes.shape != weights.shape:
        # Fallback to simple density if shapes don't match as expected (5, N)
        # Though task guarantees (5, 100)
        return np.ones(prizes.shape[1]) if prizes.ndim > 1 else np.ones(prizes.shape[0])

    # Assume shape is (C, N) where C=5, N=100
    # Calculate total prize per item. 
    # "Total prize" implies summing across the constraint/objective dimensions.
    total_prize = prizes.sum(axis=0)
    
    # Calculate total weight per constraint (sum across items)
    # weights shape (C, N) -> sum(axis=1) gives (C,)
    total_weight_per_constraint = weights.sum(axis=1)
    
    eps = 1e-9
    
    # Avoid division by zero for constraints with 0 total weight (unlikely with uniform [0,1])
    total_weight_per_constraint_safe = total_weight_per_constraint + eps
    
    # Normalize weights: w_{c,i} / W_c
    # weights (C, N) / (C, 1) -> (C, N)
    normalized_weights = weights / total_weight_per_constraint_safe[:, None]
    
    # Sum normalized weights per item to get "relative cost"
    # (C, N) -> (N,)
    relative_cost = normalized_weights.sum(axis=0)
    
    # Heuristic: Prize / Relative Cost
    # Higher prize and lower relative cost -> Higher heuristic
    h = total_prize / (relative_cost + eps)
    
    # Ensure values are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We should ensure our output respects this or is naturally positive.
    # Since prizes and weights are in [0,1], sums are positive.
    # However, if a prize is 0, h is 0. The ACO loop will replace 0 with 1e-9.
    # We can pre-clip to avoid 0s if we want, but the ACO logic handles it.
    # "non-finite values ... make the whole result invalid".
    h = np.nan_to_num(h, nan=1e-9, posinf=1e9, neginf=1e-9)
    h = np.clip(h, 1e-9, 1e9)
    
    return h
