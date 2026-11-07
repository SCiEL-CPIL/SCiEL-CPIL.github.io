import numpy as np

def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy:
    Estimates the 'opportunity cost' of each resource by calculating the mean efficiency
    (prize/weight) of the top-K most efficient items for that resource.
    Items that consume resources with high opportunity cost are penalized more heavily
    in the heuristic score, encouraging the selection of items that are efficient across
    all constraints rather than just having low total weight.
    
    Args:
    prizes: np.array of shape (5, 100), uniform in [0, 1].
    weights: np.array of shape (5, 100), uniform in [0, 1].
    
    Returns:
    np.array of shape (100,), heuristic scores for each item.
    """
    # Ensure weights are positive to avoid division by zero in efficiency calculation
    w_safe = np.maximum(weights, 1e-6)
    
    # Calculate efficiency for each item and constraint: p_i / w_ij
    # Shape: (5, 100)
    efficiencies = prizes / w_safe
    
    # Determine the 'price' or scarcity of each resource (constraint)
    # We use the mean efficiency of the top 10 items for each constraint.
    # This represents the best alternative use of that resource.
    K = 10
    # np.partition is O(N) on average, faster than full sort
    # We want the largest K elements, so we partition at index -K
    # The elements from -K to end are the top K (unsorted)
    top_effs = np.partition(efficiencies, -K, axis=1)[:, -K:]
    
    # Mean efficiency of the top K items for each constraint
    # Shape: (5,)
    resource_prices = np.mean(top_effs, axis=1)
    
    # Calculate the total opportunity cost for each item
    # Cost of item i = sum_j (w_ij * price_j)
    # weights: (5, 100), resource_prices: (5, 1)
    weighted_costs = weights * resource_prices[:, np.newaxis]
    total_cost = np.sum(weighted_costs, axis=0)
    
    # Heuristic score: Prize / Total Opportunity Cost
    # Items with high prize and low opportunity cost are more attractive
    heuristic = prizes / (total_cost + 1e-9)
    
    # Ensure all heuristic values are positive as required by the task
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
