"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ant colony optimization of the
    multiple-constraint knapsack problem.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one attractiveness score per item
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Validate shapes
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Invalid input shapes")
    
    # Check for non-finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        raise ValueError("Non-finite values in input")
    
    # Total prize per item (sum across constraints/dimensions)
    total_prize = prizes.sum(axis=0)  # shape (100,)
    
    # Total weight per item (sum across constraints/dimensions)
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Capacity per constraint is 25, so total capacity across 5 constraints is 125
    # But each constraint has its own capacity of 25
    
    # We want to prioritize items with high prize-to-weight ratio
    # But also consider that an item must fit in ALL constraints
    
    # For each constraint, the maximum weight an item can have is 25
    # Items with very high weight in any single constraint are less likely to fit
    
    # Compute the minimum remaining capacity factor per item per constraint
    # If weights[i, j] is close to 25, that item is risky for constraint j
    
    # A better heuristic: normalize prize by the maximum weight across constraints
    # This penalizes items that are heavy in any single constraint
    
    max_weight_per_item = weights.max(axis=0)  # shape (100,), max weight across constraints
    
    # Prize-to-max-weight ratio: high prize, low max weight is good
    ratio = total_prize / (max_weight_per_item + 1e-9)
    
    # Also consider the sum of weights: items with low total weight are more flexible
    # Combine: use geometric mean or weighted combination
    
    # Let's use a combined score that rewards:
    # 1. High total prize
    # 2. Low maximum weight per constraint (fits easily)
    # 3. Low total weight (flexible)
    
    # Normalize each component to [0, 1]
    prize_norm = total_prize / (total_prize.max() + 1e-9)
    max_w_norm = max_weight_per_item / (max_weight_per_item.max() + 1e-9)
    total_w_norm = total_weight / (total_weight.max() + 1e-9)
    
    # Score: high prize, low max weight, low total weight
    # Use: prize_norm * (1 - max_w_norm) * (1 - total_w_norm) + small epsilon
    # But this can be zero for some items. Let's use a multiplicative approach with floor.
    
    # Alternative: log-scale ratio
    # score = total_prize / (max_weight_per_item * total_weight + 1e-9)
    # This strongly penalizes items that are heavy in any constraint
    
    # Let's try: prize / (1 + max_weight_per_item) * (1 / (1 + total_weight))
    # This gives high score to high-prize, low-weight items
    
    score = total_prize / (1.0 + max_weight_per_item)
    score = score * (1.0 / (1.0 + total_weight))
    
    # Ensure all scores are positive
    score = np.maximum(score, 1e-9)
    
    # Normalize to a reasonable range
    score = score / (score.max() + 1e-9)
    
    # Add a small constant to ensure positivity and avoid extreme ratios
    score = score + 0.1
    
    return score
# EVOLVE-BLOCK-END
