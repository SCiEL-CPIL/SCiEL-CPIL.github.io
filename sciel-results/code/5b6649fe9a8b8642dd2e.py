"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a fast, deterministic, multi-start greedy simulation
    combined with bottleneck-slack feasibility scoring.
    
    Strategy:
    1. Perform multiple greedy passes with different item orderings (sorted by different criteria).
    2. In each pass, select items if they fit all constraints, tracking selection frequency.
    3. Weight selections by the total prize of the resulting solution.
    4. Compute a bottleneck-slack score: for each item, min_c (25 - w_ic) / w_ic, 
       which measures how much slack remains in the tightest constraint if this item is added first.
    5. Combine frequency-weighted counts with bottleneck-slack * prize to prioritize items 
       that are both frequently selected in high-value solutions AND fit easily into tight constraints.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shapes: prizes (5, 100) -> (100,), weights (5, 100)
    if prizes.ndim == 2:
        prizes = prizes.sum(axis=0)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    n_constraints = weights.shape[0]
    capacity = 25.0
    
    # Time guard: ensure we don't exceed time limits
    start_time = time.time()
    max_time = 290.0  # Safety margin before 300s limit
    
    # Compute bottleneck slack score: min_c (capacity - w_ic) / w_ic
    # This measures feasibility margin - higher means the item fits more easily
    # into the tightest constraint
    # Avoid division by zero
    safe_weights = np.maximum(weights, 1e-9)
    slack_ratios = (capacity - weights) / safe_weights
    # Handle negative values (items that don't fit at all) - they get very low slack
    slack_ratios = np.maximum(slack_ratios, 0.0)
    bottleneck_slack = slack_ratios.min(axis=0)  # shape (n_items,)
    
    # Selection counts weighted by solution quality
    weighted_counts = np.zeros(n_items, dtype=float)
    
    # Define several greedy strategies with different sorting criteria
    strategies = []
    
    # Precompute various weight metrics
    sum_w = weights.sum(axis=0)
    max_w = weights.max(axis=0)
    min_w = weights.min(axis=0)
    avg_w = weights.mean(axis=0)
    
    # Strategy 1: Sort by prize/sum_weight (original heuristic idea)
    ratio1 = prizes / (sum_w + 1e-9)
    order1 = np.argsort(-ratio1)
    strategies.append(order1)
    
    # Strategy 2: Sort by prize/max_weight (binding constraint focus)
    ratio2 = prizes / (max_w + 1e-9)
    order2 = np.argsort(-ratio2)
    strategies.append(order2)
    
    # Strategy 3: Sort by prize only
    order3 = np.argsort(-prizes)
    strategies.append(order3)
    
    # Strategy 4: Sort by prize/(max_weight^2) - sharper
    ratio4 = prizes / (max_w ** 2 + 1e-9)
    order4 = np.argsort(-ratio4)
    strategies.append(order4)
    
    # Strategy 5: Sort by prize/min_weight (if min weight is low, item is "cheap")
    ratio5 = prizes / (min_w + 1e-9)
    order5 = np.argsort(-ratio5)
    strategies.append(order5)
    
    # Strategy 6: Sort by average prize per unit weight
    ratio6 = prizes / (avg_w + 1e-9)
    order6 = np.argsort(-ratio6)
    strategies.append(order6)
    
    # Strategy 7: Sort by bottleneck_slack * prize (feasibility-aware)
    slack_prize = bottleneck_slack * prizes
    order7 = np.argsort(-slack_prize)
    strategies.append(order7)
    
    # Strategy 8: Random-like ordering based on hash of index (deterministic)
    rng = np.random.RandomState(42)
    order8 = np.array(rng.permutation(n_items))
    strategies.append(order8)
    
    # Strategy 9: Another random ordering
    rng2 = np.random.RandomState(123)
    order9 = np.array(rng2.permutation(n_items))
    strategies.append(order9)
    
    # Strategy 10: Sort by (bottleneck_slack + 1) * prize / (max_w + 1)
    combo10 = (bottleneck_slack + 1.0) * prizes / (max_w + 1.0)
    order10 = np.argsort(-combo10)
    strategies.append(order10)
    
    # Run greedy simulations for each strategy
    for order in strategies:
        if time.time() - start_time > max_time:
            break
            
        # Greedy selection
        remaining_capacity = np.full(n_constraints, capacity, dtype=float)
        selected = []
        total_prize = 0.0
        
        for idx in order:
            # Check if item fits in all constraints
            if np.all(weights[:, idx] <= remaining_capacity):
                remaining_capacity -= weights[:, idx]
                selected.append(idx)
                total_prize += prizes[idx]
        
        # Weight the selections by the total prize of this solution
        # This rewards items that are part of high-value solutions
        if total_prize > 0:
            for idx in selected:
                weighted_counts[idx] += total_prize
    
    # Normalize weighted_counts
    max_wc = weighted_counts.max()
    if max_wc > 0:
        wc_norm = weighted_counts / max_wc
    else:
        wc_norm = np.ones(n_items)
    
    # Normalize bottleneck_slack to [0, 1]
    max_slack = bottleneck_slack.max()
    if max_slack > 0:
        slack_norm = bottleneck_slack / max_slack
    else:
        slack_norm = np.ones(n_items)
    
    # Normalize prizes to [0, 1]
    max_prize = prizes.max()
    if max_prize > 0:
        prize_norm = prizes / max_prize
    else:
        prize_norm = np.ones(n_items)
    
    # Combine: emphasize both frequency of selection AND feasibility slack AND prize
    # Use a product form with powers to sharpen
    # freq_norm^1.5 * slack_norm^1.0 * prize_norm^1.0
    scores = (wc_norm ** 1.5) * (slack_norm ** 1.0) * (prize_norm ** 1.0)
    
    # Add a small base to avoid zeros
    base = 1e-9
    scores = scores + base
    
    # Normalize to have similar scale
    if scores.max() > 0:
        scores = scores / scores.max()
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
