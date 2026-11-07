import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    1. Calculate total prize per item (sum across constraints).
    2. Calculate "scarcity" of each constraint (total weight available).
    3. Adjust item weights by scarcity to penalize usage of tight constraints.
    4. Compute efficiency ratios (Prize / Adjusted Weight) for each constraint.
    5. Blend geometric mean and minimum of log-ratios to balance overall efficiency
       and bottleneck constraint awareness.
    6. Rank-normalize the final scores for robustness.
    """
    eps = 1e-9
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit

    # 1. Total prize per item: shape (100,)
    # Assuming the "total prize" is the sum of prizes across all 5 rows.
    # If the problem meant row 0 is prize and others are weights, the shape would likely be (1, 100) for prizes.
    # Given shape (5, 100) for both, summing is the most logical interpretation of "total prize".
    total_prizes = np.sum(prizes, axis=0)
    
    # 2. Scarcity per constraint: shape (5,)
    # Total weight in each constraint dimension.
    total_weights = np.sum(weights, axis=1)
    
    # 3. Adjusted weights: shape (5, 100)
    # Multiply each weight by the total weight of its constraint.
    # This scales the "cost" of weight in a tight constraint (high total_weights) to be higher.
    # w_adj[i, j] = w[i, j] * total_weights[i]
    adjusted_weights = weights * (total_weights + eps)[:, np.newaxis]
    
    # 4. Efficiency ratios: shape (5, 100)
    # ratio[i, j] = total_prizes[j] / adjusted_weights[i, j]
    # This measures how much total prize you get per unit of "scarcity-weight" in constraint i.
    ratios = total_prizes[np.newaxis, :] / (adjusted_weights + eps)
    
    # 5. Log-space combination
    # Clip to avoid log of 0 or negative (ratios should be positive)
    log_ratios = np.log(np.maximum(ratios, eps))
    
    # Geometric mean in log-space (mean of logs)
    log_geo_mean = np.mean(log_ratios, axis=0)
    
    # Minimum in log-space (min of logs) - captures the bottleneck constraint
    log_min = np.min(log_ratios, axis=0)
    
    # Blend: 0.6 weight on geometric mean, 0.4 on minimum
    # Increasing min weight slightly from 0.35 to 0.4 to better handle tight bottlenecks
    w_geo = 0.6
    w_min = 0.4
    log_score = w_geo * log_geo_mean + w_min * log_min
    
    # Exponentiate to get raw scores
    raw_scores = np.exp(log_score)
    
    # Rank-based normalization for robustness
    # This ensures the heuristic is not dominated by outliers in raw magnitude
    ranks = np.argsort(np.argsort(raw_scores)).astype(float)
    ranks = ranks + 1.0  # Shift to 1..n
    scores = ranks / (ranks.mean() + eps)
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
