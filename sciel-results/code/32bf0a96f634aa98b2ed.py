"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 300.0  # seconds wall-clock limit


def heuristics(prizes, weights):
    t0 = time.time()
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Ensure positive values to avoid division issues
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)

    # Per-constraint efficiency: prize/weight for each constraint
    # Shape: (5, 100)
    ratios = prizes / weights

    # Log of ratios for smooth operations
    log_ratios = np.log(ratios)

    # Soft-min across constraints (smooth surrogate for min-ratio)
    # Using log-sum-exp trick: softmin(x) = -logsumexp(-x)
    # We want to penalize items that are poor on any single constraint
    neg_log_ratios = -log_ratios
    max_neg = neg_log_ratios.max(axis=0)
    log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0))
    softmin_log = -log_sum_exp_neg  # This is log(geometric mean of ratios) = softmin of log ratios

    # Geometric mean of ratios (smooth version via log-sum-exp with beta=1)
    # GM = exp(mean(log(ratios)))
    log_gm = log_ratios.mean(axis=0)

    # Total prize per item, normalized
    total_prize = prizes.sum(axis=0)
    max_prize = total_prize.max()
    norm_prize = total_prize / (max_prize + 1e-9)

    # Combine: softmin (penalizes worst constraint) * GM * normalized prize
    # Use exp to get back to ratio scale for softmin component
    softmin_ratio = np.exp(softmin_log)
    gm_ratio = np.exp(log_gm)

    # Bottleneck slack: minimum remaining capacity ratio across all constraints
    # Capacity is 25 for each constraint
    capacity = 25.0
    # For each item, compute (capacity - weight) / weight for each constraint
    # This measures how much "room" remains after adding this item
    slack = (capacity - weights) / weights  # Shape: (5, 100)
    # The bottleneck slack is the minimum across constraints
    bottleneck_slack = slack.min(axis=0)  # Shape: (100,)
    # Normalize bottleneck slack to [0, 1] range for blending
    min_slack = bottleneck_slack.min()
    max_slack = bottleneck_slack.max()
    if max_slack > min_slack:
        norm_slack = (bottleneck_slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.ones_like(bottleneck_slack)

    # --- Co-occurrence penalty via greedy packing ---
    # Perform a single greedy packing to identify which items conflict
    # with high-value selections, then penalize items that share
    # high weight correlation with the top-selected items.
    
    # Greedy packing: sort by total prize / max weight across constraints
    max_weight_per_item = weights.max(axis=0)  # (100,)
    greedy_score = total_prize / (max_weight_per_item + 1e-9)
    order = np.argsort(greedy_score)[::-1]
    
    # Track remaining capacity
    remaining = np.full(5, capacity)
    selected = []
    rejected = set()
    
    for idx in order:
        if np.all(weights[:, idx] <= remaining):
            remaining -= weights[:, idx]
            selected.append(idx)
        else:
            rejected.add(idx)
    
    selected = np.array(selected) if selected else np.array([], dtype=int)
    
    # Compute weight correlation matrix (100x100)
    # Correlation of weight vectors across the 5 constraints
    if len(selected) > 0:
        # Compute pairwise correlation of weight vectors
        # weights shape: (5, 100), so each item is a 5-d vector
        # Center each item's weight vector
        W = weights.T  # (100, 5)
        W_centered = W - W.mean(axis=1, keepdims=True)
        
        # Compute correlation matrix
        norms = np.linalg.norm(W_centered, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        W_normed = W_centered / norms
        corr_matrix = W_normed @ W_normed.T  # (100, 100)
        
        # For each item, compute max correlation with any selected item
        # This measures how much it "conflicts" with already-chosen items
        if len(selected) > 0:
            sel_corr = corr_matrix[:, selected]  # (100, len(selected))
            max_sel_corr = sel_corr.max(axis=1)  # (100,)
        else:
            max_sel_corr = np.zeros(100)
        
        # Normalize correlation to [0, 1] for penalty
        max_corr_val = max_sel_corr.max()
        if max_corr_val > 1e-9:
            norm_corr = max_sel_corr / max_corr_val
        else:
            norm_corr = np.zeros(100)
        
        # Items that were rejected get a stronger penalty if they correlate
        # with selected items (they were blocked due to resource conflicts)
        rejection_mask = np.zeros(100, dtype=bool)
        for idx in rejected:
            rejection_mask[idx] = True
        
        # Co-occurrence penalty: penalize items that both (a) correlate
        # highly with selected items AND (b) were rejected (blocked)
        # This discourages ants from choosing conflicting pairs
        penalty = 1.0 - 0.3 * norm_corr * rejection_mask.astype(float)
        # Also apply a mild penalty to all items that correlate with selected
        penalty = penalty * (1.0 - 0.1 * norm_corr)
    else:
        penalty = np.ones(100)

    # Ensure penalty is positive
    penalty = np.maximum(penalty, 0.1)

    # Final heuristic: blend of softmin efficiency, geometric mean efficiency, prize, slack, and co-occurrence penalty
    heuristic = softmin_ratio * gm_ratio * (0.5 + 0.5 * norm_prize) * (0.5 + 0.5 * norm_slack) * penalty

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Deadline guard
    if time.time() - t0 > _DEADLINE - 5.0:
        # Return a simple fallback
        return prizes.sum(axis=0) / (weights.sum(axis=0) + 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
