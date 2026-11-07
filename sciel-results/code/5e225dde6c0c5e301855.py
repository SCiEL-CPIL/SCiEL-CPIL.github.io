"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for ACO on multiple-constraint knapsack.
    
    Strategy:
    Combines static efficiency metrics with a dynamic marginal utility/scarcity heuristic.
    
    1. Aggregate Efficiency: Prize per total weight.
       Captures overall resource usage.
    2. Bottleneck Efficiency: Geometric mean of per-constraint prize/weight ratios.
       Captures the constraint that limits the item the most (smooth min).
    3. Scarcity-Adjusted Marginal Utility:
       Estimates the opportunity cost of each item by computing the weighted sum
       of the top-k most efficient items for each constraint. Items that consume
       resources which are already "scarce" (highly contended by other high-value items)
       are penalized. This encourages the heuristic to favor items that are efficient
       relative to the current scarcity of the specific resources they consume.
    
    All metrics are normalized to have a mean of 1 across items to ensure
    they contribute equally to the final heuristic score.
    """
    eps = 1e-9
    n_items = prizes.shape[1]
    
    # 1. Aggregate Efficiency: prizes / sum(weights)
    # Shape: (100,)
    total_weights = weights.sum(axis=0)
    sum_ratio = prizes / (total_weights + eps)
    
    # 2. Bottleneck Efficiency: Geometric Mean of (prizes / weights)
    # Shape: (5, 100)
    ratios = prizes / (weights + eps)
    
    # Geometric mean in log-space for stability
    log_ratios = np.log(ratios)
    log_geo_mean = log_ratios.mean(axis=0)
    geo_ratio = np.exp(log_geo_mean)
    
    # 3. Scarcity-Adjusted Marginal Utility
    # For each constraint, we identify the "scarcity" of that resource.
    # A resource is scarce if the top items consuming it have high efficiency.
    # We approximate the opportunity cost of an item by looking at how "expensive"
    # its weight is relative to the best available alternatives for that constraint.
    
    # For each constraint c, let E_c[i] = prizes[c, i] / weights[c, i]
    # We want to penalize item i if it consumes weight in constraint c where
    # other items have much higher efficiency.
    
    # A simple proxy for scarcity/competition:
    # For each constraint, sort items by efficiency. The "value" of weight in that
    # constraint is high if the top items are very efficient.
    
    # Let's compute a "scarcity score" for each constraint.
    # scarcity[c] = mean of the top 10% efficiencies in constraint c.
    # This represents the "market rate" for that resource.
    
    top_k = max(1, n_items // 10)
    scarcity_scores = np.zeros(prizes.shape[0])
    for c in range(prizes.shape[0]):
        eff_c = ratios[c, :]
        # Top k efficiencies
        top_effs = np.sort(eff_c)[-top_k:]
        scarcity_scores[c] = np.mean(top_effs)
    
    # Now, for each item, we compute a "scarcity penalty" or "marginal utility".
    # If an item has a low efficiency in a constraint that has high scarcity,
    # it is a poor choice because it uses a valuable resource inefficiently.
    # If an item has high efficiency in a scarce constraint, it is a great choice.
    
    # We can model the marginal utility as:
    # MU[i] = sum_c ( efficiency[i, c] / scarcity[c] )
    # This normalizes the efficiency by the "market rate" of that resource.
    # An item with efficiency 1.0 in a scarce resource (scarcity 2.0) gets 0.5,
    # while an item with efficiency 1.0 in a common resource (scarcity 0.5) gets 2.0.
    # Wait, this rewards items in COMMON resources. We want to reward items that
    # are GOOD at using SCARCE resources.
    
    # Let's rethink. The goal is to pick items that give high prize for low weight.
    # If a resource is scarce, weight is "expensive". So we should prioritize items
    # that have HIGH efficiency in scarce resources.
    
    # Alternative:
    # Compute a "resource value" vector v[c] = scarcity_scores[c].
    # The "cost" of item i in terms of opportunity loss is sum_c (weights[i, c] * v[c]).
    # The "benefit" is prizes[i, :].
    # So the "net value" or "adjusted efficiency" could be:
    # adjusted_eff[i] = sum_c prizes[i, c] / sum_c (weights[i, c] * v[c])
    
    # Let's try this.
    weighted_weights = weights * scarcity_scores[None, :]  # Shape: (5, 100)
    total_weighted_weight = weighted_weights.sum(axis=0)    # Shape: (100,)
    total_prize = prizes.sum(axis=0)                        # Shape: (100,)
    
    adjusted_eff = total_prize / (total_weighted_weight + eps)
    
    # Normalize all three metrics to mean 1
    sum_ratio_mean = sum_ratio.mean()
    geo_ratio_mean = geo_ratio.mean()
    adjusted_eff_mean = adjusted_eff.mean()
    
    if sum_ratio_mean > 0:
        sum_ratio_norm = sum_ratio / sum_ratio_mean
    else:
        sum_ratio_norm = np.ones_like(sum_ratio)
        
    if geo_ratio_mean > 0:
        geo_ratio_norm = geo_ratio / geo_ratio_mean
    else:
        geo_ratio_norm = np.ones_like(geo_ratio)
        
    if adjusted_eff_mean > 0:
        adjusted_eff_norm = adjusted_eff / adjusted_eff_mean
    else:
        adjusted_eff_norm = np.ones_like(adjusted_eff)
        
    # Blend the three metrics
    # Weights: 0.3 for aggregate, 0.3 for bottleneck, 0.4 for scarcity-adjusted
    # The scarcity-adjusted metric is the most "dynamic" and context-aware.
    scores = 0.3 * sum_ratio_norm + 0.3 * geo_ratio_norm + 0.4 * adjusted_eff_norm
    
    # Ensure all values are positive and finite
    scores = np.maximum(scores, 1e-9)
    scores = np.where(np.isfinite(scores), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
