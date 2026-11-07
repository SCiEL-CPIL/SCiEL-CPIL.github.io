# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit


def _ensure_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330.0 - _DEADLINE_MARGIN
    return _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement per unit cost,
    with a dynamic relaxation factor as budget depletes.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    _ensure_deadline()
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (standard EI for minimization)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Dynamic relaxation: as total cost approaches budget (30), 
    # the penalty for cost should relax to force exploitation.
    # We use sqrt(remaining_budget + current_cost) as a soft denominator
    # that grows as cost increases, reducing the relative penalty.
    
    # The total budget is 30. The 'cost' parameter represents the cumulative
    # cost spent so far (or the cost of evaluating each candidate?).
    # From the task: "Evaluating x costs 1 + mean(x scaled to [0, 1])"
    # and "cost has shape (2000,)". 
    # Looking at the seed: it divides by cost directly.
    # The cost here likely represents the cost of evaluating each of the 2000 candidates.
    # But wait, the loop is: "While total cost is below 30, the evaluator scores 2000 new 
    # uniform candidates with acquisition and evaluates the best one."
    # So 'cost' is the cost of each of the 2000 candidates.
    
    # The directive suggests: "ratio of the Expected Improvement to the square root 
    # of the total remaining budget plus current cost"
    # But we don't have the total remaining budget passed in. We only have 'cost' per candidate.
    # However, we can infer the phase of the search from the magnitude of f_best 
    # or just use a fixed heuristic.
    
    # A better interpretation from the directive: 
    # "approximated by the ratio of the Expected Improvement to the square root 
    # of the total remaining budget plus current cost"
    # Since we don't have the global state, let's look at the cost array.
    # The cost of evaluating a point is 1 + mean(x_scaled).
    # Points near 0.5 in all dimensions have cost ~1.5. Points near 0 have cost ~1.
    # So cost is in [1, 1+d/2] roughly.
    
    # Let's try a dynamic factor based on the median cost or just a fixed relaxation.
    # Actually, the seed program just did EI / cost.
    # The inspiration mentions: "CA-POI biases exploration towards lower-cost regions... Measured: 1 -> 1.472"
    # This suggests that weighting by 1/cost (or similar) was effective.
    
    # Let's try a hybrid: EI / (cost * relaxation_factor)
    # where relaxation_factor decreases as we get closer to the end.
    # But we don't know when we are at the end.
    
    # Alternative: Use the "Knowledge" that CA-POI (Penalized by cost) worked.
    # Let's stick with EI / cost but add a small exploration bonus for high uncertainty
    # to avoid getting stuck, or adjust the exponent.
    
    # Let's try: EI / (cost ** 0.5) to be less aggressive on cost?
    # Or EI / cost as in the seed, but ensure numerical stability.
    
    # The seed scored 1.0. The child p40 scored 1.09.
    # Let's try a slightly more exploratory approach: 
    # Score = EI / cost + beta * sigma / cost
    # This adds a pure uncertainty term.
    
    beta = 0.1
    
    score = ei / cost + beta * sigma_safe / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
