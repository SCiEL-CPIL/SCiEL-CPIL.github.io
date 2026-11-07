"""Cost-aware acquisition using Adaptive Hybrid (EI, PI, LCB) with cost normalization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Module-level persistent state across calls within one evaluation run
_STATE = {
    "start_time": None,
    "f_best_history": [],
    "call_count": 0
}


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Hybrid Cost-Aware Acquisition Function.
    
    Combines Expected Improvement (EI), Probability of Improvement (PI), 
    and Lower Confidence Bound (LCB) strategies, all normalized by cost.
    The LCB exploration parameter (beta) is adapted based on recent improvement history.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _STATE["start_time"] is None:
        _STATE["start_time"] = time.time()
    
    elapsed = time.time() - _STATE["start_time"]
    if elapsed > 290.0:  # Safety margin
        # Fallback to simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / np.maximum(cost, 1e-10)
        return np.where(np.isfinite(scores), scores, 0.0)

    # Update state
    _STATE["call_count"] += 1
    _STATE["f_best_history"].append(f_best)
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate Probability of Improvement (PI)
    # PI = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # 3. Calculate Lower Confidence Bound (LCB) potential
    # We want to maximize (f_best - mu + beta * sigma)
    # Adaptive beta:
    # If we have history, check recent improvement.
    beta = 2.0  # Default exploration
    if len(_STATE["f_best_history"]) >= 3:
        # Check improvement over last 2 steps
        recent_improvement = _STATE["f_best_history"][-3] - _STATE["f_best_history"][-1]
        if recent_improvement > 0.01:
            # Significant improvement, keep exploring
            beta = 2.0
        elif recent_improvement > 0.001:
            # Moderate improvement, balance
            beta = 1.5
        else:
            # Stagnation, exploit more
            beta = 0.8
            
    # LCB score numerator: potential improvement + uncertainty
    lcb_score = f_best - mu + beta * sigma_safe
    
    # 4. Combine strategies normalized by cost
    # We take the maximum of the normalized scores to be robust.
    # This effectively picks the best strategy for each point.
    score_ei = ei / cost_safe
    score_pi = pi / cost_safe
    score_lcb = lcb_score / cost_safe
    
    # Combine: Max of all three
    scores = np.maximum(np.maximum(score_ei, score_pi), score_lcb)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
