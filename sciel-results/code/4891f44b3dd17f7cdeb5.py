"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition score for 2000 candidates.
    Uses a hybrid of Expected Improvement and Lower Confidence Bound,
    normalized by evaluation cost.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # Compute Lower Confidence Bound (LCB) for minimization
    # LCB(x) = mu - beta * sigma
    # We want to maximize the "confidence" of finding a low value.
    # A common heuristic is to use a beta that adapts or is fixed.
    # Using a fixed beta of 2.0 (roughly 95% confidence) is standard.
    beta = 2.0
    lcb = mu - beta * sigma_safe

    # Hybrid score: Combine EI and LCB.
    # EI is scale-dependent and handles the "improvement" aspect well.
    # LCB is a direct proxy for the lower bound of the function value.
    # We normalize LCB to be comparable to EI.
    # Since we are minimizing, a lower LCB is better.
    # We want a score that is high for good candidates.
    # So for LCB, we can use -LCB or (f_best - LCB) to make it an "improvement-like" score.
    # Let's use (f_best - LCB) as the LCB-based improvement score.
    lcb_improvement = f_best - lcb
    lcb_improvement = np.maximum(lcb_improvement, 0.0)

    # Blend the two scores.
    # EI tends to be more exploitative when f_best is good.
    # LCB provides a steady exploration pressure.
    # We can weight them. Let's try a 50/50 blend or a dynamic blend.
    # A simple average is a good starting point for a structural change.
    # score = 0.5 * ei + 0.5 * lcb_improvement
    
    # However, EI and LCB_improvement might have different scales.
    # EI is bounded by f_best - min(mu). LCB_improvement is similar.
    # Let's just use the average.
    
    hybrid_score = 0.5 * ei + 0.5 * lcb_improvement

    # Cost-aware: divide by cost
    scores = hybrid_score / cost_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
