"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def _fast_norm_cdf(x):
    """Fast approximation of the standard normal CDF."""
    # Use the error function approximation for speed
    return 0.5 * (1.0 + np.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x**2)))

def _fast_norm_pdf(x):
    """Fast approximation of the standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed function value so far
    cost : array of shape (2000,) - cost of evaluating each point
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Time guard for safety
    start_time = time.time()
    deadline = start_time + 2.0  # 2 second soft limit for this function call
    
    # Handle edge cases where sigma might be very small or zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    
    # Use fast approximations for speed
    cdf_val = _fast_norm_cdf(z)
    pdf_val = _fast_norm_pdf(z)
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * cdf_val + sigma_safe * pdf_val
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Check time
    if time.time() > deadline:
         # Fallback: return EI only (ignoring cost) if we are running out of time
         scores = ei
         scores = np.where(np.isfinite(scores), scores, 0.0)
         scores = np.maximum(scores, 0.0)
         return scores
    
    # Structural change: Use a probabilistic utility framework that models
    # the trade-off between improvement and cost more explicitly.
    # Instead of simple EI/cost, we model the probability of achieving a
    # "good" improvement per unit cost.
    
    # The cost is C(x) = 1 + mean(x_scaled). Range [1, 2].
    # We want to maximize the expected improvement while minimizing cost.
    
    # Define a target improvement threshold relative to current best
    # This creates a probabilistic "success" criterion
    # The idea: what is the probability that f(x) < f_best - delta, weighted by cost?
    
    # We use a sigmoid-like transformation to create a smooth probability of success
    # where success means improving by more than a small amount
    
    # Scale factor for the sigmoid - controls how sharp the probability transition is
    # This effectively creates a "soft" threshold for what counts as a meaningful improvement
    scale = 0.5  # Tuning parameter for the sensitivity of the probability model
    
    # Compute the "improvement margin" - how much better than f_best we expect to be
    # For points where mu < f_best, this is positive; otherwise negative
    margin = f_best - mu
    
    # Probability of achieving at least a small improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # But we want to weight this by how much improvement we get
    
    # Use a weighted combination:
    # 1. Probability of improvement (from GP)
    # 2. Expected magnitude of improvement
    # 3. Cost penalty
    
    # The key insight: in a tight budget, we should prioritize points that are
    # likely to provide substantial improvement, not just any improvement.
    
    # Compute a "quality" score that combines probability and magnitude
    # This is similar to EI but with a different weighting
    
    # Alternative: Use the concept of "probability of being in the top-k"
    # or "probability of significant improvement"
    
    # Let's use a hybrid approach:
    # Score = P(improvement) * E[improvement | improvement] / cost^alpha
    
    # P(improvement) = Phi(z) where z = (f_best - mu) / sigma
    # E[improvement | improvement] = EI / P(improvement) when P(improvement) > 0
    
    # So Score = EI / cost^alpha (which is what we had before)
    
    # But we can add a bonus for high-confidence improvements
    # High confidence = low sigma relative to the improvement margin
    
    # Define a confidence factor:
    # confidence = exp(-sigma / (f_best - mu + epsilon)) when mu < f_best
    # This gives higher weight to points where we're confident about the improvement
    
    epsilon = 1e-8
    improvement_margin = np.maximum(f_best - mu, epsilon)
    
    # Confidence: how certain are we that this point will improve?
    # Low sigma relative to margin = high confidence
    confidence = np.exp(-sigma_safe / improvement_margin)
    
    # Combine EI with confidence
    # This creates a preference for "safe" improvements over "risky" ones
    # In a tight budget, we can't afford to waste evaluations on uncertain points
    
    # Score = EI * confidence / cost^alpha
    alpha = 1.0
    cost_factor = np.power(cost, alpha)
    
    scores = ei * confidence / cost_factor
    
    # Add a small exploration bonus for very uncertain regions
    # But only if they're not too expensive
    # This prevents the algorithm from getting stuck in local optima
    
    # Exploration bonus: high when sigma is large relative to the range of f_best
    # But penalized by cost
    f_range = max(abs(f_best), 1.0)  # Scale for exploration
    exploration = sigma_safe / f_range
    exploration_bonus = 0.1 * exploration / cost_factor
    
    scores = scores + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative scores
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
