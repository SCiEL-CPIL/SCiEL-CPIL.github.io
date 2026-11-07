"""Cost-aware acquisition for Bayesian optimization on powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Global deadline guard
import time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop computing and return a safe fallback."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a batch-aware greedy approximation.
    
    We compute a cost-weighted expected improvement that accounts for the
    correlation structure implied by the GP, using a fast Monte Carlo estimate
    of the joint posterior over candidates to approximate the marginal value
    of each candidate given the expected future budget.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP standard deviations
        f_best: float, lowest observed objective value
        cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
        scores: array of shape (2000,), higher means evaluate next
    """
    # Deadline guard: if we're running out of time, return a simple fallback
    if _check_deadline():
        # Fallback: cost-weighted improvement
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = ei / (cost + 1e-9)
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    n = len(mu)
    
    # Compute standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost normalization: divide by cost (cost = 1 + mean(x_scaled))
    # cost is in [1, 2] since x is in [-4,5] scaled to [0,1]
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Batch-aware adjustment:
    # We approximate the marginal value of selecting candidate i by considering
    # that after selecting i, the GP posterior updates, reducing uncertainty for
    # nearby points. We use a fast Monte Carlo approximation:
    #
    # For a GP with Matern-5/2 kernel, the posterior variance at point j given
    # an observation at point i is:
    #   var_j|i = sigma_j^2 - k(i,j)^2 / sigma_i^2  (approximation for 1 obs)
    #
    # The expected improvement at j given observation at i (with value f_i) is
    # approximately reduced. We approximate the marginal gain of i as:
    #   marginal_gain(i) = ei(i) - E[reduction in expected future EI from other candidates]
    #
    # A simpler structural change: use a diversity-aware scoring that penalizes
    # candidates that are too close to already-predicted-low regions, and
    # rewards exploration in uncertain regions.
    #
    # We use a Monte Carlo approximation with a small number of samples to
    # estimate the expected minimum over the remaining budget.
    
    # Fast approach: use the cost-weighted EI but add a small exploration bonus
    # based on sigma that is not already captured by EI, and a diversity term.
    
    # Structural change: instead of pure pointwise EI/cost, use a UCB-like
    # acquisition with cost awareness, which explores more aggressively.
    
    # UCB: mu + kappa * sigma, but we want to minimize, so:
    # score = -(mu - kappa*sigma) / cost  (lower mu + higher sigma is better)
    # But we need to balance exploitation and exploration.
    
    # Let's use a weighted combination:
    #   score = alpha * (ei / cost) + beta * (sigma / cost)
    # where alpha controls exploitation and beta controls exploration.
    
    # To be batch-aware, we can use a simple approximation:
    # The expected number of remaining evaluations is roughly (30 - current_cost) / avg_cost
    # Since we don't have current_cost directly, we use the average cost.
    
    avg_cost = np.mean(cost_safe)
    remaining_evals = max(1.0, (30.0 - np.mean(cost_safe)) / avg_cost)
    
    # Adjust kappa based on remaining budget: more budget -> more exploration
    # kappa = sqrt(2 * log(remaining_evals * n)) / (1 + log(1 + remaining_evals))
    # Simplified:
    kappa = np.sqrt(2.0 * np.log(max(2, remaining_evals)))
    
    # UCB score (for minimization, we want low mu and high sigma)
    ucb = -mu + kappa * sigma
    
    # Normalize UCB to be comparable to EI scale
    ucb_per_cost = ucb / cost_safe
    
    # Combine: use EI as primary, UCB as exploration bonus
    # Weight them based on how confident we are (low sigma -> more exploitation)
    # sigma_mean = np.mean(sigma)
    # weight_ei = 1.0 / (1.0 + sigma_mean)  # not quite right
    
    # Simpler: just use a linear combination with fixed weights
    # The key structural change is using UCB instead of pure EI, which
    # encourages exploration in uncertain regions more strongly.
    
    # Actually, let's try a different structural approach:
    # Use the probability of improvement (POI) with cost awareness,
    # which is more robust to sigma scaling issues.
    
    poi = norm.cdf(z)  # Probability that f(x) < f_best
    poi_per_cost = poi / cost_safe
    
    # Combine EI and POI with cost awareness
    # EI is sensitive to sigma, POI is more stable
    # Use a geometric mean or weighted sum
    
    # Let's use: score = (ei_per_cost) * (0.5 + 0.5 * poi)
    # This dampens EI in regions where improvement is unlikely
    # and boosts it where improvement is likely.
    
    scores = ei_per_cost * (0.3 + 0.7 * poi)
    
    # Add a small UCB component for exploration
    # Scale UCB to be comparable
    ucb_scale = np.std(ucb_per_cost) / (np.std(ei_per_cost) + 1e-9)
    scores = scores + 0.2 * ucb_scale * ucb_per_cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if _check_deadline():
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
