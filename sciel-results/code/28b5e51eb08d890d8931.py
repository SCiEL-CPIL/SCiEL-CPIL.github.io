"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement and
    UCB, normalized by evaluation cost.
    
    This approach shifts from a penalized objective to a rate-of-improvement objective
    (Score = Desirability / Cost), which is structurally different from the parent
    program's multiplicative penalty. It leverages the strong performance of the
    UCB component observed in inspiration 1 while retaining the exploitation
    capability of EI.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin for 330s limit

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative by definition, but numerical errors might cause tiny negatives
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: we want to maximize (-mu + beta * sigma)
    # Beta = 2.0 was effective in Inspiration 1 for exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB.
    # We use a weighted sum. UCB provides exploration, EI provides exploitation.
    # Since UCB can be larger in magnitude, we weight it slightly lower than EI
    # to prevent pure exploration from dominating when a clear improvement is predicted.
    # However, Inspiration 1 (Pure UCB) scored well, so we keep UCB significant.
    w_ei = 1.0
    w_ucb = 0.5
    base_score = w_ei * ei + w_ucb * ucb
    
    # Structural Change: Divide by cost instead of multiplying by a soft penalty.
    # This directly maximizes the expected improvement per unit cost.
    # Add a small epsilon to cost to avoid division by zero (though cost >= 1).
    cost_safe = np.maximum(cost, 1e-9)
    scores = base_score / cost_safe
    
    # Clip negative scores to 0.
    # If base_score is negative, it implies the point is unlikely to improve
    # and has a high predicted value, so it should not be selected.
    scores = np.maximum(scores, 0.0)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
