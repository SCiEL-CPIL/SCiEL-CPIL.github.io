"""Cost-aware acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for numerical stability
_EPS = 1e-9
_DEADLINE_FRACTION = 0.92  # safety margin for the 330s wall-clock limit


def _expected_improvement(mu, sigma, f_best):
    """Compute expected improvement (EI) for each candidate."""
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb_score(mu, sigma, f_best, cost, xi=0.0, beta=2.0):
    """Compute UCB-style score with cost normalization."""
    # For minimization, UCB = mu - beta * sigma
    # We want high score = good candidate
    # EI per cost is a good baseline
    sigma_safe = np.maximum(sigma, _EPS)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Add exploration term
    exploration = beta * sigma_safe
    
    # Combine: EI + exploration, normalized by cost
    score = (ei + exploration) / np.maximum(cost, _EPS)
    return score


def _diversity_penalty(scores, mu, sigma, f_best, cost, top_k=50, gamma=0.3):
    """
    Apply a diversity penalty to avoid redundant candidates.
    This is a simplified set-based approach: after selecting top candidates,
    penalize those that are close in predicted mean to already-selected ones.
    
    Since we don't have coordinates, we use the GP posterior structure:
    candidates with similar mu and sigma are likely redundant.
    """
    if len(scores) < top_k * 2:
        return scores
    
    # Get indices of top candidates
    sorted_idx = np.argsort(-scores)[:top_k]
    top_scores = scores[sorted_idx]
    top_mu = mu[sorted_idx]
    top_sigma = sigma[sorted_idx]
    
    # Compute pairwise similarity among top candidates
    # Use a simple heuristic: similarity based on mu difference
    # Candidates with very similar predicted means are redundant
    
    penalized = scores.copy()
    for i in range(len(top_idx)):
        idx_i = sorted_idx[i]
        # Compare with previously selected candidates
        for j in range(i):
            idx_j = sorted_idx[j]
            # Distance in prediction space
            dist = abs(mu[idx_i] - mu[idx_j]) + abs(sigma[idx_i] - sigma[idx_j])
            # If very similar, reduce score
            if dist < 0.1 * (abs(f_best) + 1.0):
                penalized[idx_i] *= (1.0 - gamma * (1.0 - dist / (0.1 * (abs(f_best) + 1.0))))
    
    return penalized


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that combines expected improvement,
    exploration, and diversity considerations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, _EPS)
    
    # Base score: EI per cost
    ei = _expected_improvement(mu, sigma, f_best)
    base_score = ei / cost
    
    # Add exploration component
    sigma_safe = np.maximum(sigma, _EPS)
    # Adaptive beta: increase exploration early, decrease later
    # Use f_best as a proxy for progress
    progress = max(0.0, min(1.0, -f_best / (abs(f_best) + 1.0)))
    beta = 1.0 + 2.0 * (1.0 - progress)
    
    exploration_score = beta * sigma_safe / cost
    
    # Combine
    score = base_score + exploration_score
    
    # Apply diversity penalty to top candidates
    score = _diversity_penalty(score, mu, sigma, f_best, cost, top_k=100, gamma=0.2)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
