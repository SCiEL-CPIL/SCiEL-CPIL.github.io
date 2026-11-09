"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _ensure_start()
    return 320.0 - (time.time() - _START_TIME)  # 10s safety margin


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (n_candidates,)
        Mean of the GP posterior at the candidate points.
    sigma : array_like, shape (n_candidates,)
        Standard deviation of the GP posterior at the candidate points.
    f_best : float
        The best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher values are preferred.
    """
    _ensure_start()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Check time budget
    if _time_remaining() < 1.0:
        # Fallback to EI if running out of time
        z = (f_best - mu) / sigma
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge Gradient: E[max(0, f_best - min(f_new))]
    # where f_new ~ N(mu, sigma^2) for each candidate
    # 
    # For a single candidate x_i with posterior N(mu_i, sigma_i^2):
    # KG_i = E[max(0, f_best - f_i)] where f_i ~ N(mu_i, sigma_i^2)
    # This is just EI for a single point.
    #
    # But the full KG considers that after evaluating x_i, the best value
    # becomes min(f_best, f_i). The value of information is:
    # V(f_best) - E[V(min(f_best, f_i))]
    # where V(b) = E[max(0, b - f*)] is the expected regret of the best
    # observed value b.
    #
    # For simplicity and robustness, we use a Monte Carlo estimate of KG:
    # Sample S samples from the joint posterior at all candidates,
    # and for each sample, compute the expected improvement in the global
    # minimum.
    
    # Number of MC samples - balance accuracy vs speed
    # With 2000 candidates, we need to be careful about memory
    n_mc = 50
    
    # Pre-compute some quantities
    # For each candidate i, the posterior predictive is N(mu_i, sigma_i^2)
    
    # Monte Carlo KG estimate:
    # For each MC sample s:
    #   Draw f_i^s ~ N(mu_i, sigma_i^2) for all i
    #   The new best after evaluating i would be min(f_best, f_i^s)
    #   But we also need to account for the fact that the GP is updated.
    #   For simplicity, we use the approximation that the value of
    #   evaluating i is E[max(0, f_best - f_i)] which is just EI.
    #
    # A better approximation: use the fact that the KG for candidate i is
    # approximately the expected reduction in the lower confidence bound
    # of the global minimum.
    
    # Let's use a hybrid approach:
    # 1. Compute EI for each candidate
    # 2. Compute a "coverage" bonus based on how unique the candidate is
    #    in terms of its posterior distribution
    # 3. Combine them
    
    # EI component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # For KG, we want to favor candidates that are likely to be the global
    # minimum. A good proxy is the probability that f_i < f_best, weighted
    # by the magnitude of improvement.
    
    # Compute the probability that each candidate beats f_best
    p_beat = norm.cdf((f_best - mu) / sigma)
    
    # Expected minimum of f_i and f_best (approximation)
    # E[min(f_i, f_best)] = f_best - EI_i (from the definition of EI)
    
    # KG approximation: we want to maximize the expected improvement in
    # the global best. This is closely related to EI but with a different
    # weighting.
    
    # Let's try: KG_i ≈ EI_i * (1 + p_beat_i)
    # This gives extra weight to candidates that are very likely to beat f_best
    
    kg_score = ei * (1.0 + p_beat)
    
    # Add a small exploration bonus based on uncertainty
    # Candidates with high uncertainty that are near the current best
    # are valuable for exploration
    exploration_bonus = sigma * norm.pdf((f_best - mu) / sigma)
    
    # Combine
    scores = kg_score + 0.1 * exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
