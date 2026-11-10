"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def _ei(mu, sigma, f_best):
    """Compute Expected Improvement for minimization."""
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _ucb(mu, sigma, beta):
    """Compute UCB for minimization: -mu + beta * sigma."""
    return -mu + beta * sigma

def _cost_aware_ucb(mu, sigma, f_best, cost):
    """
    Cost-aware UCB variant.
    Score = (f_best - mu) / cost + sigma / cost^2
    """
    term_exploit = (f_best - mu) / cost
    term_explore = sigma / (cost * cost)
    return term_exploit + term_explore

def _cost_aware_ei(mu, sigma, f_best, cost):
    """
    Standard EI divided by cost.
    """
    ei = _ei(mu, sigma, f_best)
    return ei / cost

def _cost_aware_poi(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement.
    POI = Phi((f_best - mu) / sigma)
    Score = POI / cost
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    return poi / cost

def _hybrid_score(mu, sigma, f_best, cost, w_ucb, w_poi):
    """
    Blend of cost-aware EI, UCB, and POI.
    w_ucb controls the mix of UCB vs (EI + POI)
    w_poi controls the mix of POI vs EI within the non-UCB part
    """
    score_ei = _cost_aware_ei(mu, sigma, f_best, cost)
    score_ucb = _cost_aware_ucb(mu, sigma, f_best, cost)
    score_poi = _cost_aware_poi(mu, sigma, f_best, cost)
    
    # Combine EI and POI first
    score_base = (1.0 - w_poi) * score_ei + w_poi * score_poi
    
    # Then blend with UCB
    return (1.0 - w_ucb) * score_base + w_ucb * score_ucb

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a structural blend of
    Cost-Aware EI, Cost-Aware UCB, and Cost-Aware POI.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Estimate progress from f_best
    # Use a sigmoid-like mapping to determine exploration weight
    # When f_best is large, we want more exploration (higher w_ucb, higher w_poi)
    # When f_best is small, we want more exploitation (lower w_ucb, lower w_poi)
    
    # Clip f_best to a reasonable range for the logistic
    f_best_clipped = np.clip(f_best, -5.0, 20.0)
    
    # Logistic for w_ucb: weight for the UCB component
    # w_ucb = 1.0 / (1.0 + exp(k * (f_best - c)))
    k_ucb = 2.0
    c_ucb = 1.0
    w_ucb = 1.0 / (1.0 + np.exp(k_ucb * (f_best_clipped - c_ucb)))
    
    # Clamp to [0.1, 0.7] to avoid extremes
    w_ucb = np.clip(w_ucb, 0.1, 0.7)
    
    # Logistic for w_poi: weight for POI within the base score
    # POI is more robust to noise and can help when EI is zero
    k_poi = 1.5
    c_poi = 0.5
    w_poi = 1.0 / (1.0 + np.exp(k_poi * (f_best_clipped - c_poi)))
    
    # Clamp to [0.1, 0.6]
    w_poi = np.clip(w_poi, 0.1, 0.6)
    
    # Compute the hybrid score
    score = _hybrid_score(mu, sigma, f_best, cost, w_ucb, w_poi)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
