"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Batch-aware Look-Ahead acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Batch-aware Look-Ahead: simulate the next B evaluations
    # We greedily select top candidates and update the posterior variance
    # locally to approximate information gain
    
    B = 5  # Number of look-ahead steps
    n = len(mu)
    
    # Track which candidates have been "selected" in the simulation
    selected = np.zeros(n, dtype=bool)
    
    # Simulate B steps of greedy selection
    simulated_f_best = f_best
    simulated_sigma = sigma_safe.copy()
    simulated_mu = mu.copy()
    
    for step in range(B):
        # Check time budget
        if time.time() - start_time > 280:
            break
        
        # Compute EI with current simulated state
        z_sim = (simulated_f_best - simulated_mu) / simulated_sigma
        z_sim = np.clip(z_sim, -10, 10)
        ei_sim = (simulated_f_best - simulated_mu) * norm.cdf(z_sim) + simulated_sigma * norm.pdf(z_sim)
        ei_sim = np.where(np.isfinite(ei_sim), ei_sim, 0.0)
        
        # Mask already selected points
        ei_sim[selected] = -np.inf
        
        # Select the best candidate
        best_idx = np.argmax(ei_sim)
        
        if not np.isfinite(ei_sim[best_idx]):
            break
        
        selected[best_idx] = True
        
        # Update simulated_f_best with the posterior mean (conservative estimate)
        simulated_f_best = min(simulated_f_best, simulated_mu[best_idx])
        
        # Locally reduce uncertainty for candidates near the selected point
        # This approximates the information gain from observing the selected point
        # We use a simple heuristic: reduce sigma for points within a certain distance
        # Since we don't have coordinates, we use index-based proximity as a proxy
        
        # Reduce uncertainty for the selected point itself
        simulated_sigma[best_idx] *= 0.5
        
        # Reduce uncertainty for nearby indices (approximate local effect)
        window = 50
        left = max(0, best_idx - window)
        right = min(n, best_idx + window + 1)
        
        # Linear decay of uncertainty reduction with distance from selected point
        for i in range(left, right):
            if i == best_idx:
                continue
            dist = abs(i - best_idx)
            decay = 1.0 - dist / (window + 1)
            simulated_sigma[i] *= (1.0 - 0.3 * decay)
    
    # Now compute the final scores based on the simulated future state
    # The idea is that candidates which remain valuable after the simulated
    # B evaluations are those that provide additional information or improvement
    
    z_future = (simulated_f_best - simulated_mu) / simulated_sigma
    z_future = np.clip(z_future, -10, 10)
    ei_future = (simulated_f_best - simulated_mu) * norm.cdf(z_future) + simulated_sigma * norm.pdf(z_future)
    ei_future = np.where(np.isfinite(ei_future), ei_future, 0.0)
    
    # Also compute the original EI for the immediate step
    ei_immediate = ei.copy()
    
    # Combine immediate and future EI
    # Points that have high immediate EI AND high future EI are most valuable
    # Points with high future EI but low immediate EI might be good for exploration
    
    # Normalize both for combination
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    norm_ei_imm = normalize_scores(ei_immediate)
    norm_ei_fut = normalize_scores(ei_future)
    
    # Adaptive weighting: early on, prioritize immediate improvement
    # Later on, prioritize future potential (exploration)
    distance_to_min = max(f_best - f_min_known, 0.0)
    
    if distance_to_min > 15.0:
        w_imm = 0.7
        w_fut = 0.3
    elif distance_to_min > 5.0:
        w_imm = 0.6
        w_fut = 0.4
    else:
        w_imm = 0.5
        w_fut = 0.5
    
    final_scores = w_imm * norm_ei_imm + w_fut * norm_ei_fut
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
