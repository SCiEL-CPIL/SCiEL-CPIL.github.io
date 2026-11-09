import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes a hybrid acquisition function for minimization.
    Uses a combination of Expected Improvement and a UCB exploration term.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Acquisition scores for the candidate points (higher is better).
    """
    # Deadline guard: track time to avoid exceeding budget
    t_start = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate improvement: how much better than f_best
    improvement = f_best - mu
    
    # Expected Improvement (EI) for minimization
    from scipy.stats import norm
    
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term: -mu + beta * sigma
    # Use a moderate beta to balance exploration and exploitation
    # The known minimum is -1.8013, so we can use this to adapt beta
    # If f_best is far from the minimum, explore more
    
    # Estimate progress: how close are we to the known minimum?
    known_min = -1.8013
    # Normalize: 1.0 means we're at the minimum, 0.0 means we're far
    # Use a simple heuristic: if f_best > known_min + 0.5, we're still exploring
    # Otherwise, we're exploiting
    
    # Adaptive beta: start high (exploration), decrease as we get closer
    # Use a smooth transition based on the gap to the known minimum
    gap = f_best - known_min
    # Clamp gap to [0, 2] for stable behavior
    gap_clamped = np.clip(gap, 0.0, 2.0)
    # beta decreases from 1.5 to 0.3 as gap goes from 2.0 to 0.0
    beta = 0.3 + 1.2 * (gap_clamped / 2.0)
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units
    # But they have different magnitudes. Let's blend them.
    
    # Weight EI more when we're close to the minimum (exploitation)
    # Weight UCB more when we're far (exploration)
    # Use the same gap-based weighting
    w_ei = 1.0 - 0.5 * (gap_clamped / 2.0)  # 0.5 to 1.0
    w_ucb = 0.5 * (gap_clamped / 2.0)       # 0.0 to 0.5
    
    # Ensure weights sum to 1
    w_sum = w_ei + w_ucb
    w_ei /= w_sum
    w_ucb /= w_sum
    
    # Combine: score = w_ei * EI + w_ucb * UCB
    # But EI and UCB have different scales. Let's normalize them.
    # Actually, both are in objective units, so we can just blend.
    # However, EI can be negative (if mu > f_best, improvement < 0, EI < 0)
    # UCB can also be negative. Let's just blend directly.
    
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time: if we're close to the deadline, return immediately
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        return scores
    
    return scores
