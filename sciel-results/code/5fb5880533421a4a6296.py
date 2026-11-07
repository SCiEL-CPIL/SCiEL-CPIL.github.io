"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using a refined Bayesian Regret Minimization (BRM)
    approach with adaptive temperature and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far (lowest, since we minimize)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    t0 = time.time()
    deadline = 295.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute improvement potential: how much better could this point be?
    improvement = f_best - mu  # positive if mu < f_best
    
    # Adaptive temperature: scale by the spread of improvements and uncertainty
    # Use a combination of the standard deviation of improvements and the mean sigma
    imp_std = np.std(improvement) + 1e-12
    mean_sigma = np.mean(sigma_safe) + 1e-12
    
    # Temperature that balances exploitation and exploration
    # Higher temperature -> more uniform distribution -> more exploration
    # Lower temperature -> more concentrated on best candidates -> more exploitation
    temperature = 0.5 * (imp_std + mean_sigma) + 1e-12
    
    # Compute z-scores: improvement weighted by uncertainty
    # This gives a "confidence-weighted" improvement score
    z_improve = improvement / (temperature + sigma_safe * 0.1)
    
    # Softmax over z_improve to get probability weights
    # Subtract max for numerical stability
    z_max = np.max(z_improve)
    exp_z = np.exp(z_improve - z_max)
    prob_min = exp_z / (np.sum(exp_z) + 1e-12)
    
    # Cost-aware BRM: weight by probability of being minimum, divided by cost
    # This gives expected probability of improvement per unit cost
    brm_score = prob_min / cost_safe
    
    # Add exploration component: reward high-uncertainty regions
    # Scale relative to BRM score magnitude
    brm_max = np.max(brm_score) if np.max(brm_score) > 1e-15 else 1.0
    
    # Normalize sigma for exploration
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-12 else 1.0
    sigma_norm = sigma_safe / sigma_max
    
    # Exploration term: encourage exploring uncertain regions
    # Weighted by how far the mean is from f_best (less weight if already good)
    mean_gap = np.maximum(0, f_best - mu)
    mean_gap_max = np.max(mean_gap) if np.max(mean_gap) > 1e-12 else 1.0
    gap_weight = 1.0 - 0.3 * (mean_gap / mean_gap_max)
    
    exploration = 0.3 * sigma_norm * gap_weight * brm_max
    
    # Combine BRM and exploration
    scores = brm_score + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
