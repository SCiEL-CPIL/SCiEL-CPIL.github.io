import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) with a
    quantile-based exploration bonus.
    
    The score is primarily driven by EI (exploitation of known improvements)
    but includes a structural switch: if a candidate is in the top 5% of
    uncertainty (sigma) among the batch AND its predicted mean is better than
    the current best, it receives a bonus. This encourages exploring high-uncertainty
    regions that are still plausible to be optimal, rather than just the
    highest variance regions (which might be far from the optimum).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and non-zero
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is finite
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # 2. Compute Quantile-based Exploration Bonus
    # We want to identify candidates that are:
    # a) In the top 5% of uncertainty (high sigma)
    # b) Have a predicted mean that is better (lower) than f_best
    
    # Calculate the 95th percentile of sigma
    sigma_threshold = np.percentile(sigma, 95)
    
    # Mask for high uncertainty
    high_uncertainty_mask = sigma >= sigma_threshold
    
    # Mask for promising mean (mu < f_best)
    promising_mean_mask = mu < f_best
    
    # Combine masks: only consider points that are both high uncertainty AND promising
    exploration_mask = high_uncertainty_mask & promising_mean_mask
    
    # Define the bonus value
    # The bonus should be significant enough to compete with EI but not dominate it entirely.
    # A typical EI value near the optimum is small. A bonus of ~0.1 * (f_best - min(mu)) scale?
    # Let's use a fixed scale relative to the problem range or f_best.
    # Since we don't know the scale of f_best exactly (it's standardized in GP but mu/sigma are in objective units),
    # we can use a heuristic. 
    # However, the task says "mu and sigma ... are in objective units".
    # Hartmann 3D min is -3.86. f_best starts around -2 or so.
    # Let's use a bonus that is a fraction of the current best improvement potential.
    # If f_best is -3.8, a bonus of 0.1 is small. If f_best is 0, a bonus of 0.1 is larger.
    
    # A simpler, more robust structural change:
    # If the condition is met, add a bonus proportional to the sigma itself? No, that's UCB.
    # The directive says: "penalty derived from the empirical rank of the uncertainty".
    # Let's interpret "penalty" as a bonus in the score (since higher score is better).
    # Let's add a bonus of 0.5 * (sigma - sigma_threshold) for those in the top 5%?
    # Or just a fixed boost?
    
    # Let's try a boost that is 10% of the typical scale of the objective.
    # Since we don't know the scale, let's look at the spread of mu.
    # Actually, the simplest structural change is to just add a constant bonus to the EI
    # for points that satisfy the condition.
    
    # Let's use a bonus of 0.1. This is a guess. 
    # A better approach: The bonus should be comparable to the EI values.
    # Let's compute the median EI to scale the bonus? No, that's expensive and might be unstable.
    
    # Let's stick to the directive: "selecting points that are in the top 5% of uncertainty 
    # but only if their mean is below the current best".
    # We will add a bonus to the EI score for these points.
    # Let's set the bonus to be 0.5 * np.median(sigma) * 0.1? 
    # Let's just use a fixed small bonus for now, e.g., 0.05.
    # Wait, if f_best is -3.86, EI is 0. If f_best is 0, EI can be large.
    # A fixed bonus might be too small or too large.
    
    # Let's scale the bonus by the standard deviation of the sigma values?
    # Or simply, let's use a bonus of 0.1 * (f_best - np.min(mu))?
    # This makes the bonus relative to the potential improvement.
    
    range_improvement = f_best - np.min(mu)
    if range_improvement < 1e-6:
        range_improvement = 1.0 # Avoid division by zero or tiny values
        
    bonus_value = 0.1 * range_improvement
    
    scores = ei.copy()
    scores[exploration_mask] += bonus_value
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
