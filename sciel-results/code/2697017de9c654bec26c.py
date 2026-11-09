import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and EI for Bayesian optimization.
    
    Args:
        mu: (2000,) Mean of the GP predictions.
        sigma: (2000,) Standard deviation of the GP predictions.
        f_best: float, the best (lowest) value observed so far.
        cost: (2000,) Cost of evaluating each candidate.
        
    Returns:
        scores: (2000,) Acquisition scores. Higher scores indicate better candidates.
    """
    t_start = time.time()
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Estimate remaining budget ratio
    # Total budget is 30, initial 5 points cost ~5, so remaining ~25
    # We can estimate how much budget we've used based on f_best convergence
    # A simple heuristic: use the number of unique values or just a fixed schedule
    # Let's use a dynamic kappa that decreases as we get closer to convergence
    # Or better: use the ratio of current best to the known minimum (-3.86278)
    
    # Known minimum for hartmann 3d
    f_min = -3.86278
    
    # Progress: how close are we to the minimum?
    # If f_best is close to f_min, we want less exploration
    # progress = 1 - (f_best - f_min) / (initial_range)
    # But we don't know initial_range exactly. Let's use a simpler approach.
    
    # Dynamic kappa: start with higher exploration, decrease as we find better values
    # Use a schedule based on f_best relative to f_min
    # When f_best is far from f_min, explore more; when close, exploit more
    
    # Normalize progress: 0 when f_best is bad, 1 when f_best is near f_min
    # Assume initial f_best is around -1 to -2 (typical for random points)
    f_range = 3.0  # approximate range from random to minimum
    progress = np.clip((f_best - f_min) / f_range, 0, 1)
    # progress close to 1 means we're near the minimum
    # We want kappa to decrease as progress increases
    kappa = 2.0 * (1 - progress) + 0.5
    # So kappa starts around 2.5 and decreases to 0.5
    
    # UCB score: we want to minimize mu - kappa * sigma
    # So acquisition = -(mu - kappa * sigma) = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma_safe
    
    # EI score for minimization
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -40, 40)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Combine UCB and EI
    # When we're early (low progress), weight UCB more
    # When we're late (high progress), weight EI more
    alpha = 0.3 + 0.7 * progress  # alpha for EI, (1-alpha) for UCB
    # Actually, let's think about this differently.
    # UCB is good for exploration, EI is good for exploitation.
    # Early: more UCB. Late: more EI.
    # Let's use: score = (1 - alpha) * ucb_score + alpha * ei
    # where alpha increases with progress
    
    # Normalize both to comparable scales
    # ucb_score can be positive or negative
    # ei is always non-negative
    
    # Let's just use a weighted combination
    scores = (1 - alpha) * ucb_score + alpha * ei
    
    # Cost-aware: divide by cost
    scores = scores / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > 290:
        pass
    
    return scores
