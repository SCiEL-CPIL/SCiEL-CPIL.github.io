"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) normalized by evaluation cost.
    
    The blend between exploitation (EI) and exploration (UCB) is adapted
    dynamically based on the current uncertainty (sigma) of the Gaussian Process.
    High uncertainty (early stage) favors exploration; low uncertainty (late stage)
    favors exploitation.
    
    mu, sigma, cost: shape (2000,)
    f_best: scalar, the best objective value found so far (minimization).
    Returns: 2000 finite scores (higher = evaluate next)
    """
    t_start = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite inputs
    if not np.all(np.isfinite(mu)):
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.all(np.isfinite(sigma)):
        sigma = np.nan_to_num(sigma, nan=1e-10, posinf=1e6, neginf=1e-10)
    if not np.all(np.isfinite(cost)):
        cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10) # Numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    score_ei = ei / cost_safe
    
    # 2. Compute Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look for low mu.
    # UCB_min = -mu + beta * sigma
    # The beta parameter controls the exploration-exploitation tradeoff.
    # We adapt beta based on the mean sigma to shift from exploration to exploitation.
    
    # Estimate progress: 
    # Early: sigma is high. Late: sigma is low.
    mean_sigma = np.mean(sigma)
    
    # Normalize mean_sigma to a 0-1 range roughly.
    # Typical sigma for standardized GP is 1.0. It decays as points are added.
    # If mean_sigma > 0.5, we are likely early/mid. If < 0.2, late.
    # Let's map mean_sigma to an exploration weight.
    # High sigma -> High exploration weight.
    # Low sigma -> Low exploration weight (more exploitation).
    
    # Heuristic: exploration_weight = exp(-mean_sigma) or similar?
    # Let's use a linear mapping clipped.
    # If mean_sigma is 1.0, weight = 1.0 (max explore)
    # If mean_sigma is 0.1, weight = 0.1 (min explore)
    # This might be too aggressive. Let's use a softer curve.
    
    # Alternative: Use f_best. If f_best is close to 0 (min), exploit.
    # But we don't know the scale of f_best initially.
    # The sigma-based approach is more robust to the function scale.
    
    # Let's define alpha as the weight for Exploitation (EI).
    # alpha = 1 - exploration_weight
    # exploration_weight = clip(mean_sigma, 0.2, 0.8)
    # So alpha is between 0.2 and 0.8.
    
    # However, in the very late stages, we want alpha close to 1.0.
    # In the very early stages, we might want alpha around 0.5 or lower.
    
    # Let's try:
    # alpha = 0.5 + 0.5 * (1 - mean_sigma)
    # If mean_sigma = 1.0 -> alpha = 0.5
    # If mean_sigma = 0.0 -> alpha = 1.0
    # This seems reasonable.
    
    alpha = 0.5 + 0.5 * (1.0 - np.clip(mean_sigma, 0.0, 1.0))
    
    # For UCB, we also need a beta.
    # Standard beta is often 1.0 or 2.0.
    # Let's keep beta fixed at 1.0 for simplicity, as the alpha blending handles the shift.
    beta = 1.0
    
    ucb = -mu + beta * sigma
    score_ucb = ucb / cost_safe
    
    # 3. Combine Scores
    # The scales of score_ei and score_ucb might differ.
    # EI is always >= 0. UCB can be negative (if mu > f_best + beta*sigma? No, mu is likely < f_best for good points).
    # Actually, for minimization, if mu < f_best, EI is positive.
    # UCB = -mu + beta*sigma. If mu is small (good), -mu is large positive.
    # So both scores are generally positive for promising points.
    
    # We should ensure they are on comparable scales.
    # Normalizing by their respective means/stds might be dangerous if variance is low.
    # Instead, we can just blend them directly.
    
    scores = alpha * score_ei + (1.0 - alpha) * score_ucb
    
    # 4. Safety Checks
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        pass
        
    return scores
