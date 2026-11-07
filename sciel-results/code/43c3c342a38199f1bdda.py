"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement (EI)
    and Probability of Improvement (PoI) with adaptive weighting and cost normalization.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        return 1.0 / (np.maximum(cost, 1e-9))
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Z score
    Z = (f_best - mu) / sigma_safe
    Z_clipped = np.clip(Z, -10, 10)
    
    # Calculate Expected Improvement (EI)
    EI = (f_best - mu) * norm.cdf(Z_clipped) + sigma_safe * norm.pdf(Z_clipped)
    EI = np.maximum(EI, 0.0)
    
    # Calculate Probability of Improvement (PoI)
    PoI = norm.cdf(Z_clipped)
    
    # Adaptive weighting: 
    # Early on (high uncertainty), we might want more exploration (PoI/LCB-like).
    # Later, we want exploitation (EI).
    # However, the finding suggests a hybrid improves balance.
    # Let's blend EI and PoI. 
    # Note: EI scales with the magnitude of improvement, PoI is bounded [0,1].
    # To make them comparable, we can scale PoI by the typical improvement scale.
    # Or simply blend them directly if we normalize.
    # Let's try a simple weighted sum: score = w * EI + (1-w) * PoI * scale
    
    # A more robust approach for cost-awareness:
    # Use EI as the primary signal but boost it with PoI where uncertainty is high.
    # Or, use the hybrid from the finding: "Hybrid EI/PoI blend".
    
    # Let's try: score = (EI + PoI * sigma_safe) / cost
    # Why add PoI * sigma? 
    # EI already contains sigma. 
    # Perhaps the finding implies that PoI helps escape local minima by keeping a baseline probability of improvement.
    
    # Alternative structural change from Inspiration 1: Pure PoI / cost.
    # But the finding says Hybrid improved from 1.075 to 1.458.
    # Let's construct a hybrid that balances the magnitude of EI with the likelihood of PoI.
    
    # Normalize EI and PoI to [0,1] range roughly? No, EI can be large.
    # Let's use a multiplicative boost:
    # score = EI * (1 + PoI) / cost
    # This rewards points that have both high expected gain and high probability of gain.
    
    # Or, as in Inspiration 1, maybe PoI/cost is better for exploration?
    # The finding said "Hybrid EI/PoI blend ... improves exploration-exploitation balance".
    
    # Let's try a linear blend with a dynamic weight.
    # If f_best is still far from 0 (min), we are in exploration phase.
    # If f_best is close to 0, we are in exploitation.
    # Rastrigin min is 0.
    
    # Let's use a simple blend:
    # score = (EI + 0.5 * PoI * sigma_safe) / cost_safe
    # The term PoI * sigma_safe represents the expected "optimistic" improvement potential.
    
    # Actually, let's look at Inspiration 1 again. It used PoI/cost.
    # The current program uses EI/cost.
    # The finding says Hybrid is better.
    
    # Let's try: score = (EI + PoI) / cost_safe
    # But EI and PoI have different units. EI is in objective units, PoI is probability.
    # We should scale PoI by the characteristic scale of the objective.
    # A good scale is the current standard deviation of the observed values or just a fixed scale.
    # Since we don't have observed values here, let's use sigma_safe as a local scale.
    
    # Revised Hybrid:
    # score = (EI + PoI * sigma_safe) / cost_safe
    
    scores = (EI + PoI * sigma_safe) / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
