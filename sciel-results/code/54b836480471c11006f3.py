"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    with an adaptive exploration bonus, divided by cost.
    
    This approach uses a structural blend:
    1. PoI as the primary exploitation signal (robust for multimodal Rastrigin)
    2. A cost-effective uncertainty term as the exploration signal
    3. A power-mean combination that adapts the balance based on f_best
    4. Final division by cost for cost-efficiency
    
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
        # Fallback: Prioritize cheap points with high uncertainty
        sigma_safe = np.maximum(sigma, 1e-8)
        return sigma_safe / (cost + 1e-9)
    
    # Ensure inputs are arrays and handle types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure f_best is a scalar float
    if hasattr(f_best, 'min'):
        f_best_val = float(np.min(f_best))
    else:
        f_best_val = float(f_best)
        
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best_val - mu) / sigma_safe
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-effective uncertainty (higher is better)
    cost_eff_uncertainty = sigma_safe / (cost + 1e-9)
    
    # Normalize both metrics to [0, 1] range for fair comparison
    # Use robust normalization based on percentiles to handle outliers
    def robust_normalize(x):
        """Normalize array to [0,1] using 5th and 95th percentiles."""
        lo = np.percentile(x, 5)
        hi = np.percentile(x, 95)
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        normalized = (x - lo) / (hi - lo)
        return np.clip(normalized, 0.0, 1.0)
    
    poi_norm = robust_normalize(poi)
    unc_norm = robust_normalize(cost_eff_uncertainty)
    
    # Adaptive alpha based on f_best:
    # Early on (high f_best), explore more (lower alpha)
    # Late (low f_best), exploit more (higher alpha)
    # Rastrigin minimum is 0, typical values up to ~200
    # alpha decays from ~0.7 (exploit) to ~0.3 (explore) as f_best decreases
    alpha = 0.3 + 0.4 * np.tanh(f_best_val / 10.0)
    
    # Power mean combination
    # (poi^alpha + unc^(1-alpha)) / 2
    # This creates a smooth blend that preserves both signals
    combined = 0.5 * (np.power(poi_norm + 1e-12, alpha) + np.power(unc_norm + 1e-12, 1.0 - alpha))
    
    # Final score: divide by cost for cost-efficiency
    # This ensures we prefer points that give good acquisition per unit cost
    scores = combined / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
