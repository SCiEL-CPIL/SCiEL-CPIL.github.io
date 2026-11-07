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
    Cost-aware acquisition function using a Pareto-frontier-inspired approach
    over improvement signal and cost-effective uncertainty.
    
    This approach uses two independent metrics:
    1. Improvement: (f_best - mu) - how much we expect to improve
    2. Cost-effective uncertainty: sigma / cost - exploration per unit cost
    
    We combine these via a rank-based Pareto selection that preserves diversity
    and prevents cost from arbitrarily dominating the improvement signal.
    
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
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Metric 1: Expected improvement (higher is better)
    # For minimization, improvement is f_best - f(x)
    improvement = f_best - mu
    
    # Metric 2: Cost-effective uncertainty (higher is better)
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
    
    imp_norm = robust_normalize(improvement)
    unc_norm = robust_normalize(cost_eff_uncertainty)
    
    # Compute a combined score that balances exploitation and exploration
    # Use a power mean to combine the two normalized metrics
    # This preserves the Pareto front structure while creating a scalar score
    # Higher alpha means more weight on exploitation (improvement)
    # Lower alpha means more weight on exploration (uncertainty)
    
    # Adaptive alpha based on f_best:
    # Early on (high f_best), explore more (lower alpha)
    # Late (low f_best), exploit more (higher alpha)
    f_best_val = float(f_best)
    
    # Rastrigin minimum is 0, typical values up to ~200
    # alpha decays from ~0.7 (exploit) to ~0.3 (explore) as f_best decreases
    alpha = 0.3 + 0.4 * np.tanh(f_best_val / 10.0)
    
    # Power mean combination
    # (imp^alpha + unc^(1-alpha)) / 2
    # This creates a smooth blend that preserves both signals
    combined = 0.5 * (np.power(imp_norm + 1e-12, alpha) + np.power(unc_norm + 1e-12, 1.0 - alpha))
    
    # Add a small probability of improvement term for final ranking
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Final score: combine the Pareto-inspired score with PoI
    # PoI provides a confidence-weighted improvement signal
    scores = combined + 0.3 * poi
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
