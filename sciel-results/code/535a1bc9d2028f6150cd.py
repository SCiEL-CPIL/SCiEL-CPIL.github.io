"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _norm_cdf(x):
    """Numerically stable standard normal CDF using scipy."""
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Bayesian Regret Minimization (BRM) with Thompson sampling flavor:
    - Sample from the GP posterior (Thompson sampling) to get a stochastic surrogate
    - Weight by probability of being the global minimum within remaining budget
    - Account for cost by discounting high-cost evaluations
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard: if we're running low on time, return a simple score
    if (time.time() - _START_TIME) > _DEADLINE:
        return -mu / (np.maximum(cost, 1e-9) + 1.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 1e-10), 1e-10)
    cost = np.where(np.isfinite(cost), np.maximum(cost, 1e-10), 1.0)
    
    # Thompson sampling: draw a sample from the GP posterior
    # This gives a stochastic view of the function landscape
    rng = np.random.default_rng()
    sample = mu + sigma * rng.standard_normal(n)
    
    # Bayesian Regret Minimization approach:
    # We want to find the point that minimizes expected regret, accounting for cost.
    # 
    # The expected regret at point x is: E[max(0, f(x) - f_best)]
    # For a Gaussian posterior, this can be computed analytically.
    # But we also need to account for cost: a point with lower cost is "cheaper" 
    # to evaluate, so we should prefer it when regret is comparable.
    
    # Compute expected improvement (EI) - standard BO acquisition
    improvement = f_best - mu
    z = improvement / sigma
    z_clipped = np.clip(z, -50, 50)
    
    ei = improvement * _norm_cdf(z_clipped) + sigma * _norm_pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: reward improvement per unit cost
    ei_cost = ei / cost
    
    # Thompson sampling score: use the sampled function value
    # Lower sampled value = better (we're minimizing)
    ts_score = -sample
    
    # Combine: 
    # - Primary: EI normalized by cost (exploitation with cost awareness)
    # - Secondary: Thompson sample (stochastic exploration)
    # - Tertiary: uncertainty bonus (encourage exploration in uncertain regions)
    
    # Scale EI to be comparable with sample values
    # EI is typically small (0 to a few units), sample is in objective space
    # Normalize EI by a characteristic scale
    ei_scale = np.std(mu) + 1e-10
    ei_normalized = ei / ei_scale
    
    # Cost efficiency: how much improvement per unit cost
    cost_efficiency = ei_cost / (np.std(ei_cost) + 1e-10) if np.std(ei_cost) > 1e-10 else 0.0
    
    # Uncertainty bonus: reward high sigma (exploration)
    uncertainty = sigma / (np.std(mu) + 1e-10)
    
    # Final score: blend of cost-aware EI, Thompson sampling, and exploration
    # Weight toward cost-aware EI for exploitation, with Thompson for exploration
    scores = (0.5 * ei_normalized + 
              0.3 * cost_efficiency + 
              0.2 * uncertainty - 
              0.1 * sample / (np.std(sample) + 1e-10))
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
