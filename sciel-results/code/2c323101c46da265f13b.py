# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-conditioned acquisition: Quantile-LCB (10th percentile) with
    a cost-aware exploration bonus that scales inversely with cost,
    effectively querying the posterior at a hypothetical zero-cost point.
    
    score(x) = [f_best - mu - beta*sigma] / cost + gamma * (1/cost) * sigma
    
    The first term is a cost-normalized LCB at the 10th percentile.
    The second term is an exploration bonus that is larger for cheap
    evaluations, encouraging the optimizer to gather information
    where it is most affordable.
    """
    t_start = time.time()
    TIME_LIMIT = 300.0  # Safety margin before 330s kill

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # --- Quantile-LCB at 10th percentile ---
    # For a normal posterior, the 10th percentile is mu - 1.2816*sigma
    # LCB_10 = mu - 1.2816 * sigma
    # Improvement potential: f_best - LCB_10 = f_best - mu + 1.2816*sigma
    
    z_10 = 1.2816  # z-score for 10th percentile (norm.ppf(0.10))
    
    # Cost-normalized improvement potential (LCB-based)
    lcb_improvement = np.maximum(f_best - mu_safe + z_10 * sigma_safe, 0.0)
    score_lcb = lcb_improvement / cost_safe
    
    # --- Cost-aware exploration bonus ---
    # This term rewards high uncertainty at low cost, effectively
    # extrapolating to a "zero-cost" scenario where exploration is free
    # The bonus is proportional to sigma/cost, which is the posterior
    # std dev per unit cost
    gamma = 0.5
    exploration_bonus = gamma * sigma_safe / cost_safe
    
    # Combine: LCB score + exploration bonus
    scores = score_lcb + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time check
    if time.time() - t_start > TIME_LIMIT:
        return scores

    return scores
# EVOLVE-BLOCK-END
