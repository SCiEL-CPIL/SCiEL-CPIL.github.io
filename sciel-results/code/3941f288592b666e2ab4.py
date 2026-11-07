import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Expected Hypervolume Improvement (CE-EHI) acquisition.
    
    Structural change: instead of single-objective EI + UCB, we treat (cost, objective)
    as a bi-objective minimization problem and compute expected hypervolume improvement
    relative to a reference point derived from f_best and the cost range. The expected
    HV gain is divided by the evaluation cost to explicitly optimize the trade-off
    between objective progress and budget consumption.
    
    For each candidate i, we approximate E[HV_i] by sampling the latent objective
    y_i ~ N(mu_i, sigma_i) and computing the hypervolume contribution of the point
    (cost_i, y_i) relative to a reference point, minus the current best HV.
    We use Monte Carlo with a fixed number of samples for stability and speed.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-8)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    sigma = np.maximum(sigma, 1e-8)
    
    n = len(mu)
    
    # Reference point for hypervolume: use a point that is worse than all observed
    # points in both objectives. For objective, use f_best + margin; for cost, use max cost.
    # The margin should be large enough that improvement is meaningful.
    obj_ref = f_best + 1.0  # Reference objective value (worse than best)
    cost_ref = np.max(cost) + 0.5  # Reference cost value (worse than max observed)
    
    # Number of MC samples for approximating E[HV]
    n_mc = 50
    
    # Pre-generate MC samples for all candidates at once for speed
    # Shape: (n, n_mc)
    mc_samples = mu[:, None] + sigma[:, None] * np.random.randn(n, n_mc)
    
    # Compute hypervolume contribution for each MC sample
    # For a point (c, y) to dominate the reference (cost_ref, obj_ref),
    # it must have c < cost_ref AND y < obj_ref.
    # The HV contribution of a single point (c, y) relative to reference is:
    # (cost_ref - c) * (obj_ref - y) if both are positive, else 0.
    # But we need the *improvement* over the current best, which is the HV of the
    # non-dominated set including this point minus the HV without it.
    # 
    # For simplicity and speed, we approximate E[HV_improvement] as:
    # E[max(0, (cost_ref - cost_i) * (obj_ref - y_i) - (cost_ref - cost_best) * (obj_ref - f_best))]
    # where the second term is the "baseline" HV of the current best point.
    # This is a lower bound on the true HVI but captures the trade-off.
    
    # Baseline HV of current best (using average cost as proxy for the best point's cost)
    # Actually, the best point's cost is unknown to us. We use the minimum cost among
    # candidates as a proxy, or just use a representative value.
    # Let's use the cost that corresponds to the best mu as a proxy.
    # For simplicity, use the median cost as a representative "current best" cost.
    cost_best_proxy = np.median(cost)
    baseline_hv = max(0.0, (cost_ref - cost_best_proxy) * (obj_ref - f_best))
    
    # Compute HV for each MC sample
    hv_samples = np.where(
        (mc_samples < obj_ref) & (cost[:, None] < cost_ref),
        (cost_ref - cost[:, None]) * (obj_ref - mc_samples),
        0.0
    )
    hv_samples = np.maximum(hv_samples, 0.0)
    
    # Expected HV improvement = E[hv_sample] - baseline_hv
    expected_hv = np.mean(hv_samples, axis=1) - baseline_hv
    expected_hv = np.maximum(expected_hv, 0.0)
    
    # Cost-effective: divide by cost to get HV per unit cost
    ce_ehi = expected_hv / cost
    
    # Add a small exploration bonus to avoid premature convergence
    # Use normalized sigma as an exploration term, weighted lightly
    sigma_max = np.max(sigma)
    if sigma_max > 1e-8:
        exploration_bonus = 0.1 * (sigma / sigma_max)
    else:
        exploration_bonus = np.zeros_like(sigma)
    
    scores = ce_ehi + exploration_bonus
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280.0:
        # Fallback: return simple cost-normalized EI
        improvement = f_best - mu
        z = improvement / np.maximum(sigma, 1e-8)
        z = np.clip(z, -10, 10)
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
