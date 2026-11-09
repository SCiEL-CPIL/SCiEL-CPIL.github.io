"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Hypervolume-inspired cost-aware acquisition for Bayesian optimization.
    
    Estimates the expected volume of the objective space dominated by each
    candidate, normalized by cost. This shifts from point-wise improvement
    metrics to a set-based utility that rewards candidates which would expand
    the region of known-good values, helping navigate the Rosenbrock valley
    by balancing exploitation of high-probability improvements with exploration
    of uncertain regions that could reveal large improvements.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    z_safe = np.clip(z, -10, 10)
    
    # Probability of improvement: P(f(x) < f_best)
    p_improve = norm.cdf(z_safe)
    p_improve = np.clip(p_improve, 1e-10, 1.0 - 1e-10)
    
    # Expected improvement for minimization
    ei = (f_best - mu) * norm.cdf(z_safe) + sigma_safe * norm.pdf(z_safe)
    ei = np.maximum(ei, 0)
    
    # Hypervolume-inspired component:
    # The expected improvement already captures the "volume" of improvement,
    # but we can enhance it by considering the uncertainty-weighted potential.
    # A candidate with high sigma and low mu has high potential to dominate
    # a large region of the objective space.
    
    # Scale EI by a factor that accounts for the "spread" of potential improvements
    # Candidates with higher sigma have more potential to reveal large improvements
    # even if their expected improvement is moderate.
    # Use a normalized uncertainty factor.
    f_scale = max(abs(f_best), 1e-10)
    sigma_norm = sigma_safe / f_scale
    sigma_norm = np.clip(sigma_norm, 0, 5.0)
    
    # Hypervolume proxy: EI enhanced by uncertainty potential
    # The idea is that high-uncertainty regions with low means could dominate
    # a large volume of the objective space if the true function is low there.
    # We use a soft weighting: hv = EI * (1 + alpha * sigma_norm)
    # where alpha controls how much we reward uncertainty potential.
    alpha = 0.5
    hv_proxy = ei * (1.0 + alpha * sigma_norm)
    
    # Normalize hv_proxy to [0, 1]
    hv_max = np.max(hv_proxy)
    if hv_max > 1e-10:
        hv_norm = hv_proxy / hv_max
    else:
        hv_norm = np.zeros_like(hv_proxy)
    
    # Combine hypervolume proxy with probability of improvement
    # POI provides a stable, bounded signal; HV proxy provides magnitude
    # and uncertainty-aware potential.
    # Weighting: emphasize HV proxy for its set-based utility, but keep POI
    # for stability in regions where EI might be noisy.
    w_hv = 0.6
    w_poi = 0.4
    combined = w_hv * hv_norm + w_poi * p_improve
    
    # Cost-aware normalization: divide by cost to prioritize efficient evaluations
    # The cost is in [1, 2], so 1/cost is in [0.5, 1]
    score = combined / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
