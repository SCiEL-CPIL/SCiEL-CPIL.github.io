"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Thompson Sampling with cost normalization.
    
    Structural change: Replaces deterministic EI+UCB with stochastic Thompson
    Sampling. Each candidate is scored by drawing a sample from the posterior
    N(mu, sigma) and computing the expected improvement of that sample,
    then dividing by cost. This naturally balances exploration (via the
    stochastic draw) and exploitation (via the EI of the sample) without
    needing to tune a beta parameter. The cost normalization (division by cost)
    is the proven strategy from the best-performing inspirations.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margins
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Thompson Sampling: draw one sample from the posterior for each candidate
    # Use a fixed random seed per call? No, we want stochasticity.
    # But for reproducibility within a single acquisition call, we just draw.
    # The stochasticity across calls provides the exploration.
    
    # Draw samples from N(mu, sigma)
    # We need to be careful: sigma is the std, so sample = mu + sigma * randn
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Compute EI for each sample: E[max(f_best - sample, 0)]
    # Since the sample is a single draw, the "expected improvement" of that
    # specific sample is just max(f_best - sample, 0). But that's a point estimate.
    # Actually, Thompson Sampling typically just minimizes the sampled objective.
    # But here we want a score. Let's think:
    # If we minimize the sampled objective, the score should be -sample / cost.
    # But we also want to incorporate the fact that we're doing BO with f_best.
    # 
    # Alternative: Use the EI formula but with the sampled value as the "mean".
    # That doesn't quite make sense.
    #
    # Let's reconsider. The standard Thompson Sampling for minimization:
    # 1. Draw y_i ~ N(mu_i, sigma_i^2) for each candidate i.
    # 2. Select the candidate with the smallest y_i (or smallest y_i / cost_i).
    #
    # So the score should be -samples / cost_safe (to maximize).
    # But this doesn't use f_best at all, which is odd for BO.
    #
    # Actually, in cost-aware BO, we want to maximize expected improvement per cost.
    # With Thompson Sampling, we can compute the EI of the sample:
    # EI_sample = max(f_best - sample, 0)
    # But this is a point estimate, not an expectation.
    #
    # A better approach: Use the standard EI formula (which is the expected
    # improvement under the GP posterior) and divide by cost. This is the
    # "proven" strategy. Then add a small Thompson Sampling-like exploration
    # term by adding a small random perturbation to the EI score.
    #
    # Wait, the directive says to REPLACE the pointwise acquisition with
    # Thompson Sampling. Let's do it properly:
    #
    # Score_i = (f_best - sample_i) / cost_i if sample_i < f_best, else 0
    # But this is still a point estimate.
    #
    # Actually, the cleanest Thompson Sampling approach for BO:
    # 1. Draw y_i ~ N(mu_i, sigma_i^2)
    # 2. Score_i = max(f_best - y_i, 0) / cost_i
    # This uses the sampled value to estimate the improvement, and normalizes by cost.
    # The stochasticity of the draw provides exploration.
    
    # Compute the sampled improvement
    sampled_improvement = np.maximum(f_best - samples, 0.0)
    
    # Normalize by cost
    score = sampled_improvement / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 280.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
