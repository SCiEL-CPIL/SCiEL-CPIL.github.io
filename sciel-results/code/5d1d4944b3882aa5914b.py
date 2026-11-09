"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling surrogate for cost-aware Bayesian optimization.
    
    We approximate the posterior by drawing a single Monte Carlo sample from
    the GP posterior at each candidate point: y ~ N(mu, sigma^2).
    We then score each candidate by the expected minimum of (y / cost) over
    the Monte Carlo samples. Since we draw one sample per candidate, we can
    approximate the expected minimum by considering the sampled value normalized
    by cost. To balance exploration and exploitation, we use the sampled value
    directly (lower is better for the objective), but we want higher acquisition
    scores to indicate better candidates. So we score as -sampled_value / cost,
    which naturally balances exploration (high variance leads to more extreme
    samples) and cost efficiency.
    
    However, a single sample per candidate is noisy. A better approach: draw
    multiple samples per candidate and take the mean of the top-k minimum
    cost-normalized values, or simply use the expected value of the minimum
    over the samples. With one sample, we just use -y/cost where y ~ N(mu, sigma^2).
    
    Actually, to be more robust, let's draw a small number of samples (e.g., 10)
    per candidate and compute the expected minimum of y/cost over those samples.
    This gives a smoother acquisition function while still capturing exploration.
    """
    n_candidates = len(mu)
    n_samples = 10  # Number of Monte Carlo samples per candidate
    
    # Reshape for broadcasting: (n_samples, n_candidates)
    mu_b = mu.reshape(1, -1)
    sigma_b = sigma.reshape(1, -1)
    
    # Draw samples: (n_samples, n_candidates)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu_b, scale=sigma_b, size=(n_samples, n_candidates))
    
    # Cost-normalized objective: lower is better
    cost_b = cost.reshape(1, -1)
    normalized = samples / cost_b  # (n_samples, n_candidates)
    
    # Expected minimum over samples for each candidate
    # We want the expected minimum of the normalized values
    expected_min = np.min(normalized, axis=0)  # Conservative: use the min
    
    # Higher acquisition = better candidate
    # Since lower normalized value is better, we negate
    scores = -expected_min
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
