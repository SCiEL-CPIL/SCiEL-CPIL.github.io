# EVOLVE-BLOCK-START
import numpy as np

# Global random state for Thompson Sampling
# Using a fixed seed ensures reproducibility of the "noise pattern" sequence
# across runs of the program, which helps in benchmarking consistency.
_ts_rng = np.random.RandomState(42)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Samples a latent function value for each candidate from the GP predictive
    distribution. Scores candidates based on the expected improvement over
    f_best, penalized by evaluation cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviation predictions
        f_best: lowest observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Draw standard normal noise for Thompson Sampling
    # This uses the global RNG state, so the noise changes each call
    z = _ts_rng.standard_normal(size=mu.shape)
    
    # Sample latent function values
    y = mu + sigma * z
    
    # Calculate score:
    # We want to minimize y. An improvement is (f_best - y).
    # Positive improvement means y < f_best.
    # We divide by cost to penalize expensive evaluations.
    # This maximizes the "improvement per unit cost".
    scores = (f_best - y) / cost
    
    # Replace any non-finite values with a very low score
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
