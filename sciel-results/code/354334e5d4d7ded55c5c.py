"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling-inspired acquisition function.
    
    Instead of computing a deterministic score like EI or UCB, we sample
    a single value from the posterior distribution for each candidate.
    The candidate with the lowest sampled value (since we are minimizing)
    will be selected by the evaluator's argmax logic if we invert the sign,
    BUT the task contract says "higher meaning evaluate next".
    
    Wait, the task says: "Return 2000 finite scores, higher meaning evaluate next".
    And the evaluator "evaluates the candidate with the highest score".
    
    If we use Thompson Sampling for minimization, we want to pick the point
    that has a high probability of being low.
    
    Standard Thompson Sampling for minimization:
    1. Sample theta ~ N(mu, sigma^2) for each candidate.
    2. Select the candidate with the LOWEST theta.
    
    However, the interface requires returning scores where HIGHER is better.
    Therefore, we should return -theta (negative of the sample).
    
    Additionally, the Beale function is non-negative (min value 0).
    The GP posterior might predict negative values in some regions due to
    approximation error or high uncertainty.
    Clipping the samples at 0 (i.e., max(sample, 0)) enforces this prior knowledge.
    This prevents the algorithm from selecting points where the GP thinks
    the function is negative (which is impossible) just because the variance
    is high enough to pull the mean down or the sample down.
    
    So the score for candidate i is:
    score_i = -max(mu_i + sigma_i * z_i, 0)
    where z_i ~ N(0, 1).
    
    This makes high scores correspond to low (non-negative) sampled values.
    """
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Generate standard normal samples
    # Using a fixed seed is not possible here as we don't know the call count,
    # but Thompson Sampling is inherently stochastic.
    # We rely on the global numpy random state.
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from the posterior
    samples = mu + sigma * z
    
    # Clip at 0 to enforce non-negativity of the Beale function
    # This is a structural change based on domain knowledge.
    samples = np.maximum(samples, 0.0)
    
    # We want to minimize the objective, so we want the lowest sample.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return the negative of the sample.
    score = -samples
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
