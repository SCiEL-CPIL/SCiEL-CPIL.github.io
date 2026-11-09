"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Batch-aware Thompson Sampling heuristic with cost-awareness.
    # Sample a single function value from the GP posterior for each candidate.
    # Use a deterministic pseudo-random sample based on index for reproducibility
    # and to avoid stochastic variability across calls.
    # The sample is: y = mu - sigma * z, where z ~ N(0,1)
    # We want to maximize expected improvement per unit cost.
    # Improvement: max(0, f_best - y)
    # For a sample y, the improvement is max(0, f_best - y)
    # We use a single sample per candidate (Thompson Sampling style)
    # and divide by cost to make it cost-aware.
    
    # Generate deterministic pseudo-random normal samples
    # Use a simple hash-based approach to get reproducible "random" z values
    # This avoids needing a fixed seed that might not be available
    n = len(mu)
    
    # Create a deterministic pseudo-random sequence
    # Using a simple LCG-like approach with index-based seeding
    # Actually, let's use np.random with a fixed seed derived from the call
    # But we need it to be consistent. Let's use a simple approach:
    # z_i = norm.ppf((i + 0.5) / n) would be deterministic quantiles
    # But that's not really sampling. Let's use a proper approach.
    
    # For Thompson Sampling, we sample once per candidate.
    # To make it deterministic and reproducible, we can use the candidate index
    # to generate a pseudo-random number.
    
    # Simple approach: use a fixed-seed RNG
    rng = np.random.RandomState(42)
    z = rng.standard_normal(n)
    
    # Sampled function values: lower is better, so we sample y = mu - sigma * z
    # This means when z > 0, we sample a lower (better) value
    y = mu - sigma * z
    
    # Improvement relative to current best
    improvement = np.maximum(0, f_best - y)
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    score = improvement / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
