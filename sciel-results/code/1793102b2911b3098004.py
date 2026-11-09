"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard: 330s total budget, keep 10s margin
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback to simple EI if running out of time
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    n = len(mu)
    
    # Thompson Sampling: Sample a single function from the GP posterior
    # f(x) ~ N(mu(x), sigma(x))
    # We sample noise epsilon ~ N(0, 1)
    # f_sampled = mu + sigma * epsilon
    
    # To make it "batch-aware" or more robust, we can use a weighted combination
    # or simply sample. Standard TS is often better than EI for exploration in high dims.
    # However, simple TS might be too greedy if sigma is small everywhere.
    
    # Let's try a "Batch-Informed" Thompson Sampling approach.
    # Instead of just picking min(f_sampled), we consider that adding this point
    # will update the GP. But we don't have coordinates to update the GP here.
    # The prompt says "No coordinates are passed". So we cannot update the GP covariance.
    # This means we cannot do true Batch EI or Batch TS with GP updates.
    
    # Constraint: "No coordinates are passed."
    # This is a critical constraint. We only have mu and sigma at 2000 points.
    # We cannot recompute the GP posterior for new points or update it.
    # Therefore, any strategy requiring GP updates (like true batch acquisition) is impossible.
    # We are limited to pointwise acquisition functions based on mu and sigma.
    
    # Since we cannot do batch updates, the "batch-aware" inspiration is likely referring to
    # using a heuristic that approximates the benefit of adding a point, or perhaps
    # the "batch" refers to the 2000 candidates being treated as a set.
    
    # Let's look at Inspiration 1: "deterministic Pareto-front-based acquisition that dynamically balances normalized Expected Improvement (exploitation) and normalized Prediction Variance (exploration)"
    # This is a UCB-like approach but with dynamic weighting.
    
    # Let's try a **Dynamic UCB (Upper Confidence Bound)** with an adaptive exploration parameter.
    # Standard UCB: score = mu + beta * sigma
    # We want to minimize, so we want low mu and high sigma (uncertainty).
    # Actually, for minimization, we look for low predicted values.
    # LCB (Lower Confidence Bound) is standard for minimization: score = mu - beta * sigma.
    # Higher score means "more promising".
    # If we use LCB, we pick the point with the lowest possible value.
    # Wait, the task says "higher meaning evaluate next".
    # So if we use LCB, we should return -LCB? Or just use a formulation where higher is better.
    # EI returns higher values for better expected improvement.
    # LCB: mu - beta*sigma. Lower is better. So we can return -(mu - beta*sigma) = beta*sigma - mu.
    
    # The problem with fixed beta is it's hard to tune.
    # Let's use a **Normalized** approach.
    # Normalize mu and sigma to [0, 1] range? Or use standard scores?
    
    # Let's try **Probability of Improvement (PI)** combined with **Prediction Variance**.
    # Or better: **Expected Improvement** is already quite good (score 1.0).
    # Why did Inspiration 1 (1.1783) work? It used "deterministic Pareto-front-based".
    # This suggests balancing exploitation and exploration explicitly.
    
    # Let's implement a **Hybrid Acquisition**:
    # 1. Calculate EI.
    # 2. Calculate Prediction Variance (sigma).
    # 3. Normalize both to [0, 1] using min-max over the 2000 candidates.
    # 4. Combine with a dynamic weight.
    
    # However, min-max normalization is unstable if the range is tiny.
    # Let's use **Rank-based** normalization or **Z-score** normalization.
    
    # Strategy: **Adaptive UCB with Z-score normalization**
    # 1. Compute z_mu = (mu - mean_mu) / std_mu
    # 2. Compute z_sigma = (sigma - mean_sigma) / std_sigma
    # 3. Score = -z_mu + beta * z_sigma
    #    (We negate z_mu because lower mu is better. Higher z_sigma is better for exploration.)
    # 4. Beta can be adapted based on the number of evaluations or the spread of sigma.
    
    # Since we don't know the step count, we can adapt beta based on the current "confidence".
    # If sigma is generally high, we explore more.
    # Let's try a fixed but well-chosen beta, or a beta that decays?
    # We don't have access to step count.
    
    # Alternative: **Knowledge Gradient (KG)** approximation? Too complex without coords.
    
    # Let's stick to the **Pareto Front** idea from Inspiration 1.
    # Identify points that are non-dominated in (mu, -sigma) space?
    # No, we want low mu and high sigma.
    # So we want to minimize mu and maximize sigma.
    # Pareto front of (mu, -sigma).
    # Points on the front are "good" in at least one aspect.
    # We can score them based on their distance to the ideal point (min_mu, max_sigma).
    
    # Let's try a simpler structural change: **Thompson Sampling with a twist**.
    # Instead of sampling once, we can sample multiple times and take the median or mean?
    # No, TS is stochastic. The evaluator runs the function once per step.
    # The stochasticity of TS helps escape local minima.
    # EI is deterministic.
    
    # Let's try **Thompson Sampling** (TS).
    # f_sampled = mu + sigma * np.random.randn(n)
    # Return -f_sampled (since lower is better, higher score is better).
    # Wait, if I return -f_sampled, the highest score corresponds to the lowest sampled value.
    # This is pure exploration/exploitation balance inherent in TS.
    
    # Why might TS be better than EI?
    # EI can be overly greedy. TS has the right amount of randomness.
    # In high dimensions, TS often outperforms EI.
    
    # Let's implement TS.
    # To ensure stability, we might add a small amount of noise or regularization?
    # No, standard TS is:
    # sample = mu + sigma * N(0, 1)
    # score = -sample
    
    # Let's check if there's a risk of TS being too random.
    # If sigma is very small, it's greedy. If sigma is large, it's random.
    # This is exactly what we want.
    
    # One issue: The prompt says "Return 2000 finite scores".
    # TS scores are finite.
    
    # Let's try a **Weighted Combination of TS and EI**?
    # Or just TS.
    
    # Let's look at Inspiration 3 (1.0000): "scale-adaptive Expected Improvement".
    # It didn't improve.
    
    # Inspiration 1 (1.1783): "Pareto-front-based".
    # Let's try to implement a **Pareto-based score**.
    # 1. Identify non-dominated points in (mu, -sigma).
    # 2. For dominated points, give a lower score.
    # 3. For non-dominated points, score based on a utility function.
    
    # Actually, a simpler structural change is **Batch-Informed EI** is not possible.
    # So **Thompson Sampling** is the most distinct "design family" (Stochastic vs Deterministic).
    
    # Let's implement Thompson Sampling.
    
    # Add a small epsilon to sigma to avoid division by zero or numerical issues?
    # sigma is standard deviation, should be >= 0.
    
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    sampled = mu + sigma * eps
    
    # We want to minimize the objective.
    # So the "best" candidate has the lowest sampled value.
    # The acquisition function should return higher scores for better candidates.
    # So we return -sampled.
    
    return -sampled

# EVOLVE-BLOCK-END
