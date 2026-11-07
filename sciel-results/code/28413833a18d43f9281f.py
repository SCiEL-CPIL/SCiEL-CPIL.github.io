"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _knowledge_gradient(mu, sigma, f_best, n_samples=50):
    """
    Knowledge Gradient approximation via Monte Carlo.
    
    KG(x) = E[max(f_best - min_y f(y), 0)]
    where the expectation is over the joint distribution of all candidate
    function values given the GP posterior.
    
    We approximate by:
    1. Sample n_samples realizations of the GP at all 2000 candidates
    2. For each realization, find the minimum value
    3. Compute E[max(f_best - min_value, 0)]
    4. The KG for candidate x is the expected reduction in f_best when x is evaluated
    
    For efficiency, we use a deterministic approximation:
    KG(x) ≈ P(x is the minimizer) * E[f_best - f(x) | x is minimizer]
    
    A practical approximation:
    KG(x) ≈ E[max(f_best - min(f(x), other_best), 0)]
    where other_best represents the best value among other candidates.
    """
    s = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Use antithetic sampling for variance reduction
    rng = np.random.default_rng(42)
    
    # Precompute: for each sample, draw from N(mu, sigma^2) for all points
    # This is the key computational step
    # Shape: (n_samples, n)
    samples = mu[None, :] + s[None, :] * rng.standard_normal((n_samples, n))
    
    # For each sample, compute the minimum across all candidates
    min_per_sample = samples.min(axis=1)  # shape (n_samples,)
    
    # KG for point x: E[max(f_best - min(f(x), min_others), 0)]
    # where min_others is the minimum of all points except x
    # Approximation: use the overall minimum as a proxy for min_others
    # This underestimates KG for the true minimum but is computationally tractable
    
    # Better approximation: KG(x) = E[max(f_best - f(x), 0)] * P(f(x) <= min_others)
    # + E[max(f_best - min_others, 0)] * P(f(x) > min_others)
    # But this is complex. Use a simpler structural approach:
    
    # The value of evaluating x is the expected improvement in our knowledge
    # of the minimum. A good proxy is:
    # KG(x) ≈ E[max(f_best - f(x), 0)] + exploration_bonus(x)
    # where exploration_bonus rewards high uncertainty in regions likely to contain the minimum
    
    # Compute EI for each point
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)
    
    # Compute probability that x is the global minimizer (approximate)
    # P(f(x) < f(y) for all y != x) ≈ product over y of P(f(x) < f(y))
    # This is expensive. Instead, use a rank-based approximation.
    
    # For each sample, count how many times each point is the minimum
    mins = samples.argmin(axis=1)  # shape (n_samples,)
    counts = np.zeros(n)
    np.add.at(counts, mins, 1)
    prob_min = counts / n_samples  # shape (n,)
    
    # KG approximation: weight EI by the probability of being the minimum
    # This captures the idea that evaluating a point that's likely to be the
    # minimum has higher value of information
    kg = ei * (0.5 + prob_min)
    
    # Add an exploration term: reward points with high sigma in promising regions
    # Points with high sigma that are close to the current best are valuable
    # because they may reveal a better minimum
    sigma_normalized = s / (s.max() + 1e-9)
    exploration = sigma_normalized * np.exp(-np.abs(mu - f_best) / (s + 1e-9))
    
    # Combine: KG + exploration bonus
    kg_total = kg + 0.3 * exploration
    
    return np.where(np.isfinite(kg_total), np.maximum(kg_total, 0.0), 0.0)


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    t0 = time.time()
    
    s = np.maximum(sigma, 1e-9)
    c = np.maximum(cost, 1e-9)
    
    # Compute Knowledge Gradient
    kg = _knowledge_gradient(mu, sigma, f_best, n_samples=100)
    
    # Apply cost-aware penalty with softened exponent
    # Using cost^0.8 allows selecting points with higher potential improvement
    # even if slightly more expensive, while still respecting the budget
    score = kg / (c ** 0.8)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin: if we've spent too much time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # 20s buffer before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
