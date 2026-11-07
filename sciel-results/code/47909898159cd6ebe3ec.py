"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    if time.time() - _start_time > 325:
        return -mu
    
    t = _call_count
    n = len(mu)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Knowledge Gradient (KG) acquisition:
    # KG estimates the expected reduction in the observed minimum if we evaluate at x.
    # For minimization, KG(x) = E[max(f_best, min(f(x), f_other)) - min(f(x), f_other)]
    # where f_other represents the other candidates.
    # 
    # A practical approximation: the probability that candidate i becomes the new minimum
    # weighted by the expected improvement it would provide.
    #
    # We compute: for each candidate i, P(i is the new minimum) * E[improvement | i is new min]
    #
    # P(i is new minimum) = P(f_i < f_best AND f_i < f_j for all j != i)
    # Approximation: P(f_i < f_best) * P(f_i < f_j for all j != i)
    #
    # A simpler but effective KG approximation:
    # KG_i ≈ E[max(0, f_best - f_i)] * P(f_i is the minimum among all candidates)
    #
    # P(f_i is min) can be approximated by the probability that f_i < f_best
    # times a correction for being the best among all candidates.
    
    # Step 1: Compute z-scores relative to f_best
    z = (f_best - mu) / sigma_safe
    
    # Step 2: Probability that candidate i improves upon f_best
    p_improve = norm.cdf(z)
    
    # Step 3: Expected improvement if candidate i is evaluated
    ei = (f_best - mu) * p_improve + sigma_safe * norm.pdf(z)
    
    # Step 4: Approximate probability that candidate i is the minimum among all 2000 candidates
    # This requires considering pairwise comparisons. A tractable approximation:
    # P(i is min) ≈ p_improve * (1 / (1 + sum_{j!=i} P(j < i)))
    # 
    # For efficiency, we use a rank-based approximation:
    # The probability that i is the minimum is related to its relative position.
    # We approximate P(i is min) by considering the expected rank.
    #
    # A simpler structural approach: use the "expected minimum" formulation.
    # KG can be approximated as:
    # KG_i ≈ E[f_best - min(f_i, f_other)] where f_other is the min of other candidates
    #
    # Practical approximation: 
    # KG_i ≈ ei_i * w_i where w_i accounts for the chance of being the overall minimum
    #
    # We compute a soft-min weight: how likely is candidate i to be the best?
    # Using the fact that if mu_i is low and sigma_i is small, it's more likely to be the min.
    
    # Compute a "competitiveness" score for each candidate
    # Candidates with low mu and low sigma are more likely to be the true minimum
    # We use a softmax-like weighting over the negative mu values
    
    # Normalize mu to get relative competitiveness
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # Competitiveness: lower mu = more competitive
    # Scale by sigma to account for uncertainty
    competitiveness = (mu_max - mu) / mu_range  # 1 for best, 0 for worst
    
    # Combine with improvement probability
    # A candidate that is both likely to improve AND likely to be the minimum gets high KG
    # The "likely to be minimum" part: use a temperature-scaled softmax
    # Lower temperature = more selective (favor the best candidates)
    
    # Adaptive temperature: early on, be more selective to find good regions
    # Later, be less selective to refine
    temperature = 0.1 + 0.5 * (t / 25.0)
    
    # Soft-min weight: exp(-mu / temperature) normalized
    # This gives higher weight to candidates with lower mu
    exp_scores = np.exp(-mu / (temperature + 1e-9))
    exp_scores = np.where(np.isfinite(exp_scores), exp_scores, 0.0)
    weight_sum = np.sum(exp_scores)
    if weight_sum < 1e-30:
        weight_sum = 1e-30
    soft_min_weight = exp_scores / weight_sum
    
    # KG approximation: expected improvement weighted by probability of being the minimum
    # This directly targets the simple regret metric
    kg_scores = ei * soft_min_weight
    
    # Add a small exploration term to avoid getting stuck
    # The exploration term rewards high uncertainty in regions that could improve
    exploration = sigma_safe * p_improve * (1.0 - soft_min_weight)
    
    # Adaptive blend: more KG early, more exploration if needed
    # But primarily rely on KG since it directly optimizes the target metric
    alpha_exploration = 0.1 / (1.0 + 0.1 * t)
    
    scores = kg_scores + alpha_exploration * exploration
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
