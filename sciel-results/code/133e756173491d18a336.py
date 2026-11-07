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
    
    # Use a combination of Expected Improvement and an exploration term
    # EI is optimal for minimizing expected regret.
    # We add a small UCB component to ensure exploration in early stages.
    
    # UCB term: -mu + kappa * sigma
    # Adaptive kappa: start with high exploration, decay over time
    kappa = 1.0 * np.exp(-t / 15.0)
    
    ucb = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales?
    # EI is in objective units. UCB is in objective units.
    # They should be comparable.
    
    # Blend: primarily EI, with some UCB
    # As t increases, rely more on EI
    alpha_ucb = 0.5 * np.exp(-t / 10.0)
    
    scores = (1.0 - alpha_ucb) * ei + alpha_ucb * ucb
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
