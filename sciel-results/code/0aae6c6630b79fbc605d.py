"""Rewritten cost-aware acquisition for Griewank 2D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a two-phase strategy:
    - Early: aggressive exploration via upper confidence bound
    - Late: exploitation via expected improvement
    
    Uses adaptive switching based on f_best and cost budget.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    f_best_val = float(f_best)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-8)

    # Compute z-scores: how many standard deviations below f_best the mean is
    z = (f_best_val - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)

    # Expected Improvement (EI) for minimization
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best_val - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)

    # Probability of Improvement (PI)
    pi = Phi_z

    # Upper Confidence Bound (UCB) for exploration
    # beta adapts: larger when f_best is far from 0 (early), smaller when close
    # f_best is in [0, 1] for Griewank
    f_best_clipped = np.clip(f_best_val, 0.0, 1.0)
    # beta: starts high (~2.5) when f_best ~ 1, drops to ~0.5 when f_best ~ 0
    beta = 0.5 + 2.0 * f_best_clipped
    ucb = mu - beta * sigma_safe  # For minimization, we want low values

    # Cost-normalized scores
    ei_score = ei / cost_safe
    pi_score = pi / cost_safe
    ucb_score = -ucb / cost_safe  # Negate because we want high scores, and lower ucb is better

    # Adaptive phase switching:
    # Phase 1 (f_best > 0.3): explore with UCB + some EI
    # Phase 2 (f_best <= 0.3): exploit with EI + some PI
    if f_best_clipped > 0.3:
        # Exploration phase
        w_ucb = 0.55
        w_ei = 0.35
        w_pi = 0.10
    elif f_best_clipped > 0.1:
        # Transition phase
        w_ucb = 0.30
        w_ei = 0.50
        w_pi = 0.20
    else:
        # Exploitation phase
        w_ucb = 0.10
        w_ei = 0.60
        w_pi = 0.30

    # Combine scores
    scores = w_ucb * ucb_score + w_ei * ei_score + w_pi * pi_score

    # Deadline guard
    if time.time() > deadline:
        # Fallback: simple exploitation
        scores = ei / cost_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
