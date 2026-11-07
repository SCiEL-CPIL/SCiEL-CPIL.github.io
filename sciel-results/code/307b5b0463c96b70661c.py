"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing sphere 1d on [-5, 5].
    Uses an adaptive hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    Early in the optimization (high f_best), it favors UCB for exploration.
    Late in the optimization (low f_best), it favors EI for exploitation.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Progress metric: 1.0 at start (f_best ~ 25), 0.0 at end (f_best ~ 0)
    # Max possible value for sphere on [-5, 5] is 25.
    f_best_clipped = np.clip(f_best, 0.0, 25.0)
    progress = f_best_clipped / 25.0
    
    # Adaptive Kappa for UCB
    # Early (progress ~ 1): kappa ~ 2.0 (High exploration)
    # Late (progress ~ 0): kappa ~ 0.5 (Low exploration)
    kappa = 0.5 + 1.5 * progress
    
    # 1. Expected Improvement (EI)
    # EI measures the expected improvement over f_best
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. UCB Score (to be maximized)
    # For minimization, UCB value is mu - kappa * sigma.
    # We maximize the negative of this: -mu + kappa * sigma.
    ucb_score = -mu + kappa * sigma_safe
    
    # 3. Adaptive Weights
    # Early: More UCB, Less EI
    # Late: More EI, Less UCB
    w_ei = 1.0 - progress
    w_ucb = progress
    
    # Combine scores
    # Both EI and UCB_score are in objective units.
    # Early on, UCB dominates due to larger scale and weight.
    # Late on, EI dominates due to weight and smaller UCB scale.
    score = w_ei * ei + w_ucb * ucb_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
