"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We maximize -mu - kappa * sigma, which favors regions with low predicted
    # mean and high uncertainty (exploration).
    #
    # kappa is chosen to balance exploitation and exploration. For Branin 2d
    # with the given GP settings, a moderate kappa around 1.5-2.5 tends to
    # work well. We use a slightly adaptive approach: start with a base kappa
    # and adjust based on the spread of sigma values.

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)

    # Base exploration parameter
    kappa_base = 1.8

    # Adaptive adjustment: if sigma values are very small overall (converged
    # region), reduce exploration; if they are large, keep it higher.
    # Use median sigma to gauge the typical uncertainty level.
    med_sigma = np.median(sigma_safe)
    
    # Scale kappa inversely with the median sigma to maintain consistent
    # exploration pressure. When sigma is small, we want less exploration
    # (kappa effectively smaller in impact), and vice versa.
    # But we want the *relative* contribution of kappa*sigma to be meaningful.
    # A simple approach: normalize kappa by a reference sigma scale.
    # For Branin, typical sigma values after some evaluations are O(0.1-1.0).
    # We'll use a soft scaling.
    
    # Actually, a cleaner approach: just use a fixed kappa that works well.
    # LCB with kappa=2.0 is a standard choice for moderate exploration.
    kappa = kappa_base

    # LCB score: higher is better for selection (we want low mu, high sigma)
    scores = -mu - kappa * sigma_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores
# EVOLVE-BLOCK-END
