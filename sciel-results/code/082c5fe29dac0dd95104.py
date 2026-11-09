# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as scipy_t
import time

_start_time = time.time()
_DEADLINE = 300.0  # Leave margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Student-t Process (STP) approximation.
    
    Structural Change:
    Replaces the Gaussian GP posterior assumption with a Student-t Process approximation.
    This uses heavier-tailed predictive distributions (3 degrees of freedom) to better
    model epistemic uncertainty in regions with sharp curvature or high noise, such as
    the Beale function. The acquisition score is a blend of the Probability of Improvement
    (computed with t-distribution CDF) and a UCB term (using t-distribution scale).
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency fallback: pure exploitation
        return -np.asarray(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Student-t Process parameters
    nu = 3.0  # Degrees of freedom, 3 gives heavy tails
    
    # Convert standard deviation (sigma) to scale parameter (s) for t-distribution
    # std_dev = s * sqrt(nu / (nu - 2))  =>  s = std_dev * sqrt((nu - 2) / nu)
    scale_factor = np.sqrt((nu - 2.0) / nu)
    s = sigma_safe * scale_factor
    
    # Compute z-score for t-distribution
    # z = (f_best - mu) / s
    z = (f_best - mu) / s
    
    # Probability of Improvement (PI) using t-distribution CDF
    # P(f(x) < f_best) = P(t_nu < z)
    pi = scipy_t.cdf(z, df=nu)
    
    # UCB for minimization: -mu + beta * s
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    # f_best ranges from ~355 (max) to 0 (min)
    f_best_clamped = max(f_best, 1e-12)
    
    # Beta decay:
    # Early (f_best high): beta ~ 2.5 (explore)
    # Late (f_best low): beta ~ 0.5 (exploit)
    # Using a smooth decay: beta = a + b / (1 + f_best)
    a, b = 0.5, 2.0
    beta = a + b / (1.0 + f_best_clamped)
    
    ucb = -mu + beta * s
    
    # Blend strategy based on optimization progress
    # progress = 0 when f_best is large, 1 when f_best is near 0
    progress = 1.0 / (1.0 + f_best_clamped)
    
    # Weights: 
    # Early: more UCB weight; Late: more PI/EI weight
    # PI is a bounded [0,1] score, UCB is unbounded. We need to normalize or scale them.
    # To avoid scaling issues, we can normalize UCB to have similar magnitude to PI.
    # Or, we can use a weighted sum where we scale PI by a factor to match UCB's typical range.
    # However, a simpler robust method is to normalize both to zero mean and unit variance, then blend.
    
    def normalize(arr):
        std = np.std(arr)
        if std < 1e-12:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std
    
    ucb_norm = normalize(ucb)
    pi_norm = normalize(pi)
    
    # Weights for blending
    # Early: UCB dominates. Late: PI dominates (since PI captures the probability of finding a better point)
    w_ucb = 0.7 * (1.0 - progress) + 0.3 * progress
    w_pi = 0.3 * (1.0 - progress) + 0.7 * progress
    
    score = w_ucb * ucb_norm + w_pi * pi_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
