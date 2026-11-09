# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as t_dist
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Student-t Expected Improvement (EI).
    Uses Student-t distribution for the predictive model to account for
    heavier tails, which can be beneficial for functions with sharp
    curvature like the Beale function.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Degrees of freedom for Student-t distribution
    # Lower nu means heavier tails (more exploration)
    nu = 3.0 

    # Compute z-score
    # Note: In Student-t parameterization, if we treat sigma as the scale parameter,
    # z = (x - mu) / sigma
    z = (f_best - mu) / sigma_safe

    # Student-t CDF and PDF
    # scipy.stats.t.cdf takes (x, df)
    cdf_t = t_dist.cdf(z, nu)
    pdf_t = t_dist.pdf(z, nu)

    # Expected Improvement for minimization
    # EI = E[max(f_best - f, 0)]
    # If f ~ t(mu, sigma), let Y = f - mu ~ t(0, sigma)
    # f_best - f = f_best - mu - Y
    # We want E[max(f_best - mu - Y, 0)]
    # Let A = f_best - mu
    # E[max(A - Y, 0)]
    # Integral from -inf to A of (A - y) pdf(y) dy
    # = A * P(Y <= A) - E[Y 1(Y <= A)]
    # For symmetric t-dist, E[Y 1(Y <= A)] = -sigma * pdf_t(A/sigma) * (1 - cdf_t(A/sigma))? 
    # Actually, standard formula for Gaussian is (A)Phi(z) + sigma phi(z).
    # For Student-t, the formula is similar:
    # EI = (f_best - mu) * cdf_t(z) + sigma * pdf_t(z)
    
    ei = (f_best - mu) * cdf_t + sigma_safe * pdf_t

    # Ensure non-negative EI (numerical stability)
    ei = np.maximum(ei, 0.0)

    scores = ei

    # Check time budget
    if time.time() > deadline:
        # Return best-effort result
        return scores

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
