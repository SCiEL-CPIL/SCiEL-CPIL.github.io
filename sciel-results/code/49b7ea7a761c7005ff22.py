"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    KG estimates the expected reduction in the global minimum's value
    by approximating the posterior distribution of the minimum over
    the 2000 candidates.
    
    For each candidate i, we want E[max(0, f_best - min_j X_j)] where
    X_j ~ N(mu_j, sigma_j^2) are the posterior predictions.
    
    A common efficient approximation: for each candidate i, compute
    the expected improvement of the minimum given that candidate i is
    the next point to evaluate. This can be approximated by:
    
    KG_i ≈ E[max(0, f_best - min(X_i, min_{j≠i} X_j))]
    
    A simpler and well-performing approximation is to use the fact that
    min of independent normals is approximately normal with:
    mu_min ≈ min(mu_j)
    sigma_min ≈ min(sigma_j)
    
    But a better KG approximation considers that evaluating point i
    updates our belief about the global minimum. The expected value
    of the minimum after evaluating at point i is:
    
    E[min(X_i, min_{j≠i} X_j)]
    
    where X_i ~ N(mu_i, sigma_i^2) and the other X_j are the current
    posterior predictions.
    
    A practical KG approximation:
    KG_i = E[max(0, f_best - min(X_i, M))]
    where M = min_{j≠i} X_j
    
    Since M is the minimum of ~1999 Gaussians, we can approximate
    E[min(X_i, M)] using the fact that for two independent Gaussians
    X ~ N(m1, s1^2) and Y ~ N(m2, s2^2):
    E[min(X, Y)] = m1 - s1 * phi((m1-m2)/(sqrt(s1^2+s2^2))) * sqrt(1 + (m1-m2)^2/(s1^2+s2^2) * 0)
    
    Actually, a cleaner approach: use the standard KG approximation
    where we treat the minimum of all other points as a single Gaussian
    with mean = min(mu) and std = min(sigma), then compute the
    expected improvement of the minimum.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # For KG: estimate expected reduction in f_best
    # Approximation: for each point i, the expected value of 
    # min(X_i, min_{j≠i} X_j) where X_j ~ N(mu_j, sigma_j^2)
    
    # Efficient approximation: 
    # The minimum of N independent Gaussians is approximately
    # N(mu_min, sigma_min^2) where mu_min = min(mu) and 
    # sigma_min is related to the smallest sigmas.
    
    # Better: use the fact that E[min(X_i, M)] where M = min of others
    # can be computed if we know the distribution of M.
    
    # Practical KG: For each candidate, compute the expected improvement
    # of the global minimum. A well-known efficient approximation:
    
    # KG_i ≈ E[max(0, f_best - X_i)] - E[max(0, f_best - min(X_i, M))]
    # This is getting complex. Let's use a simpler but effective form.
    
    # Use the "expected minimum improvement" approximation:
    # The probability that X_i < f_best is Phi((f_best - mu_i)/sigma_i)
    # The expected value of X_i | X_i < f_best is mu_i - sigma_i * phi(z)/Phi(z)
    # where z = (f_best - mu_i) / sigma_i
    
    # KG approximation: the expected reduction in the best value
    # is approximately the expected improvement of the minimum.
    
    # A clean and effective KG-like score:
    # For each point, compute the expected minimum of (X_i, current best)
    # and the expected reduction.
    
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    # Expected improvement of point i over f_best
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # KG correction: account for the fact that the minimum of many
    # points is likely much lower than f_best. The more uncertain
    # the point, the more it can reduce the global minimum.
    
    # Approximate the "effective" f_best as lower due to other candidates
    # Use a simple scaling: the more candidates, the lower the expected minimum
    n = len(mu)
    
    # The expected minimum of n Gaussians with similar parameters
    # is approximately mu_min - sigma_min * sqrt(2*log(n))
    mu_min = np.min(mu)
    sigma_min = np.min(sigma_safe)
    
    # Expected minimum of all candidates (rough)
    expected_min = mu_min - sigma_min * np.sqrt(2.0 * np.log(max(n, 2)))
    
    # Effective best: use min(f_best, expected_min) as a reference
    effective_best = min(f_best, expected_min)
    
    # Recompute EI with effective best
    z_eff = (effective_best - mu) / sigma_safe
    z_eff = np.clip(z_eff, -50, 50)
    
    ei_eff = (effective_best - mu) * norm.cdf(z_eff) + sigma_safe * norm.pdf(z_eff)
    
    # Blend original EI and effective EI
    # Weight toward EI_eff when the expected minimum is well below f_best
    weight = 0.5 * (1.0 + np.tanh((f_best - expected_min) / (sigma_min + 1e-9)))
    
    scores = weight * ei_eff + (1.0 - weight) * ei
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
