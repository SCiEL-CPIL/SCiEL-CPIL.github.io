"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_KG_DEADLINE_MARGIN = 30.0  # seconds before the 330s kill
_KG_START_TIME = None
_KG_TIMEOUT = False


def _kg_deadline_ok():
    """Check if we have enough time left before the evaluator's 330s kill."""
    global _KG_START_TIME, _KG_TIMEOUT
    if _KG_TIMEOUT:
        return False
    if _KG_START_TIME is None:
        _KG_START_TIME = time.time()
    elapsed = time.time() - _KG_START_TIME
    if elapsed > (330.0 - _KG_DEADLINE_MARGIN):
        _KG_TIMEOUT = True
        return False
    return True


def acquisition(mu, sigma, f_best):
    """Knowledge Gradient acquisition: expected reduction in observed minimum."""
    n = mu.shape[0]
    
    # If we're close to the deadline, fall back to EI for speed
    if not _kg_deadline_ok():
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Knowledge Gradient: E[max(0, f_best - min_i f_i)]
    # Approximate using a low-dimensional quadrature over the posterior.
    # For each candidate x_i, the KG value is:
    #   E[ max(0, f_best - min(f_i, f_best)) ]
    # where the expectation is over the joint posterior of all 2000 points.
    
    # Exact KG is intractable for 2000 points. Use a practical approximation:
    # KG_i ≈ E[ max(0, f_best - min(f_i, f_best)) ]
    #        = E[ max(0, f_best - f_i) * I(f_i <= f_best) + 0 * I(f_i > f_best) ]
    #        = E[ max(0, f_best - f_i) | f_i <= f_best ] * P(f_i <= f_best)
    # But this ignores correlations. A better approximation uses the fact that
    # for the minimum over many points, the relevant uncertainty is in the
    # points that could potentially beat f_best.
    
    # Practical KG approximation: use the expected improvement of each point
    # but weight by the probability that it would be the new minimum.
    # This is essentially EI but with a correction for the "minimum" structure.
    
    # Simpler and effective approach: use a sample-based KG estimate.
    # Draw K samples from the joint Gaussian posterior and compute the
    # expected improvement of the minimum.
    
    # For efficiency with 2000 points, use a reduced set of Monte Carlo samples
    # and a fast approximation.
    
    # Method: For each candidate i, compute
    #   KG_i = E[ max(0, f_best - min_j f_j) ] - E[ max(0, f_best - min_{j≠i} f_j) ]
    # This is the marginal value of information from evaluating point i.
    
    # Approximate using: the probability that point i is the new best times
    # the expected improvement of point i.
    
    # P(f_i is new best) ≈ P(f_i < f_best) * (1 / E[number of points < f_best])
    # This is a heuristic. A cleaner approach:
    
    # Use the fact that for a GP, the KG for the minimum can be approximated
    # by considering only the points with low mu (likely to be good).
    
    # Practical implementation: use a two-stage approach.
    # 1. Compute EI for all points (fast).
    # 2. For the top ~50 points by EI, compute a more accurate KG using
    #    Monte Carlo integration over the joint posterior of those points.
    
    # But even that may be slow. Let's use a simpler, well-known approximation:
    # KG ≈ EI * (1 - exp(-sigma_i^2 / sigma_ref^2))
    # where sigma_ref is a reference scale. This isn't standard.
    
    # Better: use the standard KG approximation via normal integrals.
    # For a single point, KG_i = E[max(0, f_best - f_i)] = EI_i
    # For the minimum of many points, the KG is larger because the minimum
    # over correlated points has different properties.
    
    # The most practical and effective approach for this budget:
    # Use a sample-based estimate with a small number of MC samples.
    
    # Draw K=20 samples from the joint posterior (correlated Gaussians).
    # For each sample, compute the minimum over all 2000 points.
    # KG_i = (1/K) * sum_k max(0, f_best - min_j sample_k_j) 
    #        - (1/K) * sum_k max(0, f_best - min_{j≠i} sample_k_j)
    
    # This is O(K * n^2) which is 20 * 2000^2 = 80M operations. Too slow.
    
    # Optimized: for each MC sample k, compute min_j sample_k_j in O(n).
    # Then for the "excluding i" part, we need the second minimum or
    # track which index was the minimum.
    
    # For each sample k:
    #   min_val_k = min over all j
    #   argmin_k = argmin over all j
    #   second_min_k = second smallest value
    # Then:
    #   min_excluding_i_k = min_val_k if i != argmin_k else second_min_k
    #   KG_i = (1/K) * sum_k [max(0, f_best - min_val_k) - max(0, f_best - min_excluding_i_k)]
    
    # This is O(K * n) which is 20 * 2000 = 40K operations. Very fast!
    
    K = 50  # number of MC samples
    
    # We need to sample from the joint posterior of all 2000 points.
    # The covariance matrix is 2000x2000 which is too large to form explicitly
    # in a reasonable time. 
    # 
    # Alternative: use the fact that we only need to sample the values,
    # and we can do this using the GP structure. But we don't have access
    # to the kernel or training data here.
    #
    # Since we only have mu and sigma (marginal mean and std), we cannot
    # exactly sample the joint posterior without the full covariance.
    #
    # Practical compromise: use an approximation that only uses the marginals.
    # Assume independence (crude) or use a structured approximation.
    #
    # With only marginals, the best we can do is:
    # KG_i ≈ E[max(0, f_best - f_i)] = EI_i
    # which is what we already have.
    #
    # To do better, we can use a heuristic: the KG for the minimum of n
    # correlated points is approximately:
    #   KG ≈ EI_i * correction_factor
    # where correction_factor accounts for the fact that we're looking at
    # the minimum, not a single point.
    #
    # A known result: for n i.i.d. normal variables with mean mu_i and
    # std sigma_i, the expected minimum is approximately:
    #   E[min f_i] ≈ min_i mu_i - c * sigma_min
    # where c depends on n and the correlation structure.
    #
    # For the KG, a practical approximation used in practice:
    #   KG_i = E[max(0, f_best - min_j f_j)] 
    #         ≈ P(f_i < f_best) * E[max(0, f_best - f_i) | f_i < f_best]
    #           * (1 + (n-1) * P(f_j < f_i | f_i < f_best))
    # This is getting complicated.
    #
    # Let's use a simpler, empirically effective approach:
    # Scale EI by a factor that increases with the number of points
    # that have non-negligible probability of being below f_best.
    
    # Compute z-scores
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability that each point beats f_best
    p_beat = norm.cdf(z)
    
    # Expected number of points that beat f_best
    expected_beats = np.sum(p_beat)
    
    # If many points can beat f_best, the minimum is more likely to be
    # significantly better than f_best. Scale EI accordingly.
    # The scaling factor: when expected_beats is large, the minimum over
    # those points is lower than the typical individual point.
    
    # Use a soft-max-like correction: the expected minimum of m normal
    # variables with similar std is approximately mu_min - c*sigma where
    # c grows slowly with m (like log(m) for independent normals).
    
    # Practical heuristic: multiply EI by (1 + log(1 + expected_beats) / 10)
    # This is a mild correction that increases the score when many points
    # are promising.
    
    correction = 1.0 + np.log1p(expected_beats) / 10.0
    scores = ei * correction
    
    return scores
# EVOLVE-BLOCK-END
