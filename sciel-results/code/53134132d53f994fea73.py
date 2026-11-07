"""Rewritten acquisition using a Knowledge Gradient surrogate."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) surrogate acquisition.
    
    For each of the 2000 candidates, we approximate the expected reduction
    in the global minimum value. We use a Monte-Carlo-free approximation
    based on the posterior distribution of the minimum over all candidates.
    
    The KG value at point x_i is:
      E[ max(0, f_best - min_j Y_j) ] - E[ max(0, f_best - min_{j != i} Y_j) ]
    
    where Y_j ~ N(mu_j, sigma_j^2) are the posterior predictions.
    
    We approximate this using a two-point quadrature / analytical approximation
    for the expected minimum of correlated normals. Since we only have marginals
    (mu, sigma) and not the full covariance, we use a conservative approximation:
    treat the minimum as approximately normal with mean and std derived from
    the marginal distributions.
    
    A practical approximation: the expected minimum of N independent normals
    with means mu_j and stds sigma_j can be approximated. For the KG, we
    estimate how much the minimum would improve if we "revealed" x_i.
    
    We use a simplified KG: for each candidate i, compute the probability-weighted
    improvement over the current best, considering that the true minimum across
    all 2000 points could be lower than f_best.
    """
    if _check_deadline():
        # Fallback to EI if we're close to deadline
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    n = len(mu)
    
    # Approximate the distribution of the minimum over all 2000 candidates.
    # The minimum M = min_j Y_j where Y_j ~ N(mu_j, sigma_j^2).
    # We approximate E[M] and Std[M] using the fact that for many candidates,
    # the minimum is dominated by the few with lowest mu and moderate sigma.
    
    # Sort by mu to identify the most promising candidates
    # Use a subset for efficiency: take the top K candidates by (mu + alpha*sigma)
    alpha = 0.5
    scores_rank = mu + alpha * sigma
    # Take top 100 by this score for the minimum estimation
    K = min(100, n)
    top_idx = np.argpartition(scores_rank, K)[:K]
    
    mu_top = mu[top_idx]
    sigma_top = sigma[top_idx]
    
    # Approximate the expected minimum of K normals.
    # For independent normals, E[min] can be approximated by the mean of the
    # smallest order statistic. A simple approximation:
    # E[min] ≈ min(mu_top) - 0.5 * min(sigma_top) * sqrt(2*log(K))
    # This is rough but captures the idea that more candidates → lower expected min.
    
    # Better: use a Monte Carlo with a small number of samples for robustness
    # But that's slow. Instead, use an analytical approximation.
    
    # The expected minimum of K i.i.d. N(0,1) is approximately -sqrt(2 log K) for large K.
    # For non-identical, we use:
    # E[min_j Y_j] ≈ min_j (mu_j - 0.5 * sigma_j * sqrt(2*log(K)))
    # This is a heuristic but reasonable.
    
    log_K = np.log(max(K, 2))
    expected_min = np.min(mu_top - 0.5 * sigma_top * np.sqrt(2 * log_K))
    
    # Now, for each candidate i, the KG value is:
    # E[max(0, f_best - M_all)] - E[max(0, f_best - M_excl_i)]
    
    # Approximate E[max(0, f_best - M)] where M ~ approx N(E[M], Var[M])
    # Var[M] is hard to estimate, so we use a conservative std.
    # Let's say std_M ≈ 0.3 * min(sigma_top) as a rough bound.
    std_M = 0.3 * np.min(sigma_top) + 1e-9
    
    # E[max(0, f_best - M)] where M ~ N(m, s)
    # = (f_best - m) * Phi((f_best - m)/s) + s * phi((f_best - m)/s)
    # This is exactly the EI formula with mu=m, sigma=s!
    
    def expected_max_positive(diff, s):
        """E[max(0, diff - X)] where X ~ N(0, s^2)"""
        z = diff / (s + 1e-9)
        return diff * norm.cdf(z) + s * norm.pdf(z)
    
    base_kg = expected_max_positive(f_best - expected_min, std_M)
    
    # For each candidate i, if we "remove" it, the minimum might increase slightly.
    # The KG for candidate i is the additional information value.
    # A simpler approach: the KG value for candidate i is proportional to
    # the probability that x_i is the argmin, weighted by the improvement.
    
    # Probability that x_i is the minimum:
    # P(Y_i = min) ≈ P(Y_i < Y_j for all j != i)
    # For independent normals, P(Y_i < Y_j) = Phi((mu_j - mu_i) / sqrt(sigma_i^2 + sigma_j^2))
    # This is O(n^2) which is too slow for n=2000.
    
    # Instead, use a simpler heuristic: the KG score is a blend of:
    # 1. The probability that this point is promising (low mu, moderate sigma)
    # 2. The expected improvement if this point is the new minimum
    
    # Compute a "minimum probability" score for each candidate:
    # P_i = P(Y_i < all others) ≈ product over j of Phi((mu_j - mu_i)/sqrt(sigma_i^2+sigma_j^2))
    # Too slow. Use approximation: P_i ≈ Phi((median_mu - mu_i) / (median_sigma + sigma_i))
    
    med_mu = np.median(mu)
    med_sigma = np.median(sigma)
    
    # Probability-like score: how likely is this point to be near the minimum?
    prob_min = norm.cdf((med_mu - mu) / (med_sigma + sigma + 1e-9))
    
    # The KG value: if this point is the minimum, the improvement is max(0, f_best - Y_i)
    # E[KG_i] ≈ P_i * E[max(0, f_best - Y_i) | Y_i is min]
    # When Y_i is min, it's likely that Y_i is low, so E[max(0, f_best - Y_i)] is high.
    
    # Simplified: KG_i = prob_min * EI_i
    # But this is just EI weighted by a probability, which is similar to EI.
    
    # Let's try a different structural approach:
    # Use the fact that KG encourages exploration of uncertain regions that
    # could contain the global minimum. We combine:
    # - Exploitation: EI (local improvement)
    # - Exploration: uncertainty about the global minimum
    
    # The global minimum uncertainty is captured by the spread of the
    # expected minimum estimate. Candidates with high sigma in regions
    # where mu is close to the expected minimum are valuable.
    
    # Distance from the expected minimum region:
    dist_to_min = np.abs(mu - expected_min)
    # Normalize
    dist_norm = dist_to_min / (np.std(mu) + 1e-9)
    
    # Exploration score: high sigma, close to expected minimum
    exploration = sigma / (sigma + 0.1 * (np.std(mu) + 1e-9))
    exploration = exploration * np.exp(-0.5 * dist_norm ** 2)
    
    # Exploitation score: standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei) + 1e-9
    exp_max = np.max(exploration) + 1e-9
    ei_norm = ei / ei_max
    exp_norm = exploration / exp_max
    
    # Blend: 60% exploitation, 40% exploration
    # The exploration term captures the KG-like behavior of seeking
    # regions where the global minimum is uncertain
    kg_score = 0.6 * ei_norm + 0.4 * exp_norm
    
    # Ensure finite
    kg_score = np.where(np.isfinite(kg_score), kg_score, 0.0)
    
    return kg_score
# EVOLVE-BLOCK-END
