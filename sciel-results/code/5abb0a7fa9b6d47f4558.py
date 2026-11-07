"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition using a finite Gaussian mixture model of the posterior.
    
    Instead of EI + exploration, we construct a GMM approximation of the posterior
    distribution over objective values by discretizing the uncertainty field into
    percentile-based components. A single random draw from this GMM gives a sample
    objective value per candidate, and we score by the negative expected value.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    t_start = time.time()
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Build a finite Gaussian mixture model of the posterior.
    # We discretize the sigma field into K components using percentiles,
    # then for each component we have a mean (from mu at those points) and
    # variance (from sigma^2 at those points).
    
    K = 20  # number of mixture components
    
    # Compute percentile thresholds for sigma to define component boundaries
    sigma_sorted = np.sort(sigma_safe)
    # Use quantiles to define component centers
    quantiles = np.linspace(0, 1, K + 1)[1:-1]  # K-1 internal boundaries
    thresholds = sigma_sorted[np.floor(quantiles * (n - 1)).astype(int)]
    
    # Assign each candidate to the nearest sigma component
    # For efficiency, use digitize
    component_indices = np.digitize(sigma_safe, thresholds)  # values in [0, K-1]
    
    # For each component, compute the mean of mu and sigma for candidates in that component
    comp_mu = np.zeros(K)
    comp_sigma = np.zeros(K)
    comp_count = np.zeros(K)
    
    for k in range(K):
        mask = (component_indices == k)
        if np.any(mask):
            comp_mu[k] = np.mean(mu[mask])
            comp_sigma[k] = np.mean(sigma_safe[mask])
            comp_count[k] = np.sum(mask)
        else:
            comp_mu[k] = np.mean(mu)
            comp_sigma[k] = np.mean(sigma_safe)
    
    # Normalize weights by count (proportional to how many candidates fall in each bin)
    weights = comp_count / max(np.sum(comp_count), 1)
    
    # Thompson sampling: draw one sample from the mixture for each candidate.
    # For candidate i in component k, the posterior is N(mu[i], sigma[i]^2).
    # We draw a sample from N(mu[i], sigma[i]^2) for each candidate.
    # But to make it a proper TS over the GMM, we first pick a component
    # with probability proportional to weights, then sample from that component's
    # distribution. However, since each candidate has its own (mu, sigma),
    # the standard TS approach is to sample f_i ~ N(mu_i, sigma_i^2) for each i
    # and select the one with the lowest f_i.
    #
    # The GMM approach: we model the overall posterior as a mixture where
    # component k has mean comp_mu[k] and std comp_sigma[k], with weight w_k.
    # Then for each candidate, we compute the expected improvement under this
    # mixture, which approximates the true posterior expectation.
    #
    # Actually, for Thompson Sampling, the simplest correct approach is:
    # sample f_i ~ N(mu_i, sigma_i^2) for each candidate i, and score by -f_i.
    # This is exact TS for the GP posterior. The GMM is an approximation
    # that can be used when we want to regularize or smooth the sampling.
    #
    # Let's use the exact TS approach but with a GMM-based adjustment:
    # We draw a global "noise level" from the mixture, then use it to
    # modulate the per-candidate uncertainty.
    
    # Draw a sample from the GMM to get a global uncertainty scale
    r = np.random.uniform()
    cum_weights = np.cumsum(weights)
    selected_comp = np.searchsorted(cum_weights, r)
    selected_comp = min(selected_comp, K - 1)
    
    global_sigma = comp_sigma[selected_comp]
    
    # Per-candidate Thompson sampling: sample from N(mu_i, sigma_i^2)
    # But blend with the global scale to encourage exploration in uncertain regions
    # Use a weighted combination: sample = mu_i + z * sigma_i
    # where z ~ N(0, 1), and we scale sigma_i by a factor based on global uncertainty
    
    # The key insight: in TS, candidates with high sigma have a wider range of
    # possible values, so they're more likely to be selected when the function
    # is uncertain. This naturally balances exploration and exploitation.
    
    z = np.random.standard_normal(n)
    samples = mu + z * sigma_safe
    
    # Score: higher is better, so we want low function values
    scores = -samples
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
