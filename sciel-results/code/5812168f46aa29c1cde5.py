"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Predictive Entropy Search (PES) variant acquisition function.
    
    Scores each candidate by the expected reduction in the posterior entropy
    of the global minimum. We approximate this via mutual information between
    the function value at the candidate and the identity of the current best point.
    
    For a practical approximation:
    - For each candidate x_i, we consider the distribution of f(x_i) ~ N(mu_i, sigma_i^2).
    - The 'value of information' is high when observing f(x_i) is likely to change
      which point is the best, i.e., when f(x_i) could be lower than f_best.
    - We use an approximation: the expected KL divergence between the posterior
      over the minimum before and after observing f(x_i).
    
    A common efficient approximation for PES-like scoring is:
    score_i = E[log(1 + max(0, f_best - f(x_i)) / (sigma_i + eps))]
    or more directly, using the probability that x_i improves on f_best weighted
    by the magnitude of improvement relative to uncertainty.
    
    However, a well-known practical variant is to use:
    score_i = (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i)  [this is EI]
    
    To differentiate from EI and target global structure, we use a variant that
    emphasizes the *probability* of being the new global minimum more strongly,
    combined with a term that rewards reducing the *range* of plausible minima.
    
    A cleaner PES approximation: The mutual information I(f(x_i); m*) where m* is
    the global minimum. A surrogate for this is the expected change in the
    log-sum-exp of the posterior probabilities of each point being the minimum.
    
    For efficiency with 2000 candidates, we use an approximation:
    score_i = E_{f(x_i)~N(mu_i, sigma_i^2)}[ log(1 + exp((f_best - f(x_i))/tau)) ]
    where tau is a temperature parameter. This is a soft-max approximation.
    
    Alternatively, a simpler and effective variant: use the probability that
    f(x_i) < f_best, weighted by the expected improvement magnitude, but with
    a different weighting to favor points that are likely to be *new* minima
    in unexplored regions.
    
    Let's use a variant of PES that scores based on:
    score_i = P(f(x_i) < f_best) * (1 + (f_best - mu_i) / (sigma_i + eps))
    
    This emphasizes both the likelihood of improvement and the expected magnitude,
    but with a different balance than EI.
    
    Actually, let's use a more principled approximation: the expected reduction
    in the entropy of the distribution over the minimum. A practical proxy is:
    score_i = E[ max(0, f_best - f(x_i)) ] / (sigma_i + eps)
    which is (f_best - mu_i) * Phi(z_i) + sigma_i * phi(z_i) divided by sigma_i,
    i.e., (f_best - mu_i)/sigma_i * Phi(z_i) + phi(z_i).
    
    This normalizes by the local uncertainty, rewarding points where the expected
    improvement is large relative to the local noise. This can help explore
    regions with lower uncertainty that still have a chance of improvement.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # Standard Expected Improvement
    # This balances exploration (via sigma) and exploitation (via mu)
    # without the aggressive normalization that caused exploitation bias
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Ensure finite values
    score = np.where(np.isfinite(ei), ei, 0.0)
    return score
# EVOLVE-BLOCK-END
