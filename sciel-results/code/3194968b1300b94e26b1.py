"""Direct acquisition for sphere 1d: Expected Hypar (EHypR).

The task fully specifies the objective (sphere 1d = x^2 on [-5,5], minimum 0
at x=0) and the GP geometry (Matern-5/2, length scale 0.2 on a coordinate
scaled to [0,1], standardized targets). We use Expected Hypar, which
maximizes E[max(f_best, f(x))/f(x)] over the GP posterior f(x) ~ N(mu, sigma^2).
This directly targets simple regret by balancing the probability of
improvement with the magnitude of the potential gain, rather than relying on
fixed or heuristic uncertainty bounds like UCB.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition for minimizing sphere 1d on [-5, 5]^1 (min value 0 at x=0).

    mu, sigma: shape (2000,), posterior mean / std in objective units.
    f_best: lowest observed value.
    Returns 2000 finite scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)

    # Clamp f_best to a small positive value to avoid division by zero
    # and to handle the case where f_best is very close to 0 (optimal).
    f_best = max(float(f_best), 1e-10)

    # Expected Hypar: E[max(f_best, f(x))/f(x)] where f(x) ~ N(mu, sigma^2)
    # = E[1] * P(f(x) > f_best) + E[f_best/f(x)] * P(f(x) <= f_best)
    # = P(f(x) > f_best) + f_best * E[1/f(x); f(x) <= f_best]
    #
    # For f(x) ~ N(mu, sigma^2), the conditional expectation of 1/f(x)
    # given f(x) <= t can be computed via the non-central t distribution
    # or via numerical integration. A practical approximation uses:
    #
    # E[1/f(x); f(x) <= t] = (1/sigma) * phi((t-mu)/sigma) / Phi((t-mu)/sigma)
    #   where phi is the standard normal pdf and Phi is the cdf,
    #   evaluated at z = (t-mu)/sigma, but this is the conditional mean of
    #   the reciprocal, which is tricky.
    #
    # A more robust approach: use the fact that for f(x) ~ N(mu, sigma^2),
    # E[1/f(x)] is undefined in general, but the truncated expectation
    # E[1/f(x); f(x) <= t] can be approximated by:
    #   (1/sigma) * phi(z) / Phi(z)  where z = (t - mu)/sigma
    #   This is the mean of 1/f(x) conditional on f(x) <= t, times Phi(z).
    #
    # Actually, let's use a cleaner derivation:
    # E[max(f_best, f)/f] = P(f > f_best) + f_best * E[1/f | f <= f_best] * P(f <= f_best)
    #
    # Let z = (f_best - mu) / sigma
    # P(f > f_best) = 1 - Phi(z)
    # P(f <= f_best) = Phi(z)
    #
    # For the conditional expectation E[1/f | f <= f_best]:
    # This is a well-known quantity. For f ~ N(mu, sigma^2),
    # E[1/f | f <= t] = (1/sigma) * phi(z) / Phi(z) * (1 / (1 - mu*z/sigma^2))
    # This gets complicated. Let's use a simpler, robust approximation:
    #
    # E[1/f | f <= f_best] ≈ 1 / (mu - sigma * phi(z)/Phi(z))
    #   where the denominator is the conditional mean E[f | f <= f_best].
    #
    # This is a reasonable approximation when the distribution is not too
    # spread out relative to its mean.

    z = (f_best - mu) / sigma
    Phi = norm.cdf(z)
    phi = norm.pdf(z)

    # Avoid division by zero in Phi
    Phi_safe = np.maximum(Phi, 1e-15)
    phi_safe = np.maximum(phi, 1e-15)

    # P(f > f_best)
    p_improve = 1.0 - Phi

    # Conditional mean E[f | f <= f_best] = mu - sigma * phi(z) / Phi(z)
    cond_mean = mu - sigma * phi_safe / Phi_safe

    # Approximate E[1/f | f <= f_best] ≈ 1 / cond_mean
    # Clamp cond_mean to avoid division by zero or negative values
    cond_mean_safe = np.where(cond_mean > 1e-10, cond_mean, 1e-10)
    inv_cond_mean = 1.0 / cond_mean_safe

    # EHypR = P(f > f_best) + f_best * E[1/f | f <= f_best] * P(f <= f_best)
    score = p_improve + f_best * inv_cond_mean * Phi_safe

    # Enforce finiteness
    score = np.where(np.isfinite(score), score, -1e12)

    return score
# EVOLVE-BLOCK-END
