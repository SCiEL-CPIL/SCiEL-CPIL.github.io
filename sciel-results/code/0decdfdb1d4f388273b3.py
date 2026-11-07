"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = _start_time + 320.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're close to the limit, return a fast fallback
    if _elapsed() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Thompson sampling score: draw from the posterior and add a small
    # exploration bonus proportional to uncertainty.
    # For minimization, lower sampled values are better.
    # We invert: score = -sample + beta * sigma
    # Use multiple samples for a more stable estimate.
    n_samples = 20
    rng = np.random.default_rng()
    # Draw samples from N(mu, sigma) for each candidate
    # Shape: (n_samples, 2000)
    samples = rng.standard_normal((n_samples, len(mu))) * sigma_safe[None, :] + mu[None, :]

    # For each sample, the best (minimum) value across samples is the
    # "expected best" under Thompson sampling.
    # Actually, Thompson sampling: pick the sample argmin, then pick the
    # candidate that minimizes that sample. But we want a score per candidate.
    # Better approach: score_i = -mean of posterior draws + beta * sigma_i
    # This is equivalent to a UCB-like score.

    # Alternative structural approach: Use a "rank-based" acquisition.
    # Score by rank of (mu - alpha * sigma) to be robust to scale.
    # Combined with cost awareness.

    # Let's try: score = -mu + beta * sigma - gamma * cost_penalty
    # where cost_penalty is a smooth function of cost.
    # This is a "noisy" UCB that directly accounts for cost.

    # Structural change: Use a "knowledge gradient"-inspired approximation.
    # For minimization, the probability that candidate i beats f_best is:
    # P(y_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    # The expected improvement is EI_i = (f_best - mu_i)*Phi(z_i) + sigma_i*phi(z_i)
    #
    # New approach: "Cost-normalized EI with adaptive beta"
    # Instead of fixed weights, use a log-sum-exp style combination.

    # Compute EI
    ei_vals = np.maximum(ei, 0.0)

    # Compute a "value of information" that accounts for cost
    # The effective reward of evaluating at cost c is EI / c, but
    # we want to balance this against the variance of the estimate.
    #
    # Use a "soft" cost discount: score = EI / (1 + c)^alpha
    # This is smoother than EI / c^alpha for large c.

    # Adaptive alpha: reduce cost sensitivity when EI is high relative to sigma
    # (i.e., we're confident in improvement) vs when EI is low (exploration).
    # ratio = EI / sigma
    ratio = ei_vals / sigma_safe
    # When ratio is high (confident improvement), use lower alpha (less cost penalty)
    # When ratio is low (uncertain), use higher alpha (more cost penalty)
    # alpha = alpha_base - alpha_range * tanh(ratio)
    alpha_base = 1.0
    alpha_range = 0.5
    alpha = np.clip(alpha_base - alpha_range * np.tanh(ratio), 0.3, 1.5)

    cost_factor = (1.0 + cost_safe) ** alpha
    score = ei_vals / cost_factor

    # Add a small exploration bonus for very uncertain regions
    # where EI is near zero but sigma is large.
    # exploration_bonus = eps * sigma / cost
    eps = 0.05
    exploration_bonus = eps * sigma_safe / cost_safe
    score = score + exploration_bonus

    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
