"""Acquisition for Bayesian optimization of Ackley 2D; Thompson Sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Explicit wall-clock deadline guard (evaluator kills at 330 s; keep margin).
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses Thompson Sampling: samples from the posterior N(mu, sigma^2) and 
    returns the negative sampled value. Maximizing this score is equivalent 
    to minimizing the sampled function value, which naturally balances 
    exploitation (low mu) and exploration (high sigma) by randomly 
    selecting the best candidate from a plausible realization of the 
    objective function.
    """
    # --- Time guard: if we are too close to the deadline, fall back to a
    # deterministic pure-exploitation score so the loop always gets a valid
    # finite vector back quickly.
    if time.time() > _DEADLINE:
        return -np.asarray(mu, dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Robustify sigma: clip tiny values and drop non-finite entries.
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.clip(sigma, 0.0, None)

    # Thompson Sampling: Draw a sample from the posterior for each candidate.
    # We use a fixed seed derived from the current best value to ensure 
    # reproducibility within a specific state, but the randomness is 
    # intrinsic to the method. However, standard Thompson sampling usually 
    # uses fresh random numbers each call. Since the evaluator calls this 
    # once per step, using np.random is appropriate. 
    # To avoid global state issues or seeding conflicts in the evaluator, 
    # we just draw directly.
    
    # Ensure sigma is strictly positive to avoid issues if 0, though 
    # standard normal * 0 is 0 which is fine.
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    
    # We want to minimize the function, so we maximize the negative sample.
    score = -samples

    # Safety: replace any non-finite values with a conservative fallback so
    # the returned vector is always finite.
    if not np.isfinite(score).all():
        score = np.where(np.isfinite(score), score, -np.where(np.isfinite(mu), mu, 0.0))
        if not np.isfinite(score).all():
            score = np.zeros_like(mu)

    return score
# EVOLVE-BLOCK-END
