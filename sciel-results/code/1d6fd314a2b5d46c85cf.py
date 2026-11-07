"""Rewritten acquisition: Thompson Sampling score normalized by cost, with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track elapsed time and bail out with a safe fallback if we're running low.
_DEADLINE = 330.0  # seconds
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.perf_counter() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.perf_counter()

    # Safety: if we're too close to the deadline, return a simple exploitation score
    if _remaining() < 2.0:
        # Fallback: pure exploitation (lower mu is better), normalized by cost
        return -mu / (cost + 1e-9)

    n = len(mu)

    # Thompson Sampling: sample one draw from the GP posterior at each candidate.
    # score_k = sampled_value_k / cost_k  (lower sampled value = better; we want highest score)
    # Since we want HIGHER score to select next, and lower objective is better,
    # we use: score = -sampled_value / cost
    # This implicitly balances exploration (high sigma -> more variance in sample)
    # and exploitation (low mu -> lower sampled value) while normalizing by cost.

    # Use a fixed seed per call? No -- we want randomness for exploration.
    # But we need to be deterministic enough that results are reproducible across runs.
    # Actually, the task says "seeded instances" for the function f_k, but the acquisition
    # itself can use randomness. However, for stability and to avoid pathological behavior,
    # let's use a single random draw.

    # Draw one sample from N(mu, sigma^2) for each candidate
    # Use standard normal draws
    rng = np.random.default_rng()
    eps = rng.standard_normal(n)
    sampled = mu + sigma * eps

    # Score: we want higher = better. Lower objective is better, so negate.
    # Normalize by cost: lower cost is better, so divide by cost.
    # score = -sampled / cost
    # But we need to handle cost > 0 (it should be, since cost = 1 + mean(x_scaled) >= 1)
    scores = -sampled / (cost + 1e-9)

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores
# EVOLVE-BLOCK-END
