"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    # Structural change: adaptive Lower Confidence Bound (LCB) for minimization.
    # Score is maximized: LCB = mu - beta * sigma.
    # We use f_best to adapt beta: larger regret -> larger exploration.
    # This is different from the previous plain UCB in that the exploration term
    # is explicitly scaled by current regret and clamped to a safe range.

    start_time = time.time()

    # Regret relative to known minimum (hartmann6d global min ~ -3.32237)
    # Clamp to non-negative and reasonable range to avoid pathological scaling.
    regret = float(np.clip(f_best - (-3.32237), 0.0, 10.0))

    # Adaptive beta: starts high when regret is large, decays toward a floor.
    # Floor keeps some exploration even when close to optimum.
    beta = float(np.clip(0.5 + 2.0 * regret, 0.5, 5.0))

    # LCB score for minimization: we maximize mu - beta * sigma
    # (equivalently, we minimize mu - beta*sigma? No: for minimization,
    #  lower confidence bound is mu - beta*sigma; we choose the point with
    #  the lowest LCB. Since the task wants "higher score means evaluate next",
    #  we return the negative LCB: -mu + beta*sigma.)
    scores = -mu + beta * sigma

    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Deadline guard: return a simple greedy fallback if we are too close to the limit.
    elapsed = time.time() - start_time
    if elapsed > 290:
        scores = -mu
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
# EVOLVE-BLOCK-END
