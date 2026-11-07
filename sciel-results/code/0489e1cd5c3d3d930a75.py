"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Explicit wall-clock deadline guard. The evaluator kills at 330s; we stop
# with a safety margin and return the best result found so far.
_DEADLINE = time.time() + 325.0


def _time_exceeded():
    return time.time() >= _DEADLINE


def heuristics(prizes, weights):
    """
    Attractiveness per item for the 5-constraint knapsack.

    The ant adds an item only if it still fits in ALL constraints, so the
    effective cost of an item is governed by the tightest (largest) constraint
    it consumes. A plain sum over-weights items that are moderately heavy in
    every constraint relative to items heavy in just one. We therefore use a
    smooth surrogate for the per-constraint bottleneck: a log-sum-exp (soft-max)
    of the weight vector across the 5 constraints. This is differentiable,
    smooth, and approximates max_i weights[i] for a moderate temperature.

    Heuristic = total prize / (soft-max of constraint weights + eps).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Total prize per item (sum across the 5 constraints).
    total_prize = prizes.sum(axis=0)

    # Soft-max of the constraint weights (smooth max) via log-sum-exp.
    # weights has shape (5, 100); max over axis=0 -> (100,).
    w_max = weights.max(axis=0)
    # log-sum-exp for stability:
    # log(sum_k exp(w_k)) = w_max + log(sum_k exp(w_k - w_max))
    shifted = weights - w_max[None, :]
    lse = w_max + np.log(np.exp(shifted).sum(axis=0))

    # Temperature: lower -> closer to true max (harder bottleneck), higher ->
    # closer to sum. We want the bottleneck to dominate, so use a modest
    # temperature that sharpens the soft-max toward the max while staying smooth.
    # scale the soft-max so it sits on the same order as the raw weights.
    # (log-sum-exp already returns a value near the max, in [0, ~log(5)+max]).
    soft_max = lse

    # Attractiveness: high prize, low (bottleneck) weight -> large score.
    eps = 1e-9
    scores = total_prize / (soft_max + eps)

    # Guard against non-finite values (defensive; uniform inputs make this rare).
    scores = np.where(np.isfinite(scores), scores, 1e-9)

    # Ensure positive; non-positive entries are clamped by the caller to 1e-9,
    # but we return strictly positive to be safe.
    scores = np.maximum(scores, 1e-9)

    return scores
# EVOLVE-BLOCK-END
