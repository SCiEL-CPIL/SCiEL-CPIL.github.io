"""Cost-aware acquisition for Bayesian optimization (levy 2d).

Structural change vs. the seed: instead of a smooth expected-improvement-per-cost
trade-off (which always leaks some budget toward high-cost regions), this uses a
HARD feasibility gate. We compute a predicted cost for each candidate and only
award Expected Improvement to candidates whose predicted cost is below a
threshold. The threshold is set so the *average cost of the improving (low-cost)
candidates* equals the current mean cost per unit improvement. As the GP gets
confident in the low-cost corner, the improvement per unit cost shrinks, the
threshold drops, the feasible set tightens to the cheapest points, and once the
predicted improvement there is exhausted the search is forced to explore.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Hard wall-clock guard: the evaluator kills the process at 330 s. We keep a
# safety margin so we always return a finite, valid vector.
_DEADLINE = time.time() + 320.0
_DEADLINE_SET = True


def _expected_improvement(mu, sigma, f_best):
    """Standard expected improvement (higher = more promising)."""
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.shape[0]

    # Ensure finite inputs; fall back gracefully.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))
            and np.all(np.isfinite(cost))):
        return np.zeros(n)

    # Guard against pathological cost values.
    cost = np.maximum(cost, 1e-9)

    # Expected improvement for every candidate.
    ei = _expected_improvement(mu, sigma, f_best)
    ei = np.maximum(ei, 0.0)

    # Improvement per unit cost across all candidates.
    ipc = ei / cost
    ipc_max = float(np.max(ipc)) if ipc.size else 0.0

    if ipc_max <= 1e-12:
        # Nothing to improve on; fall back to a cheap exploration signal so we
        # still return a finite, finite-ordered vector (prefer low cost).
        return -cost

    # Threshold: average cost of the improving (EI>0) candidates equals the
    # current best improvement-per-cost ratio. This is the multiplicative
    # feasibility constraint described in the directive.
    improving = ei > 1e-12
    if np.any(improving):
        avg_cost_improving = float(np.mean(cost[improving]))
    else:
        avg_cost_improving = float(np.mean(cost))

    threshold = avg_cost_improving / ipc_max

    # Hard gate: only candidates with predicted cost below the threshold score.
    feasible = cost < threshold

    if not np.any(feasible):
        # If the gate is too strict (no feasible point), open it slightly so we
        # always have at least one candidate to pick, preserving validity.
        # Relax to the cheapest candidate's cost.
        threshold = float(np.min(cost)) * 1.01
        feasible = cost < threshold
        if not np.any(feasible):
            feasible = cost <= float(np.min(cost)) * 1.01 + 1e-9

    scores = np.zeros(n)
    scores[feasible] = ei[feasible]

    # Among the feasible set, also reward cheapness mildly so the search stays
    # in the low-cost corner (the gate already enforces this, but a small
    # secondary term breaks ties toward cheaper points).
    scores = scores - 1e-3 * cost

    # Final safety: guarantee finite output.
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
