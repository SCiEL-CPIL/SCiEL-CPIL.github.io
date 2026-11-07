# EVOLVE-BLOCK-START
import time
import math
import numpy as np

# Deadline guard: the evaluator kills at 330 s wall-clock.
# We keep a safety margin so we always return in time.
_DEADLINE = 330.0
_SAFETY_MARGIN = 3.0  # seconds we want to finish before the hard kill
_T0 = time.time()

# Precompute a small helper for the normal CDF (stable)
def _norm_cdf(z):
    return 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))

def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware blended acquisition for rosenbrock-2d.

    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we are close to the wall-clock limit,
    # return a fast, deterministic score so we never time out.
    elapsed = time.time() - _T0
    if elapsed > (_DEADLINE - _SAFETY_MARGIN):
        # Fast fallback: plain cost-aware EI, computed with cheap ops.
        s = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / s
        ei = (f_best - mu) * _norm_cdf(z) + s * _norm_pdf(z)
        c = np.maximum(cost, 1e-6)
        out = ei / c
        return np.nan_to_num(out, nan=0.0, posinf=1e300, neginf=-1e300)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # --- Term 1: cost-aware expected improvement (EI / cost) ---
    z = (f_best - mu) / sigma
    cdf = _norm_cdf(z)
    pdf = _norm_pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei_cost = ei / cost

    # --- Term 2: "reachability" — how good could this point be
    #     after paying its cost?  Reward points whose predicted
    #     minimum (mu - 2*sigma) is low AND whose cost is small.
    #     We model the best value we could observe here as
    #     max(mu - 2*sigma, 0) (rosenbrock min is 0) and
    #     subtract the cost as a penalty in objective-like units.
    #     Since cost is in [1, 2] (1 + mean(x in [0,1])), we
    #     scale it to objective units by a factor that reflects
    #     the typical curvature of rosenbrock near the minimum.
    #     A cheap point near a low-mu region is strongly rewarded.
    predicted_min = np.maximum(mu - 2.0 * sigma, 0.0)
    # Penalise cost in objective units: cost is ~[1,2];
    # near the minimum, rosenbrock values are small, so we
    # scale cost by 0.1 to make it comparable to objective values.
    reachability = -(predicted_min + 0.1 * (cost - 1.0))
    # Normalise reachability to a similar scale as ei_cost.
    # ei_cost is typically small (EI is small when f_best is small).
    # We use a soft scaling: tanh of reachability scaled by 1/f_best-ish.
    scale = max(abs(f_best), 1e-6)
    reachability_scaled = np.tanh(reachability / scale) * scale

    # --- Term 3: exploration bonus, decays as budget is used ---
    # Remaining budget: total budget 30, initial cost ~5*1.5=7.5 (rough),
    # so remaining ~ 30 - cost_total.  We approximate progress by
    # f_best (lower = more progress) and by the current cost level.
    # Exploration weight: high early (f_best large), low late (f_best small).
    # Use a logistic decay in f_best.
    exploration_weight = 1.0 / (1.0 + np.exp((f_best - 0.5) / 0.1))
    # Exploration score: high where sigma is large (uncertain) and
    # the point is not obviously bad (mu not too high).
    exploration = exploration_weight * sigma / (1.0 + np.maximum(mu, 0.0))

    # --- Combine ---
    # Weights: EI/cost is the primary signal; reachability is secondary;
    # exploration is a small bonus that fades as we converge.
    w_ei = 1.0
    w_reach = 0.5
    w_exp = 0.2

    score = w_ei * ei_cost + w_reach * reachability_scaled + w_exp * exploration

    # Ensure finiteness
    score = np.nan_to_num(score, nan=0.0, posinf=1e300, neginf=-1e300)
    return score
# EVOLVE-BLOCK-END
