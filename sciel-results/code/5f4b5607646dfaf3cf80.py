"""Cost-aware acquisition for Bayesian optimization on 2D Rastrigin."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills after 330s; keep a safety margin.
_DEADLINE = 325.0  # seconds
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    Uses a cost-aware EI with a rank-based Pareto front on (EI, 1/cost).
    """
    # Safety margin: if we're close to the deadline, return a simple fast score.
    if _elapsed() > _DEADLINE - 5.0:
        # Fallback: simple cost-aware EI, no Pareto computation.
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / (cost + eps)
        return np.where(np.isfinite(score), score, 0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # Expected improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware base score: EI per unit cost
    cost_safe = np.maximum(cost, eps)
    base_score = ei / cost_safe

    # Exploration bonus: reward high uncertainty at low cost
    explore = sigma_safe / cost_safe

    # Combine: weighted sum with a tunable exploration weight
    w_explore = 0.3
    combined = base_score + w_explore * explore

    # Ensure finite
    combined = np.where(np.isfinite(combined), combined, 0.0)

    # Rank-based Pareto front on (ei, 1/cost)
    # Higher ei and higher 1/cost (lower cost) is better.
    inv_cost = 1.0 / cost_safe
    inv_cost = np.where(np.isfinite(inv_cost), inv_cost, 0.0)

    n = len(mu)
    # Compute non-dominated ranks via a simple O(n^2) dominance check
    # For 2000 points, this is 4M comparisons, which is fast in NumPy vectorized form.
    # A point i dominates j if ei[i] >= ei[j] and inv_cost[i] >= inv_cost[j]
    # and at least one is strict.

    # Vectorized dominance: for each i, count how many j are dominated by i.
    # Actually we want rank: number of points that dominate i (Pareto rank level).
    # Simpler: assign rank = number of points that are strictly better in at least one
    # objective and not worse in the other.

    # Let's compute for each point i: is it dominated by any point?
    # Use a loop over points but vectorize the comparison against all others.
    # To keep it fast, we do a vectorized approach:
    # For each i, dominated_by[i] = exists j != i such that
    #   ei[j] >= ei[i] and inv_cost[j] >= inv_cost[i] and (ei[j] > ei[i] or inv_cost[j] > inv_cost[i])

    # Build matrices (n x n)
    # This uses 4M floats ~ 32 MB, acceptable.
    ei_col = ei[:, None]      # n x 1
    ei_row = ei[None, :]      # 1 x n
    ic_col = inv_cost[:, None]
    ic_row = inv_cost[None, :]

    # j dominates i if:
    #   ei_row >= ei_col AND ic_row >= ic_col AND (ei_row > ei_col OR ic_row > ic_col)
    # Note: ei_row is (1, n), ei_col is (n, 1), so broadcasting gives (n, n) where
    # element [i, j] compares point j (row) vs point i (col).

    dom_mask = (ei_row >= ei_col) & (ic_row >= ic_col) & ((ei_row > ei_col) | (ic_row > ic_col))
    # dom_mask[i, j] = True if point j dominates point i.
    # Exclude self-dominance (should be False anyway due to strict inequality).
    np.fill_diagonal(dom_mask, False)

    # Number of dominators for each point
    n_dominators = dom_mask.sum(axis=1)  # for each i, how many j dominate it

    # Pareto rank: points with 0 dominators are rank 1 (front), etc.
    # We assign rank = n_dominators + 1 (1-indexed front level approximation).
    # Then score contribution from rank: lower rank (better) gets higher score.
    # Use rank-based score: score_rank = (n - n_dominators) / n  -> in [0, 1]
    rank_score = (n - n_dominators) / n

    # Final score: combine base+explore with rank score
    # Normalize rank_score to have similar scale as combined.
    # combined can be large; rank_score in [0,1].
    # Use: final = combined * (0.5 + 0.5 * rank_score)  -> multiplicative boost for front points.
    # Or additive: final = combined + alpha * rank_score.
    # Let's use a blend: final = 0.7 * combined + 0.3 * rank_score * (combined.max() + 1e-9)
    # Actually, to keep scale consistent, normalize combined to [0,1] first.
    c_min = combined.min()
    c_max = combined.max()
    if c_max - c_min > eps:
        combined_norm = (combined - c_min) / (c_max - c_min)
    else:
        combined_norm = np.zeros_like(combined)

    # Final score: weighted combination
    alpha = 0.5
    final_score = alpha * combined_norm + (1 - alpha) * rank_score

    # Ensure finite and positive
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    final_score = np.maximum(final_score, 0.0)

    return final_score
# EVOLVE-BLOCK-END
