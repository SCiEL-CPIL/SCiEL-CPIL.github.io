"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d).

Uses a Pareto-frontier selection strategy over (predicted improvement,
uncertainty, cost) to balance exploration and cost without arbitrary
weight tuning.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Pareto-frontier filtering.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Compute raw expected improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # We want to maximize EI and sigma (uncertainty), and minimize cost.
    # For Pareto non-dominated sort, we convert to a maximization problem
    # by negating cost.
    # Criteria to maximize: ei, sigma, -cost
    # A point A dominates point B if A is >= B in all criteria and > in at least one.

    n = len(mu)
    dominated = np.zeros(n, dtype=bool)

    # Efficient Pareto frontier identification
    # Use a vectorized approach: for each point, check if any other point dominates it.
    # For n=2000, O(n^2) is 4 million comparisons, which is fast.

    # Criteria matrix: rows are points, columns are [ei, sigma, -cost]
    # All to be maximized.
    criteria = np.column_stack([ei, sigma, -cost])

    # Point i dominates point j if criteria[i] >= criteria[j] in all dims and > in at least one
    # Vectorized: for each pair (i, j), check domination.
    # This is O(n^2) memory which for n=2000 is 4M * 3 floats = ~12 MB, acceptable.

    # But let's be smarter. We can do this in chunks or use a different approach.
    # Actually, for n=2000, O(n^2) is fine.

    # Let's compute domination matrix: dom[i,j] = True if i dominates j
    # To save memory, we can process in blocks.

    # Alternative: for each point j, check if there exists i != j such that i dominates j.
    # We can vectorize over i for each j, but that's still O(n^2).

    # Let's just do it directly with broadcasting in a memory-efficient way.
    # Actually, 2000x2000x3 is 12 million floats = ~96 MB. That might be too much.
    # Let's process in blocks of 100 points.

    block_size = 100
    on_frontier = np.ones(n, dtype=bool)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # For each point in [start, end), check if it's dominated by any point outside the block
        # or by another point in the block.

        # First, check domination by points outside the block
        # Criteria for candidates in block: criteria[start:end]
        # Criteria for all other points: we need to check against all n points.

        # Let's just check against all n points for simplicity, but in a block-wise manner.
        # For each candidate j in the block, we want to know if any i (i != j) dominates j.

        # Vectorized check: for candidates in block (shape [b, 3]) vs all points (shape [n, 3])
        # dom[i, j] = all(criteria[i] >= criteria[j]) and any(criteria[i] > criteria[j])

        # Let's do this in sub-blocks to limit memory.
        sub_block_size = 50
        for s_start in range(start, end, sub_block_size):
            s_end = min(s_start + sub_block_size, end)
            cand = criteria[s_start:s_end]  # shape [c, 3]
            # Compare cand against all n points
            # We need to check if any point i dominates any candidate j in cand.

            # For each candidate j, check all i
            # diff[i, j, k] = criteria[i, k] - cand[j, k]
            # We want: for some i, all_k(diff[i, j, k] >= 0) and any_k(diff[i, j, k] > 0)

            # Let's process i in blocks too.
            for i_start in range(0, n, 200):
                i_end = min(i_start + 200, n)
                other = criteria[i_start:i_end]  # shape [m, 3]

                # diff shape: [m, c, 3]
                diff = other[:, np.newaxis, :] - cand[np.newaxis, :, :]

                # all_ge: [m, c] - True if other[i] >= cand[j] in all dims
                all_ge = np.all(diff >= 0, axis=2)
                # any_gt: [m, c] - True if other[i] > cand[j] in at least one dim
                any_gt = np.any(diff > 0, axis=2)

                # dominates: [m, c] - True if other[i] dominates cand[j]
                dominates = all_ge & any_gt

                # For each candidate j, if any i dominates it, mark as dominated
                # But we need to exclude i == j (same point)
                # The indices in the block: candidates are s_start:s_end, others are i_start:i_end
                # We need to mask out i == j when they overlap.

                # Find overlap: candidates with global indices in [i_start, i_end)
                # Candidate global indices: s_start + arange(s_end - s_start)
                cand_global = np.arange(s_start, s_end)
                other_global = np.arange(i_start, i_end)

                # Create mask for i == j
                # For each (i_idx, j_idx) in dominates, check if other_global[i_idx] == cand_global[j_idx]
                # This is a bit complex. Let's just accept that a point doesn't dominate itself
                # in the strict sense (since it's equal, not strictly greater in any dim).
                # Actually, if i == j, then diff is all zeros, so all_ge is True but any_gt is False.
                # So dominates[i, j] is False when i == j. Good.

                # So we can just check if any i dominates j
                is_dominated_in_block = np.any(dominates, axis=0)  # shape [c]

                # Mark candidates as not on frontier
                local_indices = np.where(is_dominated_in_block)[0]
                if len(local_indices) > 0:
                    global_indices = s_start + local_indices
                    on_frontier[global_indices] = False

    # Now on_frontier marks points on the Pareto frontier
    # For points not on the frontier, we still need to assign a score.
    # The task says "selecting only those on the efficient frontier".
    # But we must return 2000 finite scores. So for non-frontier points, we give a low score.
    # For frontier points, we use a composite score.

    # For frontier points, use EI / cost as the score (cost-aware EI)
    # For non-frontier points, use a very low score.

    # Actually, let's use EI / cost for all, but boost frontier points.
    # The directive says "selecting only those on the efficient frontier to balance exploration and cost".
    # I think the intent is to only consider frontier points for selection, but we need to return scores for all.
    # Let's give frontier points a high score based on their Pareto position, and non-frontier points a low score.

    # Simple approach: score = ei / cost for frontier points, and 0 (or very small) for others.
    # But we need to ensure finite scores.

    # Let's use: for frontier points, score = ei / cost. For non-frontier, score = 0.
    # But this might be too aggressive. Let's use a small epsilon for non-frontier.

    base_score = ei / np.maximum(cost, 1e-12)

    # For non-frontier points, reduce the score significantly
    scores = base_score * on_frontier.astype(float)

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)

    return scores
# EVOLVE-BLOCK-END
