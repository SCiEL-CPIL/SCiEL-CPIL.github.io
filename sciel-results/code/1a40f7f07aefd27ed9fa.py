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
    n = len(mu)
    
    # 1. Compute Expected Improvement (EI)
    # We want to minimize the function, so EI is the expected reduction in f_best.
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # 2. Define Criteria for Pareto Frontier
    # We want to:
    # - Maximize EI (exploitation potential)
    # - Maximize Sigma (exploration potential / uncertainty)
    # - Minimize Cost (efficiency)
    
    # Convert to a maximization problem for all criteria:
    # c1 = EI (max)
    # c2 = Sigma (max)
    # c3 = -Cost (max, since we want to minimize cost)
    
    c1 = ei
    c2 = sigma
    c3 = -cost
    
    # 3. Identify Pareto Frontier
    # A point is on the Pareto frontier if there is no other point that is better
    # in all criteria and strictly better in at least one.
    # 
    # Efficient check:
    # A point i is dominated if there exists j != i such that:
    # c1[j] >= c1[i] AND c2[j] >= c2[i] AND c3[j] >= c3[i] AND (c1[j] > c1[i] OR c2[j] > c2[i] OR c3[j] > c3[i])
    #
    # For large N, O(N^2) is feasible (2000^2 = 4M ops), but we can use a vectorized 
    # "max-projection" heuristic which is O(N*D) but slightly less strict than full 
    # Pareto definition if we just check against the global max. 
    # However, the strict definition requires checking against *all* points.
    # 
    # Let's use a robust O(N^2) vectorized approach in blocks to ensure correctness
    # and avoid memory explosion, but optimized with NumPy broadcasting.
    
    on_frontier = np.ones(n, dtype=bool)
    
    # We will check domination in blocks of candidates to manage memory.
    # For each candidate block, we check if it is dominated by ANY point in the entire set.
    
    block_size = 100
    
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        
        # Criteria for the candidate block: shape [b, 3]
        cand_c1 = c1[start:end]
        cand_c2 = c2[start:end]
        cand_c3 = c3[start:end]
        
        # We need to check if any point j in [0, n) dominates any point i in [start, end)
        # j dominates i if:
        # c1[j] >= c1[i] AND c2[j] >= c2[i] AND c3[j] >= c3[i] AND (c1[j] > c1[i] OR c2[j] > c2[i] OR c3[j] > c3[i])
        
        # To do this efficiently without O(b*n) memory blowup per block step (which is fine for b=100, n=2000 -> 200k elements),
        # we can iterate over j in blocks too, or just vectorize the whole thing if memory allows.
        # 2000 * 100 * 3 floats is ~2.4 MB. This is very safe.
        
        # Let's compute the domination matrix for this block of candidates vs ALL points.
        # diff_c1[j, i] = c1[j] - cand_c1[i]
        # We want to know if ANY j satisfies the condition for a given i.
        
        # c1_all = c1[:, np.newaxis] # shape [n, 1]
        # cand_c1_block = cand_c1[np.newaxis, :] # shape [1, b]
        
        # This creates an array of shape [n, b]
        # is_better_or_equal_c1 = c1[:, np.newaxis] >= cand_c1[np.newaxis, :]
        # is_better_or_equal_c2 = c2[:, np.newaxis] >= cand_c2[np.newaxis, :]
        # is_better_or_equal_c3 = c3[:, np.newaxis] >= cand_c3[np.newaxis, :]
        
        # is_strictly_better_any = (c1[:, np.newaxis] > cand_c1[np.newaxis, :]) | \
        #                           (c2[:, np.newaxis] > cand_c2[np.newaxis, :]) | \
        #                           (c3[:, np.newaxis] > cand_c3[np.newaxis, :])
        
        # dominates = is_better_or_equal_c1 & is_better_or_equal_c2 & is_better_or_equal_c3 & is_strictly_better_any
        
        # dominated_in_block = np.any(dominates, axis=0) # shape [b]
        
        # However, we must exclude the case where j == i (a point does not dominate itself).
        # If j == i, then all "greater" conditions are false, so dominates[j, i] will be False anyway.
        # So we don't need to explicitly mask j==i.
        
        # Let's compute it.
        # Note: n=2000, b=100. Arrays are 2000x100. This is 200k booleans. Very fast.
        
        ge_c1 = c1[:, np.newaxis] >= cand_c1[np.newaxis, :]
        ge_c2 = c2[:, np.newaxis] >= cand_c2[np.newaxis, :]
        ge_c3 = c3[:, np.newaxis] >= cand_c3[np.newaxis, :]
        
        gt_any = (c1[:, np.newaxis] > cand_c1[np.newaxis, :]) | \
                 (c2[:, np.newaxis] > cand_c2[np.newaxis, :]) | \
                 (c3[:, np.newaxis] > cand_c3[np.newaxis, :])
        
        dominates = ge_c1 & ge_c2 & ge_c3 & gt_any
        
        is_dominated = np.any(dominates, axis=0)
        
        if np.any(is_dominated):
            local_indices = np.where(is_dominated)[0]
            global_indices = start + local_indices
            on_frontier[global_indices] = False

    # 4. Score Assignment
    # Points on the frontier are the only ones we really care about for selection.
    # We assign a score to all points, but non-frontier points get a very low score.
    
    # For frontier points, we want to pick the "best" one.
    # A common cost-aware metric is EI / Cost.
    # However, we also value uncertainty. 
    # Let's use a composite score for frontier points:
    # score = (EI + alpha * Sigma) / Cost
    # But the directive says "without arbitrary weight tuning".
    # The Pareto frontier already balances these. 
    # Among frontier points, we can just use EI/Cost as a proxy for "efficiency of improvement".
    # Or, since they are all non-dominated, we might just pick the one with the highest EI, 
    # or lowest Cost, or highest Sigma? 
    # Actually, in a Pareto frontier, there is no single "best". 
    # But we must return a single score to select the *next* point.
    # Typically, one picks the point that maximizes a utility function on the frontier.
    # Let's use EI / Cost as the score for frontier points, as it directly relates to 
    # "improvement per unit cost".
    
    cost_safe = np.maximum(cost, 1e-12)
    
    # Base score: EI / Cost
    base_score = ei / cost_safe
    
    # For non-frontier points, we set score to 0 (or a very small negative number) 
    # to ensure they are never selected if any frontier point exists.
    # If no frontier points exist (unlikely), we fall back to base_score.
    
    scores = np.where(on_frontier, base_score, 0.0)
    
    # Handle case where no points are on the frontier (should not happen with strict domination check)
    # Or if all scores are 0, we might want to fall back to something.
    if not np.any(on_frontier):
        scores = base_score
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
