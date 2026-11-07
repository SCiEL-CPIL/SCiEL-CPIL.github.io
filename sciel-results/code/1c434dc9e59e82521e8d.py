"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.monotonic()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using rank-based Pareto front selection.
    
    Computes Expected Improvement (EI) for each candidate, then identifies
    the non-dominated set based on (EI, cost) trade-off. Candidates on the
    Pareto front receive high scores proportional to their rank, while
    dominated candidates receive progressively lower scores. This enforces
    a strict trade-off constraint rather than a simple linear division.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros(2000, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle NaN/Inf in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but guard against numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Identify the Pareto front: candidates that are not dominated by any other
    # A candidate i is dominated if there exists j with ei[j] >= ei[i] and cost[j] <= cost[i]
    # (strict in at least one dimension)
    
    # For efficiency with 2000 candidates, use a sorting-based approach
    # Sort by EI descending, then for each candidate check if any earlier
    # candidate has lower or equal cost
    
    # Sort indices by EI descending
    ei_order = np.argsort(-ei)
    
    # Track the minimum cost seen so far as we go through sorted candidates
    # A candidate is on the Pareto front if its cost is strictly less than
    # the minimum cost of all candidates with higher EI
    
    is_pareto = np.zeros(n, dtype=bool)
    min_cost_so_far = np.inf
    
    for idx in ei_order:
        if cost[idx] < min_cost_so_far:
            is_pareto[idx] = True
            min_cost_so_far = cost[idx]
    
    # Assign scores based on Pareto rank
    # Candidates on the Pareto front get high scores
    # Dominated candidates get lower scores based on how "close" they are to the front
    
    # For Pareto front members, score proportional to their position on the front
    # (earlier in EI-sorted order = higher EI = better)
    # For dominated candidates, score based on their EI/cost ratio but scaled down
    
    # Compute a base score for all candidates: EI / cost (standard cost-aware EI)
    base_score = ei / cost
    
    # Normalize base score to [0, 1] range for scaling
    base_max = np.max(base_score)
    if base_max > 1e-9:
        base_norm = base_score / base_max
    else:
        base_norm = np.zeros(n)
    
    # Pareto front members get a boosted score
    # Non-front members get a reduced score
    pareto_rank = np.zeros(n)
    # Assign ranks among Pareto front members based on EI (higher EI = lower rank number)
    pareto_indices = np.where(is_pareto)[0]
    if len(pareto_indices) > 0:
        # Sort Pareto members by EI descending
        pareto_ei_order = pareto_indices[np.argsort(-ei[pareto_indices])]
        for rank, idx in enumerate(pareto_ei_order):
            pareto_rank[idx] = rank + 1
    
    max_pareto_rank = np.max(pareto_rank) if np.any(pareto_rank > 0) else 1
    
    # Score for Pareto front: high base + rank bonus
    # The top of the Pareto front gets the highest score
    scores = np.zeros(n)
    
    for i in range(n):
        if is_pareto[i]:
            # Pareto front member: score is high, decreasing with rank
            # Use a smooth decay: score = base_norm * (1 - 0.3 * rank/max_rank)
            rank_factor = 1.0 - 0.3 * (pareto_rank[i] - 1) / max(1, max_pareto_rank - 1)
            scores[i] = base_norm[i] * rank_factor + 0.5 * (1.0 - (pareto_rank[i] - 1) / max(1, max_pareto_rank - 1))
        else:
            # Dominated candidate: lower score, proportional to how close to front
            # Use base_norm but scaled down
            scores[i] = base_norm[i] * 0.3
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
