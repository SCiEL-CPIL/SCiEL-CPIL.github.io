"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def _pareto_front_indices(objectives, is_max=False):
    """
    Find indices of the Pareto front in a set of objectives.
    objectives: (N, D) array.
    is_max: if True, we want to maximize all objectives. 
            If False, we want to minimize all objectives.
            
    Returns: boolean mask or indices. Here we return a boolean mask for simplicity in scoring.
    """
    n = objectives.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    
    # Normalize direction: convert all to minimization
    if is_max:
        # Negate objectives for maximization
        obj = -objectives
    else:
        obj = objectives
        
    # Fast non-dominated sorting (simple O(N^2) for N=2000 is acceptable: 4M ops)
    # But we can do better or just use the simple check.
    # Let's use a simple O(N^2) check since N=2000 is small enough for Python/NumPy vectorized inner loop if careful, 
    # but pure Python loop is slow. Let's use NumPy vectorization for the dominance check.
    
    # obj has shape (N, D)
    # A point i dominates j if obj[i] <= obj[j] for all d, and obj[i] < obj[j] for at least one d.
    
    # To find the front, a point i is in the front if NO other point j dominates i.
    
    # Vectorized approach:
    # diff = obj[:, np.newaxis, :] - obj[np.newaxis, :, :]  # (N, N, D)
    # This creates a 2000x2000xD array. If D is small (2), it's 8M floats ~ 64MB. Acceptable.
    
    D = obj.shape[1]
    # obj_i <= obj_j for all dims
    # shape (N, N) where [i, j] is True if obj[i] <= obj[j] element-wise
    # We need: (obj[i] <= obj[j]).all(axis=2) AND (obj[i] < obj[j]).any(axis=2)
    
    # Let's compute:
    # leq: (N, N, D) -> obj[:, None, :] <= obj[None, :, :]
    # lt:  (N, N, D) -> obj[:, None, :] <  obj[None, :, :]
    
    # Memory check: 2000 * 2000 * 2 * 8 bytes = 64 MB. This is fine.
    
    leq = obj[:, np.newaxis, :] <= obj[np.newaxis, :, :]
    lt = obj[:, np.newaxis, :] < obj[np.newaxis, :, :]
    
    # i dominates j if leq[i,j].all() and lt[i,j].any()
    # We want to know if i is dominated by ANY j.
    # dominated[i] = any(j such that j dominates i)
    
    # j dominates i if: obj[j] <= obj[i] for all d AND obj[j] < obj[i] for at least one d.
    # Note: leq[i, j] is obj[i] <= obj[j]. We need obj[j] <= obj[i], which is leq[j, i].
    # So we need to check if there exists j such that:
    # (obj[j] <= obj[i]).all(d) and (obj[j] < obj[i]).any(d)
    
    # Let's transpose the logic for clarity.
    # Let A = obj.
    # We want mask[i] = True if i is in front.
    # i is in front if NOT (exists j != i such that A[j] <= A[i] and A[j] < A[i] in some dim)
    
    # Compute:
    # leq_all[j, i] = (A[j] <= A[i]).all(d)
    # lt_any[j, i] = (A[j] < A[i]).any(d)
    
    # leq_all shape (N, N): leq_all[j, i] is True if A[j] <= A[i] element-wise.
    # This is equivalent to (A[j, None, :] <= A[None, i, :]).all(axis=2)
    
    # Let's compute the full arrays.
    # A_j = A[:, np.newaxis, :]  # (N, 1, D)
    # A_i = A[np.newaxis, :, :]  # (1, N, D)
    
    A_j = obj[:, np.newaxis, :]
    A_i = obj[np.newaxis, :, :]
    
    # leq_elem: (N, N, D) where [j, i, d] is True if A[j, d] <= A[i, d]
    leq_elem = A_j <= A_i
    # lt_elem: (N, N, D) where [j, i, d] is True if A[j, d] < A[i, d]
    lt_elem = A_j < A_i
    
    # leq_all: (N, N) where [j, i] is True if A[j] <= A[i] for all d
    leq_all = np.all(leq_elem, axis=2)
    # lt_any: (N, N) where [j, i] is True if A[j] < A[i] for at least one d
    lt_any = np.any(lt_elem, axis=2)
    
    # dominates[j, i] = leq_all[j, i] AND lt_any[j, i]
    dominates = leq_all & lt_any
    
    # i is dominated if there exists any j such that dominates[j, i] is True.
    # dominated[i] = any(dominates[:, i])
    dominated = np.any(dominates, axis=0)
    
    # The Pareto front is the set of non-dominated points.
    # Note: A point does not dominate itself because lt_any[j, j] is False (since A[j] < A[j] is False).
    # So self-dominance is not an issue.
    
    front_mask = ~dominated
    
    return front_mask

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto front selection between EI and Cost.
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        return -mu / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement = f_best - f(x)
    # EI = E[max(0, f_best - f(x))]
    # Assuming f(x) ~ N(mu, sigma^2)
    # Let Z = (f(x) - mu) / sigma ~ N(0, 1)
    # f_best - f(x) = f_best - mu - sigma * Z
    # Let delta = (f_best - mu) / sigma
    # EI = sigma * [delta * Phi(delta) + phi(delta)]
    
    # Handle sigma -> 0
    # If sigma is very small, EI is approx max(0, f_best - mu)
    
    delta = (f_best - mu) / sigma
    # Clip delta to avoid overflow in norm.cdf or norm.pdf? 
    # norm.cdf is stable. norm.pdf underflows for large negative, which is fine (0).
    
    # Optimization: only compute for points where mu < f_best (potential improvement)
    # Actually, the formula works for all, but if delta is very negative, EI -> 0.
    # If delta is very positive, EI -> f_best - mu.
    
    # Standard stable computation
    # EI = sigma * (delta * norm.cdf(delta) + norm.pdf(delta))
    
    # To save time, we can mask out points where mu >= f_best (EI=0)
    # But vectorized norm is fast. Let's just compute it.
    
    # One detail: if sigma is 0, division by zero. We clipped sigma to 1e-9.
    # If mu >= f_best, delta <= 0.
    # If mu == f_best, delta = 0. EI = sigma * (0 * 0.5 + 0.398) > 0.
    # Wait, if mu = f_best, the expected improvement is small but non-zero due to uncertainty.
    # However, if we are certain (sigma=0), EI should be 0.
    # With sigma=1e-9, EI is tiny.
    
    # Let's use the standard formula.
    cdf_delta = norm.cdf(delta)
    pdf_delta = norm.pdf(delta)
    
    ei = sigma * (delta * cdf_delta + pdf_delta)
    
    # Ensure EI is non-negative (numerical noise)
    ei = np.maximum(ei, 0.0)
    
    # Objectives for Pareto:
    # 1. Maximize EI
    # 2. Minimize Cost
    
    # We want to find the non-dominated set with respect to:
    # Maximize EI
    # Minimize Cost
    
    # Convert to minimization for the helper function:
    # obj[:, 0] = -EI (minimize -EI is maximize EI)
    # obj[:, 1] = Cost (minimize Cost)
    
    objectives = np.column_stack((-ei, cost))
    
    # Find Pareto front
    front_mask = _pareto_front_indices(objectives, is_max=False)
    
    # Score assignment:
    # Points on the front get a high score.
    # Points not on the front get a low score (or 0).
    # To make the selection robust, we can rank them or just use a binary mask with a large value.
    # However, if there are many points on the front, the evaluator picks the one with the highest score.
    # If scores are identical, it might pick arbitrarily.
    # To break ties, we can add a small secondary objective or just use the EI value itself as a tie-breaker?
    # The task says "score candidates based on their rank within this front".
    # Since we only return the best one, the exact score value for the front members matters only to distinguish them from non-front members.
    # If we give all front members the same score, the evaluator picks one of them (likely the first one found by argmax).
    # To be safer, we can score them by their EI value (since they are all cost-efficient, higher EI is better) 
    # or by a combination.
    # Let's assign:
    # Front members: score = EI (or -Cost, or normalized EI)
    # Non-front: score = 0 (or very negative)
    
    # Actually, if we just use EI for front members, we are effectively picking the point on the Pareto front with the highest EI.
    # This is a reasonable heuristic: among all efficient trade-offs, pick the one with the most expected improvement.
    
    score = np.zeros_like(mu)
    score[front_mask] = ei[front_mask]
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
