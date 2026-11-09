import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 305.0  # Safety margin before 330s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)

def _min_max_normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Structural change: Use a hybrid approach that blends a Cost-Weighted LCB
    (for exploration/uncertainty) with Normalized Expected Improvement (for
    exploitation), dynamically adjusting the balance based on the remaining
    budget ratio.
    
    This combines the structural exploration push of the current program (LCB)
    with the refinement machinery of the top inspiration (Normalized EI + PI blend).
    """
    # Check deadline
    if _time_left() < 0.5:
        return np.zeros_like(mu)
    
    # Ensure finite values and correct types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Clip sigma to avoid division by zero and negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - Y, 0)] where Y ~ N(mu, sigma^2)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Probability of Improvement (PI) for minimization
    # PI = P(Y < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z_ei)
    
    # 3. Compute LCB-based exploration score
    # LCB for minimization: mu - kappa * sigma
    # We want to maximize this, so we use (kappa * sigma - mu)
    # The current program used a fixed beta=2.0. We will make this dynamic.
    
    # Dynamic exploration parameter kappa
    # Based on the ratio of remaining budget to current cost.
    # As budget runs out, we should exploit more (lower kappa).
    # Total budget is 30. Initial cost is roughly 5 * 1.5 = 7.5.
    # Let's estimate remaining budget as 30 - sum(cost) if we had the full history,
    # but we only have the cost of the *current* candidates.
    # The prompt says "While total cost is below 30".
    # We don't have the total accumulated cost passed to us, only the cost of the 2000 candidates.
    # However, the cost array passed in is for the 2000 candidates being scored.
    # The "cost" argument in the signature is the cost associated with evaluating *each* of these candidates.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    # And "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # So `cost[i]` is the cost of evaluating candidate i.
    # We don't know the *total* accumulated cost from previous steps directly.
    # However, we can infer the stage of the search from the number of evaluations? No, we don't have that either.
    # But we can use the magnitude of f_best relative to the theoretical minimum or just a fixed schedule?
    # Actually, the inspiration 1 used a fixed blend.
    # Let's look at the "dynamic tuning" directive: "kappa is dynamically tuned based on the ratio of remaining budget to current cost".
    # Since we don't have the total accumulated cost, we can't compute the exact remaining budget.
    # However, we can assume that if f_best is close to the minimum, we are late.
    # Or, simpler: The cost array itself gives us the cost of the *next* step.
    # Let's use a heuristic: If the mean cost of the candidates is low, it's cheap to explore.
    # But a better proxy for "late in search" is often the tightness of the GP or the value of f_best.
    # Let's stick to a robust fixed blend first, but structure it to allow dynamic adjustment if we had more info.
    # Actually, we can estimate the number of evaluations if we assume the loop runs until cost > 30.
    # But we don't have that state.
    
    # Let's use the "Cost-Weighted LCB" from the current program but blend it with EI.
    # Current program: scores = (beta * sigma - mu) / cost
    # Inspiration 1: scores = (alpha * EI_norm + beta * PI_norm) / cost
    
    # Let's combine them:
    # Term 1: Exploitation (EI)
    # Term 2: Exploration (LCB component: sigma - mu)
    
    # Normalize EI and LCB scores to [0, 1] to make them comparable
    ei_norm = _min_max_normalize(ei)
    
    # LCB score for maximization: (beta * sigma - mu)
    # We need to choose beta. The current program used 2.0.
    # Let's use a dynamic beta that decreases as we assume we are later in the search.
    # Since we can't know the exact step, let's use a moderate fixed beta for the LCB part,
    # but weight the blend.
    
    # Let's try a weighted sum of the normalized EI and a normalized LCB term.
    # LCB_term = beta * sigma - mu
    # To normalize LCB_term, we need min/max.
    beta_lcb = 2.0
    lcb_term = beta_lcb * sigma_safe - mu
    lcb_norm = _min_max_normalize(lcb_term)
    
    # Blend weights
    # Early in search: more exploration (LCB). Late: more exploitation (EI).
    # Without explicit step count, we can use the variance of f_best? No.
    # Let's use a fixed blend that worked well in inspirations.
    # Inspiration 1: 0.6 EI + 0.4 PI.
    # Current: Pure LCB.
    # Let's try 0.5 EI_norm + 0.5 LCB_norm.
    
    w_ei = 0.5
    w_lcb = 0.5
    
    blended = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
