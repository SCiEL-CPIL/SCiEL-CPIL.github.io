"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 5.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement efficiently."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n_candidates = len(mu)
    
    # Safety check: if we are running out of time, return a simple fallback
    if _time_exceeded():
        # Fallback: Simple EI / cost
        ei = _compute_ei(mu, sigma, f_best)
        score = ei / (cost + 1e-9)
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Strategy: Budgeted Sequential Decision (Greedy VOI)
    # 
    # The standard approach scores each point independently: score_i = Utility_i / Cost_i.
    # This ignores the fact that choosing a cheap point i allows for more future evaluations.
    # 
    # Here, we approximate the value of the *sequence* of evaluations.
    # We simulate the budget consumption. We assume that after evaluating a point,
    # the GP updates. However, we cannot update the GP here (no coords).
    # 
    # Instead, we use a "Remaining Budget Value" heuristic.
    # Let B be the total budget (30).
    # Let C_i be the cost of candidate i.
    # 
    # A more sophisticated approach:
    # We want to maximize E[Improvement] over the whole budget.
    # Approximation:
    # If we pick point i, we spend C_i. We have B - C_i left.
    # The value of the rest of the budget depends on the new information.
    # 
    # Since we can't update the GP, we use a static approximation of the "Value of Information" (VOI).
    # VOI_i = E[Improvement from i] + (Remaining Budget Factor) * E[Future Improvement potential].
    # 
    # A simpler structural change:
    # Instead of dividing by cost, we use a "Cost-Adjusted Regret" that accounts for the 
    # *opportunity cost* of the budget.
    # 
    # Let's try a "Discounted EI" where the discount factor depends on how much budget 
    # is consumed relative to the total budget, and how "promising" the point is.
    # 
    # Actually, a known effective strategy for budgeted BO is to maximize:
    # Score = EI / Cost^alpha
    # where alpha is tuned. The current program uses alpha=0.5.
    # 
    # Let's try a different structural idea: **Expected Regret Reduction per Unit Cost** 
    # but with an **Exploration Bonus** that is *budget-aware*.
    # 
    # The "different strategy" directive suggests a different design family.
    # Let's implement a **Greedy Sequential Selection** approximation.
    # We will simulate the selection process.
    # We assume the budget is 30.
    # We don't know the current total cost spent, but we know `cost` is the cost of *new* points.
    # The prompt says: "While total cost is below 30...". 
    # `f_best` is the lowest observed. `cost` array contains costs of the 2000 candidates.
    # We don't know how much budget is left. 
    # However, the loop runs "While total cost is below 30".
    # The evaluator picks the best score, evaluates it, updates total cost.
    # 
    # If we assume the budget is tight, we should prefer cheap points that have high EI.
    # If we assume the budget is loose, we should prefer points with high EI regardless of cost.
    # 
    # Since we don't know the current total cost, we can't adaptively change the strategy 
    # based on remaining budget unless we infer it.
    # 
    # Alternative Structural Change: **Probability of Improvement (PI) with Cost Penalty**
    # PI is more robust to scale differences than EI.
    # Score = PI - lambda * Cost
    # But PI is in [0,1], Cost is in [1, 2]. Lambda needs tuning.
    # 
    # Let's try **Normalized EI** where we normalize the EI by the standard deviation of the 
    # objective function (estimated from sigma) to make it dimensionless, then apply a 
    # cost-aware discount that is non-linear.
    
    # Actually, the most distinct structural change available without coordinates is 
    # to switch from a *myopic* acquisition (EI/cost) to a *lookahead* approximation.
    # 
    # Lookahead Approximation:
    # 1. Estimate the "Value" of each candidate as EI.
    # 2. Estimate the "Opportunity Cost" as the EI of the *second best* candidate that 
    #    would have been selected if we hadn't picked the first one, weighted by cost.
    #    This is too complex.
    
    # Let's go with **Cost-Weighted UCB** but with a **dynamic kappa** based on the 
    # distribution of costs.
    # 
    # If most candidates are cheap, we can afford to be more exploratory (higher kappa).
    # If most candidates are expensive, we must be exploitative (lower kappa).
    
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    
    # Normalize cost to [0, 1]
    cost_norm = (cost - cost_min) / cost_range
    
    # Calculate EI
    ei = _compute_ei(mu, sigma, f_best)
    
    # Calculate UCB term
    # We want to balance exploitation (EI) and exploration (sigma).
    # Standard UCB: mu + kappa * sigma
    # But we want to minimize, so we want low mu and high sigma.
    # Let's use a "Lower Confidence Bound" (LCB) for minimization:
    # LCB = mu - kappa * sigma
    # We want to maximize the *negative* LCB? No, we want to find the minimum.
    # So we want to pick points with low mu and high sigma.
    # Score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    # 
    # Let's blend EI and UCB.
    # EI is already a good measure of improvement.
    # 
    # New Idea: **Adaptive Cost Penalty**
    # The penalty for cost should be higher when the EI is low, and lower when EI is high.
    # Score = EI / (1 + beta * cost_norm)
    # This is a rational penalty.
    
    # Let's try a **Pareto Dominance** approach.
    # We want high EI and low Cost.
    # Score = EI * (1 / (1 + cost_norm))
    
    # To make it structurally different, let's use a **Geometric Mean** of normalized EI and 
    # normalized "Cost Efficiency".
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Cost Efficiency: 1 is best (cheapest), 0 is worst (most expensive)
    cost_eff = 1.0 - cost_norm
    
    # Geometric Mean
    score = np.sqrt(ei_norm * cost_eff)
    
    # This is a different design family: Multiplicative combination (Geometric Mean) 
    # vs Divisive (EI/Cost).
    # The geometric mean is more balanced and less prone to extreme values.
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
