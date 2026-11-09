"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0  # Evaluator kills after 330s, keep margin
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses an Adaptive Lower Confidence Bound (LCB) where the exploration
    coefficient beta decays as the evaluation cost budget is consumed.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    # Deadline check: if we are close to the limit, return a safe fallback
    # (e.g., pure exploitation or zero) to avoid timeout.
    # However, acquisition is called frequently. We assume the caller handles
    # the loop termination, but we must ensure this function is fast.
    # If we detect we are in a "danger zone" of time, we can simplify computation.
    # For now, the computation is O(N) and very fast, so we proceed.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # The total cost budget is 30.
    # We need to estimate how much budget is left.
    # The function signature does not provide total_cost_so_far directly.
    # However, `cost` is the cost of evaluating *each* of the 2000 candidates.
    # The loop logic in the evaluator: "While total cost is below 30... evaluates the best one."
    # We don't know the exact total cost spent so far inside this function.
    # BUT, we can approximate the "phase" of the optimization.
    # Early phase: High uncertainty, we want to explore.
    # Late phase: Low uncertainty, we want to exploit.
    
    # A robust proxy for "late phase" is the magnitude of f_best relative to the
    # typical scale, or simply using a fixed annealing schedule if we could track steps.
    # Since we cannot track steps easily without global state (which might be reset or unsafe),
    # we can use the spread of mu and sigma or just a well-tuned constant.
    
    # Wait, the prompt says "Runs start with 5 uniform random points... While total cost is below 30".
    # Max evaluations is roughly 30 / min_cost. Min cost is 1.0 (if x=0).
    # So max ~30 evaluations.
    # This is a very small budget.
    
    # In such a small budget, standard EI often fails to explore enough because
    # the initial GP is very uncertain.
    # LCB with a relatively high beta is better for small budgets.
    
    # Let's use a beta that is high enough to encourage exploration but
    # normalized by cost to penalize expensive points.
    
    # Structural Change: Instead of a fixed beta, let's use a beta that
    # scales with the current uncertainty of the best point? No, that's complex.
    
    # Let's try a different structural change: **Probability of Improvement (PI)**
    # with a margin, or **Thompson Sampling** approximation?
    # Thompson Sampling is hard to implement correctly without sampling from the GP.
    
    # Let's stick to LCB but change the **Cost Normalization Strategy**.
    # Current: score = (mu - beta*sigma) / cost
    # Alternative: **Cost-Effective Improvement (CEI)** style but for minimization.
    
    # Actually, the most robust structural change for small budgets (N~30) in 4D
    # is often **Pure LCB with a high exploration constant** because EI tends to
    # be "greedy" too early when the GP hasn't learned the landscape.
    
    # Let's implement a **Two-Stage Adaptive Beta**:
    # If f_best is still very high (close to random initialization levels), explore more.
    # Shekel 4d min is -10.5. Random init is likely around 0 to 5.
    # If f_best > -5, we are in early phase -> High Beta.
    # If f_best < -5, we are in late phase -> Low Beta.
    
    # This is a structural change: the acquisition function's behavior changes
    # based on the current progress (f_best) rather than a fixed constant.
    
    # Thresholds:
    # Shekel 4d range is roughly [-10.5, 5].
    # Let's define an "early" phase if f_best > -2.0
    # And a "late" phase if f_best < -2.0
    
    if f_best > -2.0:
        beta = 2.0  # Aggressive exploration
    elif f_best > -6.0:
        beta = 1.0  # Moderate exploration
    else:
        beta = 0.5  # Conservative exploitation
    
    # LCB for minimization: we want low mu and high sigma.
    # Score = mu - beta * sigma
    # We want to MAXIMIZE the score, so we negate the LCB value?
    # Standard LCB for minimization: minimize mu - beta*sigma.
    # So acquisition score (higher is better) should be: -(mu - beta*sigma) = -mu + beta*sigma.
    # Wait, the current program uses EI which is positive.
    # Let's align with the "higher is better" convention.
    # If we use LCB: lcb_val = mu - beta * sigma.
    # We want to pick the point with the LOWEST lcb_val.
    # So score = -lcb_val = -mu + beta * sigma.
    
    lcb_val = mu - beta * sigma
    score = -lcb_val
    
    # Now normalize by cost.
    # Higher cost should lower the score.
    score = score / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
