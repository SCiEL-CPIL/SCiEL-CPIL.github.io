# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 10.0  # seconds before the hard limit (300s)

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 290.0  # 300s limit, 10s safety margin
    return _DEADLINE

def _time_remaining():
    return max(0.0, _get_deadline() - time.time())

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using log-space multiplicative utility.
    
    Strategy:
    1. Calculate the Lower Confidence Bound (LCB): mu - kappa * sigma.
    2. Model the optimization as a multiplicative utility problem in log-space:
       log(LCB) - alpha * log(cost)
       This is equivalent to maximizing (LCB / cost^alpha).
    3. This structural change prevents the aggressive bias toward low-cost regions
       seen in linear ratios, ensuring that the exploration signal (sigma) is not
       drowned out by the cost penalty in early phases.
    """
    # Check time remaining to avoid timeout
    if _time_remaining() < 0.1:
        # Return a safe, finite score if time is almost up
        return np.zeros(len(mu))
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 0.0)
    
    # Tuning constants
    kappa = 2.0  # Exploration factor for LCB
    alpha = 0.5  # Cost penalty exponent in log-space (multiplicative penalty sqrt(cost))
    
    # Calculate LCB: mu - kappa * sigma
    lcb = mu - kappa * sigma
    
    # We want to maximize the utility.
    # The utility is defined as log(LCB) - alpha * log(cost).
    # However, LCB can be negative (which is good for minimization, as we want low values).
    # Standard log is undefined for negative numbers.
    # We can shift the LCB to be positive or use a different formulation.
    
    # Alternative formulation for minimization with log-cost penalty:
    # We want to minimize the expected value penalized by cost.
    # Let's look at the proposed formula in the directive:
    # log(mu - 2.0 * sigma) - 0.5 * log(cost)
    # This implies the "value" being logged is the LCB.
    # If LCB is negative, log is invalid.
    # In Bayesian optimization for minimization, we often deal with the fact that
    # we want to find the *most negative* value.
    
    # Let's re-evaluate the "multiplicative utility" concept.
    # If we define the "gain" as the potential improvement over f_best, it's always positive.
    # But the directive specifically suggests `log(mu - 2.0 * sigma)`.
    # This suggests that the LCB is expected to be positive or the domain is shifted.
    # In Styblinski-Tang, the minimum is -78.33. Initial points are random in [-5,5]^2.
    # The function values can be positive or negative.
    # If mu - 2*sigma is negative, we cannot take the log directly.
    
    # Robust approach:
    # The log-space formulation is often used for positive quantities.
    # For minimization, a common robust log-based acquisition is:
    # Score = -log(f_best - LCB + epsilon) - alpha * log(cost)
    # This rewards larger improvements (larger f_best - LCB) with diminishing returns,
    # which is a form of multiplicative utility on the improvement.
    
    # Let's try this:
    # Improvement potential: imp = f_best - LCB = f_best - (mu - kappa * sigma)
    # imp is positive if LCB < f_best.
    # Score = log(imp + epsilon) - alpha * log(cost)
    # This is structurally different from linear LCB/cost. It caps the reward for huge
    # improvements (logarithmic saturation) and applies a multiplicative cost penalty.
    
    lcb = mu - kappa * sigma
    imp = f_best - lcb
    
    # Ensure imp is positive for log
    imp = np.maximum(imp, 1e-6)
    
    # Cost is 1 + mean(x_scaled). x_scaled is in [0,1]. So cost is in [1, 2].
    # log(cost) is well-defined and positive.
    cost_safe = np.maximum(cost, 1e-6)
    
    score = np.log(imp) - alpha * np.log(cost_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Replace -inf with a very low number to avoid issues if the evaluator doesn't handle inf well
    # though the task says "finite scores".
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
