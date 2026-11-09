import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

# Constants for the specific task context
# Hartmann 6D minimum is approx -3.32237
# The function values are negative.
# We are minimizing.
# f_best is the lowest observed value (most negative).
# mu and sigma are in objective units.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a dynamic LCB (Lower Confidence Bound)
    strategy.
    
    Strategy:
    1. Calculate the Lower Confidence Bound: LCB = mu - beta * sigma
       Since we are minimizing, we want to pick points with low LCB.
       However, the return value must be HIGHER for better candidates.
       So we will return -LCB / cost.
       
    2. Dynamic Beta:
       The exploration coefficient beta starts high and decays as the cost
       accumulates. This allows the algorithm to explore uncertain regions
       early on (when we have few points and high sigma) and focus on
       exploiting the best known regions later (when cost is high and we
       need to ensure we find the minimum).
       
       We estimate the "budget usage" from the `cost` array. 
       The task says: "While total cost is below 30...".
       The `cost` array contains the cost of *each* of the 2000 candidates? 
       Wait, the prompt says: "mu, sigma and cost have shape (2000,)."
       And "Evaluating x costs 1 + mean(x scaled to [0, 1])".
       The `cost` passed to the function is likely the *potential* cost of evaluating
       each of the 2000 candidates. 
       
       How do we know the *total* cost spent so far?
       The prompt says: "Runs start with 5 uniform random points (cost counted). 
       While total cost is below 30, the evaluator scores 2000 new uniform candidates 
       with acquisition and evaluates the best one."
       
       The function signature is `acquisition(mu, sigma, f_best, cost)`.
       It does *not* receive the current total cost.
       However, `f_best` changes as we find better points.
       Also, the number of points evaluated is not directly given.
       
       Can we infer the stage of the search?
       Not directly from the signature. 
       However, we can use a heuristic based on the *distribution* of costs or 
       just a fixed schedule based on a global counter? 
       No, the function is stateless between calls except for globals.
       We can maintain a global counter of how many times `acquisition` has been called.
       Each call corresponds to one iteration of the loop (evaluating one point).
       
       Let's assume the loop runs until total_cost >= 30.
       Initial cost: 5 points. Average cost per point ~ 1 + 0.5 = 1.5? 
       Min cost 1, Max cost 2.
       If avg cost is 1.5, 30/1.5 = 20 evaluations.
       If avg cost is 1.0, 30/1.0 = 30 evaluations.
       So roughly 20-30 iterations.
       
       We can use a global counter `_call_count` to estimate the progress.
       
    3. Cost Normalization:
       We divide by the cost of the candidate to prefer cheaper evaluations.
       
    4. Finiteness and Safety:
       Ensure no NaNs or Infs.
    """
    
    # Check time budget
    if time.time() > _DEADLINE:
        # Fallback: return random-ish scores to satisfy interface
        return np.random.rand(len(mu))
    
    # Global call counter to estimate search progress
    # Since we don't have access to total_cost, we approximate the stage
    # by the number of times this function has been called.
    global _CALL_COUNT
    if not hasattr(acquisition, '_CALL_COUNT'):
        acquisition._CALL_COUNT = 0
        
    acquisition._CALL_COUNT += 1
    current_call = acquisition._CALL_COUNT
    
    # Estimated total evaluations needed is ~20-30.
    # Let's assume a max of 30 calls for the beta decay schedule.
    max_expected_calls = 30.0
    progress = min(current_call / max_expected_calls, 1.0)
    
    # Dynamic Beta:
    # Start with high exploration (beta=3.0) and decay to low exploitation (beta=0.5).
    # Using a smooth decay: beta = beta_min + (beta_max - beta_min) * (1 - progress)
    beta_max = 3.0
    beta_min = 0.5
    beta = beta_min + (beta_max - beta_min) * (1.0 - progress)
    
    # Handle edge cases
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate LCB: mu - beta * sigma
    # Since we want to MINIMIZE the function, lower LCB is better.
    # But the acquisition score must be HIGHER for better candidates.
    # So we use -LCB = -mu + beta * sigma.
    
    # Wait, standard LCB for minimization is:
    # Pick x that minimizes: mu(x) - kappa * sigma(x)
    # So the "score" to maximize is: - (mu(x) - kappa * sigma(x)) = -mu(x) + kappa * sigma(x)
    
    lcb_score = -mu + beta * sigma_safe
    
    # Cost-aware normalization
    # We divide by the cost to favor cheaper points.
    # However, if we just divide by cost, points with very low cost (close to 1)
    # will have higher scores than points with high cost (close to 2), even if
    # their LCB is worse. This is the intended "cost-aware" behavior.
    
    # To prevent cost from dominating too much, we can use a softer normalization
    # or simply divide. Given the cost range is [1, 2], the ratio is at most 2x.
    # This is a reasonable trade-off.
    
    scores = lcb_score / cost_safe
    
    # Alternative: Use a weighted combination of LCB and POI?
    # The previous successful strategy was POI-based.
    # LCB is generally more robust for optimization.
    # Let's stick to LCB but ensure it's finite.
    
    # Replace non-finite values with a very low score (worst case)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Ensure the output is a valid numpy array of floats
    return scores.astype(np.float64)
