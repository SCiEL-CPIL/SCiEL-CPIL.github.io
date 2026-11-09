"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: return uniform random scores if we're running out of time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.random.uniform(0, 1, size=len(mu))
    
    # Safety for numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Dynamic exploration coefficient
    # We estimate the number of evaluations based on total cost spent.
    # The loop starts with 5 points. Total cost budget is 30.
    # Average cost is roughly 1.5 (1 + mean(x) where x in [0,1]^6, mean(x)=0.5 -> 1.5).
    # So we expect about 30/1.5 = 20 evaluations total.
    # Let's use the current total cost to estimate progress.
    # The 'cost' input is the cost of evaluating each candidate, not the total spent.
    # However, we don't have access to total spent directly in the function signature.
    # We can infer it if we assume the caller passes the same cost vector structure, 
    # but 'cost' is (2000,) corresponding to the 2000 candidates.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
    # And "Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # The function does NOT receive the total cost spent so far.
    # It only receives mu, sigma, f_best, and the cost of each candidate.
    # Therefore, we cannot dynamically adapt beta based on total time/cost spent unless we track it externally.
    # Since we can only modify the function body and imports, we can use a global variable to track calls.
    
    # Let's use a global counter to estimate the iteration number
    if not hasattr(acquisition, '_call_count'):
        acquisition._call_count = 0
    acquisition._call_count += 1
    
    # Each call to acquisition corresponds to one iteration of the loop (evaluating 1 point).
    # The loop starts after 5 initial points. So call 1 is the 6th point.
    # Total budget 30. Initial cost ~ 5 * 1.5 = 7.5. Remaining ~ 22.5.
    # Expected remaining evals ~ 15.
    # Let's set a max expected iterations for beta decay.
    max_iters = 20
    
    # Beta decays from a high value (exploration) to a low value (exploitation)
    # High beta early on helps find the global minimum in multimodal functions.
    # Low beta later on refines the best found.
    # UCB = mu - beta * sigma (since we are minimizing, we want low mu and high sigma is good for exploration)
    # Actually, standard UCB for minimization is: mu - beta * sigma.
    # We want to maximize this score.
    
    # Let's try a fixed beta first, or a simple decay.
    # If we use a fixed beta, it might be too exploratory or not.
    # Let's try beta = 1.0 / (1 + acquisition._call_count * 0.1)
    # Or simply a constant beta that balances well.
    # In Hartmann 6d, the function is multimodal. Exploration is key.
    # A common choice for UCB is beta ~ 2 or 3.
    # Let's use a decay: start at 3.0, end at 0.5.
    
    progress = min(acquisition._call_count / max_iters, 1.0)
    beta = 3.0 * (1 - progress) + 0.5 * progress
    
    # UCB score: mu - beta * sigma
    ucb_score = mu - beta * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # But we must be careful: dividing by cost might penalize high-cost points too much
    # if they have high potential.
    # Let's try: score = ucb_score / cost_safe
    
    score = ucb_score / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
