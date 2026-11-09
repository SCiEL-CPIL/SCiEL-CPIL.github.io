import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit
    # The computation is vectorized and fast, but we check just in case.
    if time.time() - _start_time > 295:
        return np.zeros(2000)

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Cost-aware Expected Improvement
    # Standard EI for minimization:
    # I(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Since we want to minimize, we define improvement as f_best - f(x).
    # If f(x) < f_best, improvement is positive.
    
    z = (f_best - mu) / sigma_safe
    
    # Avoid numerical issues with extremely large z
    z = np.clip(z, -10, 10)
    
    # Phi is CDF, phi is PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Base EI
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost Awareness:
    # The cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to penalize high cost.
    # A linear penalty is robust and aligns with the finding that sharper trade-offs work better.
    # Penalty = lambda * cost
    # We need to scale lambda appropriately.
    # The objective values are in [-0.8, 1.0] roughly.
    # f_best starts high (e.g., 0.5) and goes to -0.8.
    # Early on, f_best is high, EI is large. We can afford some cost.
    # Later, f_best is low, EI is small. We must avoid cost.
    
    # Dynamic Lambda:
    # As we spend budget, we should become more cost-sensitive.
    # Total budget is 30. Initial cost is ~5 (5 uniform points).
    # Remaining budget = 30 - current_cost.
    # However, 'cost' passed in is the cost of the *candidates*, not the current total budget spent.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,)."
    # "cost" here is the cost of evaluating each candidate x.
    # "Runs start with 5 uniform random points (cost counted). While total cost is below 30..."
    # The function signature does NOT provide the current total budget spent.
    # It only provides the cost of the candidates.
    # Therefore, we cannot adapt lambda based on remaining budget directly unless we track it.
    # We can track the number of iterations or use a heuristic.
    
    # Let's track the number of calls to approximate progress.
    # Or, simpler: use a fixed but tuned lambda.
    # The inspiration said: "increasing the cost gate exponent to 1 creates a sharper trade-off".
    # This implies a multiplicative or additive penalty with exponent 1 (linear).
    # Let's try a linear additive penalty.
    
    # How to choose lambda?
    # If f_best is around 0, EI is around 0.1-0.2.
    # Cost is 1-2.
    # If lambda = 0.1, penalty is 0.1-0.2. This is comparable to EI.
    # If lambda = 0.5, penalty is 0.5-1.0. This would dominate EI, forcing low cost only.
    # We want to find the minimum, which might be at high cost (if the shifted minimum is there).
    # But the minimum is in the inner 80%, so it's not at the very edges.
    # The cost is 1 + mean(x_scaled).
    # x_scaled in [0,1].
    # If x is near 0.5 (center), cost is 1.5.
    # If x is near 0, cost is 1.0.
    # The minimum is a uniform point in the inner 80% (0.1 to 0.9).
    # So the optimal point could have cost anywhere from 1.1 to 1.9.
    
    # A fixed lambda might be risky.
    # Let's use a slightly adaptive approach based on f_best itself.
    # As f_best decreases (gets closer to -0.8), the potential improvement (f_best - mu) shrinks.
    # We can scale lambda inversely with the "urgency" or just keep it constant.
    
    # Let's stick to a constant lambda that balances exploration and cost.
    # From the finding: "exponent to 1" (linear cost).
    # Let's try lambda = 0.2.
    
    lambda_cost = 0.2
    penalty = lambda_cost * cost
    
    score = ei - penalty
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
