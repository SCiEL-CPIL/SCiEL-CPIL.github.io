"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: favor low cost and high uncertainty to finish gracefully
        return -cost + 0.1 * sigma

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Time-adaptive exploration coefficient
    # We have 5 initial points, and each loop iteration spends ~1 + mean(x) ~ 1.5 cost.
    # Total budget 30. Approx 20 evaluations possible.
    # Early on (low f_best magnitude or high uncertainty), explore more.
    # We don't have the exact iteration count, but we can use f_best and cost as proxies.
    # If f_best is far from the minimum (-0.8), we need to explore.
    # If cost is low, we can explore more.
    
    # A simple heuristic: 
    # beta = sqrt(2 * log(N * pi^2 / (6 * gamma))) is standard, but we don't have N.
    # Let's use a heuristic based on the "regret" so far.
    # If f_best is high (bad), increase beta.
    # If we are late in the budget (high cost), decrease beta.
    
    # Cost so far is the sum of 'cost' array? No, 'cost' is the cost of the 2000 candidates.
    # The task says: "While total cost is below 30". The 'cost' argument is the cost of the candidates being scored.
    # We don't know the total spent cost directly, but we can infer it from the number of evaluations?
    # Actually, the function signature is `acquisition(mu, sigma, f_best, cost)`.
    # The `cost` array has shape (2000,). These are the costs of the 2000 new candidates.
    # We do NOT have the total budget spent so far.
    # However, we can estimate the "phase" by looking at the spread of sigma or the value of f_best.
    
    # Alternative: Use a fixed beta that balances exploration and exploitation well for 8D.
    # In 8D, exploration is crucial. 
    # Let's try a UCB with a moderate beta.
    # UCB = -mu + beta * sigma
    # We want to minimize, so we maximize -mu + beta * sigma.
    # We also need to account for cost. The cost is 1 + mean(x).
    # Lower cost is better. So we should penalize high cost.
    # Score = -mu + beta * sigma - lambda * cost
    
    # How to choose beta and lambda?
    # If we just use -mu + beta * sigma, we ignore cost.
    # The previous program used EI / cost. This is a ratio.
    # A sum (weighted) might be more robust to the scale of cost (1 to 2).
    
    # Let's estimate the number of evaluations to set beta.
    # Initial cost: 5 points * ~1.5 = 7.5.
    # Remaining budget: 22.5.
    # Avg cost per eval: 1.5.
    # Remaining evals: 15.
    # Total evals: 20.
    
    # Standard UCB beta for N=20, d=8:
    # beta = sqrt(2 * log(20 * pi^2 * 8 / 6)) ~ sqrt(2 * log(260)) ~ sqrt(2 * 5.5) ~ sqrt(11) ~ 3.3
    # This is a reasonable starting point.
    
    # However, we don't know N exactly. Let's use a heuristic.
    # If f_best is close to -0.8, we are done exploring.
    # If f_best is far, we need to explore.
    
    # Let's try a dynamic beta.
    # beta = 2.0 + 2.0 * (f_best - (-0.8)) / (0.0 - (-0.8)) if f_best > -0.8 else 2.0
    # This scales beta from 2.0 to 4.0 as f_best goes from -0.8 to 0.0.
    # This encourages exploration when we are far from the minimum.
    
    # Also, we need to penalize cost.
    # cost is in [1, 2].
    # If we subtract cost, we favor cheap points.
    # But cheap points are just "closer to origin" (since cost = 1 + mean(x_scaled)).
    # Wait, x is in [-1, 1]. Scaled to [0, 1], x_scaled = (x+1)/2.
    # mean(x_scaled) = 0.5 + 0.5 * mean(x).
    # cost = 1 + 0.5 + 0.5 * mean(x) = 1.5 + 0.5 * mean(x).
    # So cost is higher for points with positive mean x.
    # The minimum of cosine8 is at some shifted location.
    # If the shift s_k is uniform in the inner 80%, the minimum could be anywhere.
    # Biasing towards low cost (low mean x) might be bad if the minimum is at high mean x.
    # However, the initial points are uniform.
    
    # Let's stick to a simpler, robust UCB.
    # Score = -mu + beta * sigma - lambda * (cost - 1.5)
    # Centering cost by 1.5 removes the constant offset and makes the penalty symmetric around the mean cost.
    
    # Let's try beta = 3.0 and lambda = 0.5.
    # This means 1 unit of uncertainty is worth 3 units of improvement, and 1 unit of cost (above mean) is worth 0.5 units of improvement.
    
    # To be safe and structural, let's implement a **Max-Value Entropy Search (MVES)**-like proxy or simply a **High-Probability Feasible Set** expansion.
    # But MVES is expensive.
    # Let's try **Expected Hypervolume Improvement (EHVI)**? No, single objective.
    
    # Let's go with **UCB with cost penalty**.
    # The key structural change is moving from a *ratio* (EI/cost) to a *sum* (UCB - cost_penalty).
    # Ratios can be unstable if cost is small. Sums are more linear.
    
    # Dynamic Beta:
    # We don't know the iteration. But we can use the fact that sigma decreases as we learn.
    # If max(sigma) is high, we are still exploring.
    max_sigma = np.max(sigma_safe)
    
    # If max_sigma is large, increase beta.
    # If max_sigma is small, decrease beta.
    # For risk-averse (CVaR-like) strategy, we minimize the upper quantile.
    # This corresponds to maximizing -mu - beta * sigma.
    # We use a moderate beta.
    beta = 2.0 + 2.0 * np.clip(max_sigma, 0.0, 1.0)
    
    # Cost penalty:
    # We want to minimize cost.
    # cost is [1, 2].
    # Let's penalize (cost - 1.0) so that the minimum cost (1.0) has 0 penalty.
    # lambda = 0.5
    lambda_cost = 0.5
    
    # Risk-averse acquisition: Minimize Upper Quantile (mu + beta*sigma) + Cost
    # Score to maximize: - (mu + beta*sigma) - lambda * cost
    scores = -mu - beta * sigma_safe - lambda_cost * (cost_safe - 1.0)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
