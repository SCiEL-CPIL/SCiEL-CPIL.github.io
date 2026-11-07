import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement per unit cost.
    
    mu: array of predicted means
    sigma: array of predicted standard deviations
    f_best: current best observed value
    cost: array of evaluation costs for each candidate
    """
    # Ensure inputs are numpy arrays and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard Expected Improvement for minimization
    # Improvement = f_best - y
    # E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # However, a simpler and often more robust form for UCB-like behavior is:
    # We want to maximize the probability of finding a point better than f_best,
    # weighted by the magnitude of improvement, but penalized by cost.
    
    # Let's use a variant of UCB that is more robust to cost scaling.
    # The previous strategy was (f_best - mu + beta * sigma) / cost.
    # The insight suggests using a higher gamma or different structure.
    
    # Strategy: Cost-Effective Expected Improvement (CE-EI)
    # We approximate EI as max(0, f_best - mu + kappa * sigma) is not quite right for EI,
    # but for a heuristic score, we can use the "potential improvement" 
    # which is f_best - (mu - kappa * sigma) = f_best - mu + kappa * sigma.
    
    # To make it cost-aware in a more sophisticated way, we divide by cost^gamma.
    # The prompt's inspiration mentions gamma=3.0 worked well (1.227 -> 1.516).
    # Let's adopt that structural change: high gamma to favor low-cost regions.
    
    # We also need to handle the case where f_best is very low (close to optimum).
    # If f_best is near the minimum, the improvement term becomes small,
    # and we might want to switch to pure exploration or exploitation of uncertainty.
    
    # Let's use a dynamic kappa that decreases as f_best improves?
    # No, let's stick to the structural change: Cost-normalized UCB with high gamma.
    
    # UCB for minimization: LCB = mu - kappa * sigma
    # We want to minimize LCB, so we maximize -LCB = -mu + kappa * sigma.
    # But we also want improvement over f_best.
    # Score = (f_best - mu + kappa * sigma) / cost^gamma
    
    # Let's tune kappa and gamma.
    # kappa = 1.0 was in the original.
    # gamma = 3.0 was suggested by the insight.
    
    # However, if cost is very low, cost^3 is very low, making the score huge.
    # If cost is high, score is small.
    
    # Let's add a small exploration bonus that is independent of f_best to avoid
    # collapsing to a single point if f_best is very good.
    
    kappa = 2.0  # Increased exploration
    gamma = 3.0  # Strong cost penalty
    
    # Base score: Potential improvement
    improvement_term = f_best - mu + kappa * sigma
    
    # Ensure improvement term is not negative? 
    # If mu > f_best, the GP thinks this point is worse than best.
    # But sigma might be high, so there's a chance it's better.
    # The term f_best - mu + kappa * sigma can be negative if mu is much larger than f_best.
    # We should probably clip or handle this.
    # Actually, if the score is negative, it means we don't expect improvement.
    # But we still need to return finite scores.
    
    # Let's use a softplus or just clip to 0?
    # If we clip to 0, we might stop exploring regions where GP is uncertain but mean is high.
    # But if mean is high, it's unlikely to be the minimum.
    # Let's keep the raw value but ensure it's finite.
    
    # Cost term
    cost_term = cost ** gamma
    
    score = improvement_term / cost_term
    
    # Add a small exploration bonus for high uncertainty in cheap regions
    # This helps when improvement_term is small (near optimum)
    # Bonus = sigma / cost^gamma * small_factor
    # This encourages looking at uncertain cheap spots even if mean is not low.
    
    exploration_bonus = (sigma / cost_term) * 0.1
    
    total_score = score + exploration_bonus
    
    # Ensure finite values
    total_score = np.nan_to_num(total_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return total_score
