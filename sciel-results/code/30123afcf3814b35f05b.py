import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: return early if we're getting close to the limit
    # This is a static function, so we don't track elapsed time across calls
    # But we can add a simple guard if the computation is unexpectedly slow
    # For 2000 points, vectorized ops are fast, so this is mostly defensive
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Hyperbolic Improvement (EHI) for minimization
    # We want to maximize E[atan((f_best - f(x)) / (sigma * cost_factor))]
    # where cost_factor accounts for evaluation cost
    
    # The cost is 1 + mean(x), so higher cost means we should be more conservative
    # We normalize by cost to make the improvement per unit cost
    
    # Improvement in objective units
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Cost-aware scaling: divide improvement by cost to get improvement per unit cost
    # This makes high-cost points require larger improvements to be attractive
    cost_safe = np.maximum(cost, 1e-9)
    z_cost = z / cost_safe
    
    # Hyperbolic tangent transformation for EHI
    # tanh(z_cost) maps to (-1, 1), providing smooth saturation
    # This avoids the over-exploration of UCB in high-cost, high-uncertainty regions
    ehi = np.tanh(z_cost)
    
    # Add a small exploration bonus for high uncertainty, but scaled by cost
    # This ensures we still explore somewhat, but not excessively in expensive regions
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    # Final score: EHI + exploration bonus
    score = ehi + exploration_bonus
    
    # Handle any non-finite values
    score = np.where(np.isfinite(score), score, -1e9)
    
    return score
