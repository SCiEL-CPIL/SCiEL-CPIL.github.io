import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 330s wall clock limit. 
    # We must return within a safety margin. 
    # If we are close to the limit, we return a simple fallback to avoid timeout.
    elapsed = time.time() - _start_time
    if elapsed > 325.0:
        # Fallback: return cost-based score to be safe and fast
        # Higher score = better. We want low cost, so -cost.
        return -np.asarray(cost, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and numerically stable
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Strategy: Cost-Aware Expected Improvement with Dynamic Penalty
    # 1. Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Cost-Aware Adjustment
    # The cost is in [1, 2]. 
    # We want to penalize expensive points, but the penalty should be adaptive.
    # If f_best is very close to 0 (optimal), EI is small, so even small costs matter.
    # If f_best is large, EI is large, so we can afford higher costs.
    
    # Normalize EI to [0, 1] relative to the current best improvement potential?
    # Or use a fixed scaling factor that depends on the current state of the search.
    
    # A robust approach: 
    # Score = EI - lambda * (cost - min_cost)
    # But min_cost is not passed. We know cost = 1 + mean(x_scaled).
    # So cost is in [1, 2].
    # Let's normalize cost to [0, 1] by subtracting 1.
    # cost_norm = cost - 1.0
    
    # How to choose lambda?
    # If we use a fixed lambda, it might be too aggressive or too weak.
    # Let's try a lambda that scales with the average EI.
    # avg_ei = np.mean(ei)
    # lambda = avg_ei / 10.0 ?
    
    # Alternative: Use a "Cost-Weighted EI"
    # Score = EI / (1 + cost_norm)
    # This naturally penalizes high cost.
    
    # Let's try a hybrid:
    # score = ei - 0.05 * (cost - 1.0)
    
    # To make it more structural and different from the current program (which uses a fixed lambda=0.01),
    # let's use a **Pareto-frontier** approach conceptually, but implemented as a scalarization.
    # We define a "value" for each point based on its expected improvement and cost.
    
    # Let's normalize EI by the max EI to make it scale-invariant.
    max_ei = np.max(ei)
    if max_ei > 1e-12:
        ei_norm = ei / max_ei
    else:
        ei_norm = ei
        
    # Normalize cost to [0, 1]
    # cost is in [1, 2]
    cost_norm = (cost - 1.0) / 1.0
    
    # Pareto scalarization: 
    # We want to maximize ei_norm and minimize cost_norm.
    # Score = w * ei_norm - (1-w) * cost_norm
    # Choose w based on the current phase.
    # Early on, we want to explore (high w). Late on, we want to exploit (low w, prioritize cost?).
    # Actually, we always want to minimize cost if EI is similar.
    # Let's use a dynamic weight that decreases as we get closer to the optimum.
    # How to measure "closeness"? f_best.
    # Griewank 2D minimum is 0.
    # If f_best is large, we are far. If f_best is small, we are close.
    
    # Let's use a sigmoid-like transition.
    # If f_best > 0.1, weight = 0.8 (favor EI)
    # If f_best < 0.01, weight = 0.5 (balance)
    # This is a heuristic.
    
    # Simpler structural change: **Thompson Sampling with Cost Penalty**
    # The current program uses EI. Let's switch to Thompson Sampling.
    # Draw a sample from the posterior: y = mu - sigma * z (since we are minimizing, we want low values)
    # Wait, standard TS for minimization:
    # Sample theta ~ N(mu, sigma^2)
    # Pick x with lowest theta.
    # But we need to return a score for all 2000 points.
    # So we compute a score for each point that represents its "desirability".
    # For TS, we can't just return the sample because it's random.
    # However, we can return the expected value of the sample minus a cost penalty?
    # That's just EI-like.
    
    # Let's stick to a different acquisition function: **Probability of Improvement (PI)**
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is different from EI. EI considers the magnitude of improvement, PI only the probability.
    # PI is often more exploratory in the early stages because it doesn't favor points with huge potential but low probability.
    # Let's combine PI with cost.
    
    pi = Phi_z
    pi = np.maximum(pi, 0.0)
    
    # Normalize PI?
    max_pi = np.max(pi)
    if max_pi > 1e-12:
        pi_norm = pi / max_pi
    else:
        pi_norm = pi
        
    # Score = pi_norm - lambda * cost_norm
    # Let's use a lambda that is slightly higher than the current program's to be more cost-aware.
    lambda_cost = 0.1
    score = pi_norm - lambda_cost * cost_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
