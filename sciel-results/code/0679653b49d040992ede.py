"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Base: Thompson Sampling. Sample z_i = mu_i + sigma_i * u_i.
    2. Adaptive Exploration/Exploitation:
       - Rastrigin is multimodal. Early on, we need to explore widely to find the basin of the global minimum (shifted per instance).
       - Later, we should exploit the current best region.
       - We modulate the effective sigma used for sampling: sigma_eff = sigma * exploration_factor.
       - exploration_factor starts high (e.g., 1.0) and decays as f_best approaches 0.
       - Since f_best is the minimum observed value, and Rastrigin min is 0,
         we can use a decay based on f_best. A simple exponential decay: exp(-lambda * f_best).
         If f_best is large (e.g., 20), exp(-0.1*20) = exp(-2) ~ 0.13 (low exploration? No, wait).
         Actually, if f_best is high, we are far from optimum, so we SHOULD explore more.
         If f_best is low, we are close, so we should exploit.
         So exploration_factor should be HIGH when f_best is HIGH.
         Let's try: exploration_factor = 1.0 / (1.0 + f_best/10.0).
         If f_best = 0, factor = 1.0. If f_best = 10, factor = 0.5. If f_best = 50, factor = 0.16.
         Wait, if f_best is high (bad), we want to explore more (higher sigma).
         If f_best is low (good), we want to exploit (lower sigma).
         So: sigma_eff = sigma * (1.0 + f_best / 5.0). 
         This increases the sampling spread when we are far from the optimum.
         
    3. Cost-Awareness:
       - We want to maximize (potential improvement) / (cost).
       - In TS, we pick the sample with the lowest value.
       - Score = -z_i / (cost_i ** alpha).
       - We can make alpha adaptive. Early on, cost matters less (we have budget). Late on, cost matters more.
       - Or simply keep alpha=1 for simplicity but ensure the noise scaling does the heavy lifting.
       
    4. Deadline Guard:
       - If time is running out, return a deterministic, low-cost score to finish quickly.
    """
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        # Return simple deterministic score to ensure termination
        return -mu / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Adaptive Exploration Factor
    # f_best is a scalar (lowest observed value).
    # If f_best is high, we are far from 0. We should explore more (larger effective sigma).
    # If f_best is low, we are close. We should exploit (smaller effective sigma).
    # Let's use a linear or hyperbolic decay.
    # f_best for Rastrigin can be up to ~20-30 initially.
    # exploration_scale = 1.0 + f_best / 10.0
    # If f_best=20, scale=3.0. If f_best=0, scale=1.0.
    # This effectively multiplies the uncertainty by up to 3x in early stages.
    
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val) or f_best_val < 0:
        f_best_val = 0.0
        
    # Cap the exploration boost to prevent wild sampling
    exploration_scale = 1.0 + min(f_best_val / 5.0, 2.0)
    
    sigma_eff = sigma * exploration_scale
    
    # Generate random samples
    u = np.random.normal(0, 1, size=mu.shape)
    z = mu + sigma_eff * u
    
    # Cost-aware score
    # We want to maximize this.
    # Lower z (better function value) -> Higher score.
    # Higher cost (more expensive) -> Lower score.
    
    cost_safe = np.maximum(cost, 1e-9)
    
    # Use a slightly stronger cost penalty in later stages?
    # For now, stick to 1/cost but the adaptive sigma helps.
    # To be safer against expensive points that don't improve much, 
    # we can use -z / (cost_safe ** 1.2)
    
    scores = -z / (cost_safe ** 1.2)
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
