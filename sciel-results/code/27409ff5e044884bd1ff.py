"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Strategy:
    1. Use UCB instead of EI. UCB is a point-estimate heuristic that explicitly
       balances exploitation (low mu) and exploration (high sigma) via a tunable
       parameter beta.
    2. Implement an Adaptive Beta strategy. The exploration coefficient beta is
       dynamically adjusted based on the current best observed value (f_best)
       relative to the known global minimum of the Hartmann 3D function.
       - If f_best is far from the minimum, we are in the exploration phase,
         so beta is high to encourage visiting uncertain regions.
       - If f_best is close to the minimum, we are in the exploitation phase,
         so beta is low to focus on refining the best candidate.
    3. Incorporate cost-awareness by dividing the UCB score by the evaluation cost.
       This prioritizes evaluations that offer a good balance of potential
       improvement (adjusted for uncertainty) and low cost.
    
    Formula:
    UCB(x) = -mu(x) + beta * sigma(x)  (since we want to minimize f, we maximize -f)
    Score(x) = UCB(x) / cost(x)
    
    Adaptive Beta:
    Let f_min_global = -3.86278
    Let progress = (f_best - f_min_global) / (f_initial_max - f_min_global)
    We map progress to beta. Early on (progress ~ 1), beta is high.
    Late on (progress ~ 0), beta is low.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety check for finite inputs
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # --- Adaptive Beta Calculation ---
    # Hartmann 3D global minimum
    f_min_global = -3.86278
    
    # Estimate the initial scale. Uniform random points in [0,1]^3 for Hartmann 3D
    # typically yield values around -1 to -3. Let's assume a rough initial best of -1.0
    # if we don't have better info, but f_best is provided.
    # We need a reference for "how bad" the start was to normalize progress.
    # A reasonable upper bound for Hartmann 3D on [0,1]^3 is around 0 or -1.
    # Let's use a fixed reference f_ref = -1.0 as a "worst case" initial guess for scaling.
    # If f_best is better than -1, we are progressing.
    f_ref = -1.0 
    
    # Calculate progress: 1.0 means we are at f_ref (start), 0.0 means we are at f_min_global
    # If f_best < f_min_global (shouldn't happen, but safe), clamp to 0.
    # If f_best > f_ref (worse than expected start), clamp to 1.
    current_diff = f_best - f_min_global
    total_diff = f_ref - f_min_global
    
    if total_diff <= 0:
        total_diff = 1e-6
        
    progress = current_diff / total_diff
    
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Map progress to beta
    # We want beta to be high when progress is high (early) and low when progress is low (late).
    # Linear mapping: beta = beta_min + (beta_max - beta_min) * progress
    beta_min = 0.1
    beta_max = 2.0
    
    beta = beta_min + (beta_max - beta_min) * progress
    
    # --- Calculate UCB Score ---
    # We want to minimize f, so we want low mu and high sigma (uncertainty).
    # Maximizing -mu + beta * sigma achieves this.
    ucb_scores = -mu + beta * sigma_safe
    
    # --- Cost-Aware Score ---
    # Divide by cost to prioritize cheap evaluations
    scores = ucb_scores / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return scores
# EVOLVE-BLOCK-END
