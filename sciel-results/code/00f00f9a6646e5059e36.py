"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track evaluation count and timing
_state = {
    'call_count': 0,
    'start_time': None,
    'last_f_best': None
}

def acquisition(mu, sigma, f_best):
    """
    Pareto-based multi-objective acquisition function.
    
    Strategy:
    - Maximizes a weighted sum of:
      1. Normalized Expected Improvement (EI): Measures potential improvement over f_best.
      2. Normalized Predictive Uncertainty (PU): Measures sigma relative to the scale of the problem.
    - The weight for PU (exploration) decreases as the number of evaluations increases,
      shifting focus from exploring the search space to exploiting promising regions.
    - This dynamic balance is intended to be more robust than fixed beta-UCB or single-sample TS.
    """
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
    _state['call_count'] += 1
    
    # Safety margin: If we are getting close to the 300s limit, we could degrade to a simple 
    # fast method, but EI+PU is O(N) with N=2000, which is extremely fast (<1ms).
    # So we don't need a complex deadline guard that changes logic, just ensure we return quickly.
    
    # Number of evaluations so far (including the current candidate selection step)
    # The contract says "until 30 evaluations including the first 5".
    # call_count 1 corresponds to the first acquisition call after the initial 5 points?
    # Actually, the loop structure is:
    # 1. Start with 5 points.
    # 2. Loop: draw 2000, compute posterior, call acquisition, evaluate best.
    # 3. Stop when total evaluations (5 + loop_count) == 30.
    # So call_count 1 is the 6th evaluation. call_count 25 is the 30th evaluation.
    
    n_evals = _state['call_count']
    
    # Dynamic weight for exploration (PU)
    # We want high exploration early, low exploration later.
    # Let w_explore decay from 1.0 to 0.1 over the course of the optimization.
    # Total steps ~ 25.
    # w_explore = 1.0 / (1.0 + 0.1 * n_evals)  -> 1.0 at n=0, ~0.28 at n=25
    # Or a linear decay: w_explore = 1.0 - 0.9 * (n_evals / 25.0)
    # Let's use a smooth decay that ensures exploration is still present but reduced.
    # A common heuristic is w = 1 / (1 + alpha * n).
    # Let's try: w_explore = 1.0 if n_evals < 5 else max(0.1, 1.0 - 0.05 * n_evals)
    # At n=5, w=0.75. At n=10, w=0.5. At n=15, w=0.25. At n=20, w=0.1. At n=25, w=0.1 (floor).
    
    if n_evals <= 5:
        w_explore = 1.0
    else:
        w_explore = max(0.1, 1.0 - 0.05 * (n_evals - 5))
        
    w_exploit = 1.0 - w_explore

    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize EI, so higher EI is better.
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in cdf/pdf for very large/small values
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Predictive Uncertainty (PU)
    # Use sigma directly as a measure of uncertainty.
    # Normalize both EI and PU to have zero mean and unit variance? 
    # Or just scale them to comparable ranges?
    # EI is in objective units. PU (sigma) is in objective units.
    # However, EI can be negative (if mu > f_best, EI is small/zero). 
    # Actually, standard EI is always >= 0.
    # If mu > f_best, z is negative, cdf(z) is small, pdf(z) is small. 
    # Wait, if mu > f_best, (f_best - mu) is negative.
    # Standard EI formula assumes we want to minimize f.
    # Improvement = f_best - f(x).
    # E[max(0, f_best - f(x))]
    # If f(x) ~ N(mu, sigma^2), then E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # This is always positive.
    
    # Normalize EI and PU to [0, 1] range approximately to make weights meaningful.
    # Max possible EI? Hard to say.
    # Let's normalize by the current max values in the candidate set.
    # This makes the score relative to the current batch of 2000 candidates.
    
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    pu = sigma
    pu_max = np.max(pu)
    if pu_max > 0:
        pu_norm = pu / pu_max
    else:
        pu_norm = np.zeros_like(pu)
        
    # Combine
    scores = w_exploit * ei_norm + w_explore * pu_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
