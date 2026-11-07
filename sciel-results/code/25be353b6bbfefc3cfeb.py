# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function: LCB for exploration early on, EI for exploitation later.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check for time (330s limit, we are fast, but just in case)
    if time.time() - start_time > 325:
         # Fallback to something very fast if we somehow stalled
         return -mu

    # Determine the stage of the optimization
    # The loop runs for 30 evaluations total.
    # We don't know the exact current step, but we can infer it from the quality of f_best
    # or simply use a fixed threshold. 
    # A common heuristic is to explore for the first 20-30% of the budget.
    # However, without a counter, we can look at the spread of sigma or mu.
    # Actually, the simplest robust approach without state is to use a fixed switch.
    # But wait, we can estimate the "age" by looking at how close f_best is to the known minimum?
    # No, that's dangerous. 
    # Let's look at the distribution of mu. In early stages, mu is flat. 
    # A better proxy for "early stage" in a stateless function is difficult.
    
    # Alternative Strategy: 
    # The prompt says "different strategy". 
    # The current program is pure EI. 
    # Pure EI is known to be greedy and can get stuck in local minima if the initial points are bad.
    # LCB is more exploratory.
    # A hybrid that is *adaptive* based on the current sigma values is better.
    
    # Let's try a "Soft-UCB" or "GP-UCB" variant with a time-dependent beta if we could track time.
    # Since we can't track time across calls easily (stateless), we must rely on the inputs.
    
    # Insight: In the very first few calls, sigma is high everywhere. 
    # As we collect points, sigma drops near observed points.
    # If we use LCB with a large beta, we always pick the highest uncertainty.
    # If we use EI, we pick the highest potential improvement.
    
    # Let's try a weighted combination that shifts from exploration to exploitation 
    # based on the maximum sigma relative to the initial scale? 
    # Initial sigma for standardized targets is ~1.0.
    # As we get more points, max(sigma) decreases.
    
    max_sigma = np.max(sigma)
    
    # If max_sigma is high (> 0.5), we are likely in the early phase.
    # If max_sigma is low (< 0.2), we are in the late phase.
    
    # Exploration score (LCB): -mu + beta * sigma
    # Exploitation score (EI): (f_best - mu) * Phi(z) + sigma * phi(z)
    
    beta = 2.0 # Aggressive exploration
    
    # Compute LCB scores
    lcb_scores = -mu + beta * sigma
    
    # Compute EI scores
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend the scores based on the remaining uncertainty in the domain
    # If max_sigma is large, trust LCB more.
    # If max_sigma is small, trust EI more.
    
    # Normalize max_sigma to [0, 1] assuming initial std is ~1.0
    # If max_sigma > 0.5, we are early.
    # weight_lcb = 1.0 if max_sigma > 0.5 else 0.0
    # This is a hard switch. Let's make it soft.
    
    # Transition zone: 0.2 to 0.5
    # weight_lcb = 1 - smoothstep(max_sigma, 0.2, 0.5)
    
    def smoothstep(x, edge0, edge1):
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3 - 2 * t)
    
    # If max_sigma is 0.5, weight_lcb = 0 (pure EI)
    # If max_sigma is 0.2, weight_lcb = 1 (pure LCB)
    # Wait, early on max_sigma is high (close to 1). Late on max_sigma is low.
    # So if max_sigma is HIGH, we want EXPLORATION (LCB).
    # If max_sigma is LOW, we want EXPLOITATION (EI).
    
    # So:
    # max_sigma = 1.0 -> weight_lcb = 1.0
    # max_sigma = 0.2 -> weight_lcb = 0.0
    
    weight_lcb = 1.0 - smoothstep(max_sigma, 0.2, 0.8)
    
    # Check time again
    if time.time() - start_time > 328:
        return lcb_scores # Fallback
    
    final_scores = weight_lcb * lcb_scores + (1 - weight_lcb) * ei_scores
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
