"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: ensure we return well before the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback to pure EI to be safe, though this shouldn't happen often
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Estimate progress.
    # Total evaluations = 30.
    # We don't know exactly which step we are in, but we can infer it roughly
    # or just use a fixed schedule based on elapsed time if we assume a constant
    # evaluation time. However, GP fitting time varies.
    # A more robust way: The prompt says "until 30 evaluations".
    # We can't easily count evaluations from inside acquisition without state.
    # Let's use a heuristic: 
    # Early on, we want to explore. Later, exploit.
    # Since we can't count steps, let's rely on the fact that 
    # f_best improves over time.
    
    # Alternative: Use the spread of mu and sigma to determine the phase.
    # But the most reliable signal for "progress" in a fixed budget without 
    # external state is often just assuming a linear schedule if we could count.
    
    # Wait, I can use a global counter!
    if not hasattr(acquisition, '_count'):
        acquisition._count = 0
    acquisition._count += 1
    
    # Total steps = 30.
    # First 5 are random. So acquisition is called for steps 6..30.
    # That's 25 calls to acquisition.
    # Step 1 of acquisition corresponds to eval 6.
    # Step 25 of acquisition corresponds to eval 30.
    
    total_acq_calls = 25
    current_call = min(acquisition._count, total_acq_calls)
    
    # Progress from 0 (start) to 1 (end)
    progress = current_call / total_acq_calls
    
    # Compute EI
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB
    # We need to normalize EI and UCB to similar scales to blend them.
    # EI is non-negative. UCB can be anything.
    # Let's normalize EI by its max possible value in this batch? Or just use raw?
    # Raw EI is in objective units.
    # UCB: f_best - mu - beta * sigma (minimization).
    # As f_best decreases, EI increases.
    
    # Let's try a simpler normalization:
    # Normalize EI to [0, 1] using min/max in the batch?
    # Or just use a fixed beta schedule.
    
    # Let's stick to the inspiration 1 idea: "dynamic, normalized hybrid".
    # Normalize EI:
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    # Compute UCB score (higher is better for maximization, so we negate the minimization UCB)
    # Minimization UCB: f_best - mu - beta * sigma
    # We want to maximize this score.
    # Let's pick a beta that decays.
    # At start (progress=0), high exploration. At end (progress=1), low exploration.
    # beta = beta_max * (1 - progress) + beta_min * progress
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_max * (1 - progress) + beta_min * progress
    
    ucb = f_best - mu - beta * sigma
    
    # Normalize UCB:
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_min) / ucb_range
        
    # Blend
    # Weight for EI increases with progress.
    # Weight for UCB decreases with progress.
    # At progress=0, weight_ei=0.2, weight_ucb=0.8
    # At progress=1, weight_ei=0.8, weight_ucb=0.2
    weight_ei = 0.2 + 0.6 * progress
    weight_ucb = 1.0 - weight_ei
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    return scores
# EVOLVE-BLOCK-END
