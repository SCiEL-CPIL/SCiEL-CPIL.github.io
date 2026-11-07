import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

# We will adaptively switch between exploration (Thompson/UCB) and exploitation (EI)
# based on how many evaluations have been done. Since we don't get the count directly,
# we can infer it from the magnitude of f_best or just use a fixed schedule?
# The prompt says "until 30 evaluations including the first 5".
# We don't have access to the evaluation count.
# However, we can track calls if we assume the function is called sequentially?
# No, state is not preserved between calls in a standard stateless function unless we use global vars.
# Let's use a global counter to track the number of acquisition calls.
# This is a heuristic assumption: the evaluator calls acquisition once per step.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Fallback to simple LCB
        return -mu + 2.0 * np.maximum(sigma, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy:
    # Early on (first ~10-15 calls), we want strong exploration to find the basin of attraction
    # of the global minimum, as Weierstrass is multimodal.
    # Later, we want strong exploitation to refine the minimum.
    
    # Since we have 30 total evals (5 init + 25 BO steps), 
    # _call_count will go from 1 to 25.
    
    # Let's blend Thompson Sampling (TS) and Expected Improvement (EI).
    # TS is good for exploration but can be noisy.
    # EI is deterministic and good for exploitation.
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Thompson Sample
    # We want to minimize, so we sample from the posterior and pick the lowest value.
    # The acquisition score should be high for the point we want to pick.
    # So score = -sample.
    # But we want to blend with EI.
    # Let's normalize both to [0, 1] range to make blending stable.
    
    # Normalize EI
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
        
    # For TS, we can't just normalize the raw sample because it's stochastic.
    # Instead, let's use a UCB-like term which is deterministic and represents the "potential" of the point.
    # Or, we can use a "Probabilistic" approach where we weight the likelihood of improvement.
    
    # Let's stick to the inspiration from p65 (LCB) and p67 (WEI/UCB) but improve it.
    # The best score was 2.1553 with simple LCB (beta=2.5).
    # The current program (TS) scored 1.4974.
    # Inspiration 2 (WEI+UCB) scored 2.1528.
    
    # It seems simple LCB with a high beta works well.
    # Let's try an adaptive beta LCB.
    # High beta early on for exploration, lower beta later for exploitation.
    
    # Total steps = 25.
    # We can map _call_count to a beta value.
    # Start with beta=3.0, end with beta=1.5.
    # Linear decay.
    
    if _call_count <= 1:
        # First call after initialization
        beta = 3.0
    else:
        # Decay beta
        # t is from 1 to 25
        # beta = 3.0 - (1.5 * (_call_count - 1) / 24.0)
        # Let's use a smoother decay
        progress = min(_call_count / 25.0, 1.0)
        beta = 3.0 - 1.5 * progress
        
    # LCB for minimization: maximize -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
