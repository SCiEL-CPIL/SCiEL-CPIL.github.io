import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive parameters
_call_count = 0
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _start_time
    
    _call_count += 1
    
    # Ensure sigma is non-negative and avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Time guard: if we're running out of time, fall back to a simple, fast strategy
    # The task says 300s per evaluation, but let's be safe. 
    # Actually, the acquisition function is called once per step in the loop.
    # The loop runs for 25 steps (30 total - 5 initial).
    # Each step: 2000 candidates.
    # The bottleneck is likely the GP update or the acquisition computation itself.
    # Since we only have mu and sigma, the acquisition is fast.
    # We will implement a hybrid of EI and LCB, which is known to work well.
    
    # Adaptive parameters:
    # Early on, we might want more exploration. Later, more exploitation.
    # However, Weierstrass is multimodal.
    # Let's try a combination of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    
    # EI for minimization:
    # Improvement = f_best - f(x)
    # E[Improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB for minimization (score = - (mu - beta * sigma))
    # We want to maximize this score.
    # Score_lcb = -mu + beta * sigma
    
    # We need to combine EI and LCB.
    # EI is in objective units. LCB is in objective units (negated).
    # They might have different scales.
    # A common approach is to use a weighted sum.
    # Or, use a single parameter that tunes the balance.
    
    # Let's try a simple weighted sum.
    # Since EI can be zero or small if f_best is very low, and LCB can be large if sigma is large.
    # Let's normalize them? No, that's risky if scales change.
    
    # Alternative: Use a single acquisition function that is known to be robust.
    # The inspiration 1 (score 2.9191) used a "Normalized hybrid acquisition".
    # The current program (score 1.8285) uses pure LCB with kappa=5.0.
    # The parent of the current program (in the JSON) had score 2.9191 and used EI + LCB.
    # The diff shows the parent used:
    # ei = ...
    # lcb_score = -mu + beta * sigma (beta=2.0)
    # It likely combined them.
    
    # Let's try to replicate a strong hybrid.
    # A common hybrid is: score = w_ei * EI + w_lcb * LCB
    # Or: score = EI + w * LCB
    
    # Let's look at the "Normalized hybrid" hint.
    # Maybe: score = EI / (max(EI) + eps) + LCB / (max(LCB) + eps) ?
    # Normalizing by the max of the current batch ensures both terms contribute roughly equally.
    
    max_ei = np.max(ei)
    max_lcb = np.max(-mu + 2.0 * sigma) # Using beta=2.0 as in parent
    
    # Avoid division by zero
    max_ei = max(max_ei, 1e-9)
    max_lcb = max(max_lcb, 1e-9)
    
    ei_norm = ei / max_ei
    lcb_norm = (-mu + 2.0 * sigma) / max_lcb
    
    # Weighted sum
    # Let's try equal weights first, or slightly favor EI for exploitation.
    w_ei = 0.5
    w_lcb = 0.5
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    return scores
