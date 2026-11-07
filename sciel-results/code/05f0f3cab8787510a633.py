"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Structural change: Use a hybrid acquisition that blends Expected Improvement (EI)
    # with a lower-confidence bound (LCB) to balance exploitation and exploration more effectively.
    # The LCB component uses a time-adaptive beta parameter if possible, but since we don't have
    # explicit step information, we use a fixed but potentially more aggressive beta than standard.
    # Additionally, we add a small exploration term based on the maximum sigma to ensure
    # that regions with high uncertainty are also considered.
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (LCB)
    # Beta controls the trade-off between exploitation and exploration.
    # A larger beta means more exploration.
    beta = 2.0
    lcb = mu - beta * sigma
    
    # Normalize EI and LCB to comparable scales
    # EI is always non-negative. LCB can be negative.
    # We want to maximize the score, so higher is better.
    # For LCB, lower is better, so we negate it or use -LCB.
    # Let's use a weighted combination.
    
    # Simple approach: max(EI, -LCB * weight)
    # Or: EI + weight * (-LCB)
    
    # Let's try a different structural approach:
    # Use the "Probability of Improvement" (PI) combined with EI? No, PI is less effective.
    
    # Let's try the "Knowledge Gradient" (KG) approximation or just a better tuned EI.
    # Actually, the standard EI is quite robust. The stagnation might be due to
    # the GP model being slightly off or the search space being hard.
    
    # Let's try a different kernel or hyperparameter? No, the GP is fixed by the task.
    
    # Let's try "Expected Simple Regret" (ESR) which is similar to EI but for the best point.
    # ESR is actually the same as EI for a single point.
    
    # Let's try a "Thompson Sampling" style acquisition?
    # We can't sample from the posterior easily without the full GP machinery.
    
    # Let's stick to EI but with a slight modification:
    # "Expected Improvement with a noise term" or "Robust EI"?
    
    # Alternative: Use the "Upper Confidence Bound" (UCB) for minimization?
    # UCB = mu - beta * sigma. This is what I calculated above.
    
    # Let's combine EI and UCB.
    # score = 0.5 * ei + 0.5 * (-lcb)
    # But ei and -lcb have different scales.
    
    # Let's normalize them.
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
        
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    if lcb_max - lcb_min > 1e-10:
        lcb_norm = (lcb - lcb_min) / (lcb_max - lcb_min)
        # We want to maximize, so we want low lcb.
        # So we use 1 - lcb_norm
        lcb_score = 1 - lcb_norm
    else:
        lcb_score = lcb_norm
        
    score = 0.5 * ei_norm + 0.5 * lcb_score
    
    return score
# EVOLVE-BLOCK-END
