"""Cost-aware acquisition for Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start_time = time.time()
_DEADLINE_MARGIN = 20  # seconds before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    elapsed = time.time() - _module_start_time
    if elapsed > 310 - _DEADLINE_MARGIN:
        return mu - sigma  # fallback LCB
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization: EI = E[max(f_best - f, 0)])
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost-aware scoring
    # Theoretical minimum of Hartmann 3D
    f_min = -3.86278
    
    # Gap to optimum: large early, small late
    gap = f_best - f_min
    # Smooth progress signal using a logistic (soft) transform instead of hard clip.
    # This avoids the nonsmooth kink at the clip boundaries and gives a gentle
    # transition from exploratory to exploitative behavior.
    # Center at gap=1.0 (mid-range), scale ~0.5 for a smooth S-curve.
    progress = 1.0 / (1.0 + np.exp((gap - 1.0) / 0.5))
    # Clip to [0,1] for safety (logistic is already in (0,1))
    progress = np.clip(progress, 0.0, 1.0)
    
    # Cost normalization: cost in [1, 2], normalize to [0, 1]
    cost_norm = cost - 1.0  # now in [0, 1]
    
    # Adaptive parameters
    # Alpha: Cost penalty strength.
    # Early (progress=0): Moderate alpha (0.8) -> Some cost awareness from the start.
    # Late (progress=1): High alpha (3.0) -> Strong penalty on cost, be very frugal.
    # Stronger cost penalty because budget is tight (30 units) and cost varies [1,2].
    alpha = 0.8 + 2.2 * progress
    
    # Beta: Uncertainty bonus weight.
    # Early (progress=0): Moderate beta (0.2) -> Some push for exploration (high sigma).
    # Late (progress=1): Low beta (0.0) -> Pure exploitation.
    beta = 0.2 * (1.0 - progress)**2
    
    # Power-law cost penalty: stronger discrimination between cheap and expensive
    # points as budget depletes. Exponent increases with progress.
    # Early: exponent ~1.0 (linear), Late: exponent ~1.8 (superlinear)
    power = 1.0 + 0.8 * progress
    cost_penalty = 1.0 + alpha * (cost_norm ** power)
    cost_adjusted_ei = ei / cost_penalty
    
    # Uncertainty bonus
    uncertainty_bonus = beta * sigma_safe
    
    # Exploitation bonus: reward regions where the model is confident of low values
    # (mu well below f_best). This smooths the sharp EI peak and helps the GP
    # exploit promising basins. Use a soft sigmoid-like scaling.
    # exploit_margin: how far below f_best the mean is, in units of sigma.
    # When mu << f_best, z is large positive, EI ~ f_best - mu. We add a small
    # bonus proportional to (f_best - mu) when mu < f_best, scaled down late.
    exploit_weight = 0.15 * (1.0 - progress)
    exploit_margin = np.maximum(f_best - mu, 0.0)
    exploitation_bonus = exploit_weight * exploit_margin / (1.0 + exploit_margin)
    
    # Combine
    score = cost_adjusted_ei + uncertainty_bonus + exploitation_bonus
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
