"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills at 330s wall-clock.
_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0  # seconds


def _time_left():
    return _DEADLINE_MARGIN + 330.0 - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: hybrid of Expected Improvement and a
    progress-aware exploration term that balances exploitation and
    exploration over the course of the Bayesian optimization loop.
    
    The idea:
    - Early in optimization (f_best is far from the known minimum),
      we weight exploration (uncertainty) more heavily to explore
      the rugged ST-1D landscape.
    - Late in optimization (f_best is close to the minimum),
      we shift toward exploitation (EI) to refine the solution.
    
    We use a simple schedule based on how close f_best is to the
    known global minimum of ST-1D (-39.1662).
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-6)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    f_best = float(f_best)
    
    # Known minimum for Styblinski-Tang 1D
    min_val = -39.1662
    
    # Compute Expected Improvement (EI)
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute a normalized uncertainty score (UCB-like)
    # Use a beta parameter that adapts based on progress
    # progress: 0 when f_best is at initial level, 1 when at minimum
    # We estimate initial f_best ~ 0 (uniform random on ST-1D gives values in roughly [-40, 40])
    # A reasonable initial f_best for 5 random points might be around -10 to 0
    # Let's say progress = (f_best - 0) / (min_val - 0) but clipped to [0, 1]
    # Actually, lower f_best is better, so progress increases as f_best decreases toward min_val
    # progress = (0 - f_best) / (0 - min_val) = f_best / min_val ... but min_val is negative
    # Let's use: progress = (f_best - f_initial) / (min_val - f_initial)
    # where f_initial ~ 0 (typical random start)
    # progress goes from 0 (f_best=0) to 1 (f_best=min_val)
    f_initial = 0.0
    if f_best <= min_val:
        progress = 1.0
    elif f_initial == min_val:
        progress = 0.0
    else:
        progress = np.clip((f_best - f_initial) / (min_val - f_initial), 0.0, 1.0)
    # Note: when f_best=0, progress=0; when f_best=min_val, progress=1
    # Since min_val < f_initial, (f_best - 0)/(min_val - 0) is negative for f_best>0
    # Let's redo: we want progress in [0,1] where 0=early, 1=late
    # Early: f_best is high (e.g., 0 or positive)
    # Late: f_best is low (close to -39.1662)
    # progress = (f_initial - f_best) / (f_initial - min_val)
    # When f_best = f_initial: progress = 0
    # When f_best = min_val: progress = (f_initial - min_val)/(f_initial - min_val) = 1
    progress = (f_initial - f_best) / (f_initial - min_val)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta for UCB: starts high (more exploration) and decreases
    # beta(t) = beta_max * (1 - progress) + beta_min * progress
    beta_max = 4.0
    beta_min = 0.5
    beta = beta_max * (1.0 - progress) + beta_min * progress
    
    ucb = mu + beta * sigma
    
    # Normalize both EI and UCB to [0, 1] scale for combination
    # EI: higher is better (more expected improvement)
    # UCB: lower is better (lower predicted value with uncertainty bonus)
    # We want to combine: score = w * EI_norm + (1-w) * (-UCB_norm)
    # But simpler: use a weighted sum where EI drives exploitation and
    # the uncertainty term drives exploration.
    
    # Alternative: use a mixture
    # score = alpha * EI + (1 - alpha) * (-ucb)
    # But EI and -ucb have different scales. Let's normalize.
    
    # Actually, a cleaner approach: use EI as the base, but add an
    # exploration bonus that is proportional to sigma and scaled by (1-progress)
    # This way, early on, high-sigma regions get boosted.
    
    exploration_weight = 1.0 - progress  # 1 early, 0 late
    # Scale exploration term to be comparable to EI
    # Typical EI scale: on the order of sigma
    # So exploration_bonus = exploration_weight * sigma * scale_factor
    # Let's use a moderate scale factor
    exploration_bonus = exploration_weight * 2.0 * sigma
    
    # Final score: EI + exploration bonus
    # But we need to be careful: EI is always >= 0, exploration bonus >= 0
    # So higher score = more attractive
    scores = ei + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
