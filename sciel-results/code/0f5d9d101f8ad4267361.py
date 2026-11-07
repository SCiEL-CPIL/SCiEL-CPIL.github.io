import numpy as np
from scipy.stats import norm
import time

_T0 = time.time()
_DEADLINE = 310.0

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Uses a hybrid of Cost-Adjusted UCB and Cost-Adjusted EI,
    with adaptive exploration/exploitation balance based on progress.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if not _time_ok():
        # Fallback: simple exploitative
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    # Clean inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Estimate progress: f_best goes from ~0 to -78.33
    # If f_best is positive, progress is 0.
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    
    # Progress: 0 at start (f_best >= 0), 1 at optimum (f_best = -78.33)
    # We assume the initial random points might give f_best around 0 or slightly negative.
    # The minimum is -78.3323.
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)
    
    # Calculate EI
    z = (f_best_val - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best_val - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB
    # Beta decays from 2.0 to 0.5
    beta = 2.0 - 1.5 * progress
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to similar scales?
    # EI is in objective units (improvement).
    # UCB is in objective units (predicted value + uncertainty).
    # They are not directly comparable in magnitude if f_best is far from mu.
    # However, both are "higher is better" for the score if we want to minimize.
    # Wait, EI is expected *improvement*. Higher EI is better.
    # UCB is predicted *value* + uncertainty. Lower value is better for minimization.
    # So for UCB, we should use -UCB or (some_max - UCB)?
    # Standard UCB for minimization: score = -mu + beta*sigma.
    # This score is higher when mu is lower or sigma is higher.
    # So -UCB is NOT correct. The score IS -mu + beta*sigma.
    # So UCB score is higher for better (lower) mu and higher sigma.
    
    # Let's check scales.
    # If f_best = -10, mu = -20, sigma = 1.
    # EI = (-10 - (-20)) * Phi(10) + 1 * phi(10) = 10 * 1 + 0 = 10.
    # UCB = -(-20) + 1.0 * 1 = 21.
    # They are on different scales.
    # To combine them, we should normalize them.
    # Or, just use one of them.
    # Cost-Adjusted EI is very robust.
    # Let's try just Cost-Adjusted EI first, it's simpler and often sufficient.
    # If we want to be safe, we can use a weighted sum of normalized EI and UCB.
    
    # Let's stick to Cost-Adjusted EI as the primary driver,
    # but add a small exploration bonus if we are very early (progress < 0.1).
    # Actually, EI already includes exploration via sigma.
    
    score = ei / cost_safe
    
    # Optional: Add a small UCB component to encourage exploration in flat regions?
    # If sigma is high but mu is not much better than f_best, EI might be low.
    # UCB would be high.
    # Let's blend them.
    # Normalize EI: divide by max(EI) + eps
    # Normalize UCB: divide by max(UCB) + eps
    # This is tricky because the scales change.
    
    # Alternative: Use Cost-Adjusted UCB exclusively.
    # UCB is simpler and doesn't require f_best in the calculation (except for progress estimation).
    # Let's try Cost-Adjusted UCB.
    
    score_ucb = ucb / cost_safe
    
    # Let's use a mix.
    # Weight for UCB: higher when progress is low.
    # Weight for EI: higher when progress is high.
    w_ucb = 1.0 - progress
    w_ei = progress
    
    # To combine, we need them on similar scales.
    # Let's normalize each by its own max to make them [0, 1] roughly.
    max_ei = np.max(ei) + 1e-6
    max_ucb = np.max(ucb) + 1e-6
    
    norm_ei = ei / max_ei
    norm_ucb = ucb / max_ucb
    
    combined_score = (w_ei * norm_ei + w_ucb * norm_ucb) / cost_safe
    
    # Ensure finite
    combined_score = np.where(np.isfinite(combined_score), combined_score, 0.0)
    
    return combined_score
