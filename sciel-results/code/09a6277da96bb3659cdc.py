"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement
    and a Thompson-Sampling-inspired posterior sample, weighted by cost.
    """
    # Record start time for deadline guard
    t_start = time.time()
    
    # Convert inputs to float64 arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma to avoid division issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # --- Component 1: Expected Improvement (EI) ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Component 2: Thompson Sampling Sample ---
    # Draw a single sample from the posterior predictive distribution.
    # theta = mu + sigma * z_sample
    # Using a fixed random seed for reproducibility within a single call is not possible 
    # across calls without state, but we can use a deterministic pseudo-random offset 
    # or just draw a random sample. Since the evaluator runs the loop, 
    # we should draw a random sample each time to explore the posterior.
    # However, for a static function, we just draw one sample.
    z_sample = np.random.randn(len(mu))
    theta = mu + sigma * z_sample
    
    # Normalized regret for Thompson Sampling: (theta - f_best) / sqrt(cost)
    # We want to MINIMIZE this, so the score should be negative of this or similar.
    # Let's define ts_score = - (theta - f_best) / sqrt(cost)
    # If theta < f_best, score is positive (good). If theta > f_best, score is negative (bad).
    ts_score = -(theta - f_best) / np.sqrt(cost)
    
    # --- Combine Components ---
    # Normalize EI and TS score to comparable scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ts_min, ts_max = np.min(ts_score), np.max(ts_score)
    
    ei_range = ei_max - ei_min
    ts_range = ts_max - ts_min
    
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    if ts_range < 1e-10:
        ts_norm = np.ones_like(ts_score)
    else:
        ts_norm = (ts_score - ts_min) / ts_range
    
    # Weighted combination: 60% EI, 40% TS
    # EI provides a smooth, deterministic improvement estimate.
    # TS provides a stochastic exploration signal that can jump to uncertain regions.
    combined = 0.6 * ei_norm + 0.4 * ts_norm
    
    # Cost-aware: divide by cost (or sqrt cost? The prompt says "cost-aware", 
    # and the seed used /cost. Let's stick with /cost for consistency with the 
    # strongest inspiration which used /cost on the normalized scores).
    # Actually, the TS score already has 1/sqrt(cost). 
    # If we divide the combined normalized score by cost again, we are double-penalizing cost for the TS part.
    # Let's re-evaluate.
    # EI is an absolute improvement. Dividing by cost gives improvement per unit cost.
    # TS score is already normalized by sqrt(cost).
    # Maybe it's better to combine the raw scores differently?
    
    # Alternative:
    # score_ei = ei / cost
    # score_ts = -(theta - f_best) / sqrt(cost)
    # But these have different units/scales.
    
    # Let's stick to the normalized combination and then apply a global cost penalty.
    # The normalization removes the scale difference.
    # Then dividing by cost penalizes expensive points.
    scores = combined / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we've spent too much time, return immediately.
    # The computation above is O(N) and very fast, so this is likely unnecessary,
    # but included as per directive.
    t_end = time.time()
    if t_end - t_start > 5.0:  # 5 second margin
         pass # Nothing to do, we already returned the computed scores.
    
    return scores
# EVOLVE-BLOCK-END
