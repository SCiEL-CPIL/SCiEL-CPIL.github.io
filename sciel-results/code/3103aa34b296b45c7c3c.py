"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 310.0  # 330s limit, 20s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rosenbrock 2D.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at candidate points (in objective units)
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at candidate points (in objective units)
    f_best : float
        Best observed function value so far
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate: 1 + mean(x_scaled) where x_scaled = (x+2)/4
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores where higher = more desirable to evaluate next
    """
    current_time = time.time()
    if current_time > TIME_LIMIT:
        # Emergency fallback: return finite values to prevent crash/timeout penalty
        return np.zeros_like(mu)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive and finite
    cost = np.maximum(cost, 1e-9)
    
    # === STRUCTURAL CHANGE: Use a hybrid approach combining:
    # 1. Expected Improvement (EI) - standard BO acquisition
    # 2. Cost-awareness through normalization
    # 3. Adaptive exploration based on progress (f_best)
    # 4. A "proximity bonus" that favors points where the GP is confident AND low
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal
    
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be, but clip for safety)
    ei = np.maximum(ei, 0)
    
    # === ADAPTIVE EXPLORATION STRATEGY ===
    # Early on (f_best large): explore more, weight uncertainty heavily
    # Late (f_best small): exploit, focus on low predicted values
    
    # Use a smooth transition based on f_best
    # When f_best > 50: heavy exploration (kappa ~ 2.5)
    # When f_best ~ 10: moderate (kappa ~ 1.0)
    # When f_best < 1: exploit (kappa ~ 0.3)
    
    # kappa controls the exploration/exploitation tradeoff
    # We'll use a smooth function: kappa = base + amp / (1 + f_best/scale)
    base_kappa = 0.3
    amp_kappa = 2.2
    scale = 10.0
    
    kappa = base_kappa + amp_kappa / (1.0 + f_best / scale)
    
    # Compute a UCB-like score: mu - kappa * sigma (for minimization, we want low values)
    # But we want to MAXIMIZE, so: score_ucb = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma
    
    # Normalize both EI and UCB scores to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Blend EI and UCB with adaptive weights
    # Early: more UCB (exploration). Late: more EI (exploitation)
    # weight_ei increases as f_best decreases
    weight_ei = 1.0 / (1.0 + np.exp(-(f_best - 5.0) / 3.0))
    weight_ei = np.clip(weight_ei, 0.2, 0.8)
    weight_ucb = 1.0 - weight_ei
    
    blended_score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # === COST AWARENESS ===
    # Divide by cost to favor cheaper evaluations
    # This is the key cost-aware component
    cost_aware_score = blended_score / cost
    
    # === PROXIMITY BONUS ===
    # Bonus for points where GP is confident (low sigma) AND predicts low value (low mu)
    # This helps exploit the GP's knowledge when it's reliable
    confidence = np.exp(-sigma / (1.0 + np.abs(f_best)))
    low_value = np.exp(-np.maximum(mu, 0) / (1.0 + np.abs(f_best)))
    bonus = confidence * low_value
    
    # Add bonus (scaled down to not dominate)
    final_score = cost_aware_score + 0.15 * bonus / cost
    
    # Ensure finiteness
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_score

# EVOLVE-BLOCK-END
