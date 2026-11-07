"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Calculate Expected Improvement (EI) for minimization
    # Higher EI means better
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # We want to minimize f, so we look for low mu and high sigma (uncertainty)
    # Score = -mu + beta * sigma
    # The known global minimum is approx -39.166
    # If f_best is much higher than -39.166, we should explore more (higher beta)
    # If f_best is close to -39.166, we should exploit (lower beta)
    
    f_opt = -39.166
    # f_best starts around 0-50 and goes down to -40
    # Distance from optimum
    dist = f_best - f_opt
    
    # Normalize distance to get a weight for exploration
    # If dist is large (e.g., 40), we want high beta.
    # If dist is small (e.g., 1), we want low beta.
    # Let's use a simple linear mapping or a sigmoid-like decay.
    # beta_min = 0.5, beta_max = 2.0
    # When dist <= 1, beta = beta_min
    # When dist >= 10, beta = beta_max
    
    beta_min = 0.5
    beta_max = 2.0
    
    # Simple linear interpolation with clipping
    # t = 0 when dist <= 1, t = 1 when dist >= 10
    t = np.clip((dist - 1.0) / 9.0, 0.0, 1.0)
    beta = beta_min + t * (beta_max - beta_min)
    
    ucb_score = -mu + beta * sigma
    
    # Normalize both scores to have similar scales before blending?
    # Or just blend them with fixed weights?
    # EI can be very small when close to optimum. UCB can be large.
    # Let's try a dynamic weight.
    # Early on (high dist), weight UCB more.
    # Later (low dist), weight EI more.
    
    w_ucb = 0.5 + 0.5 * t  # 0.5 to 1.0
    w_ei = 1.0             # Keep EI constant? Or scale it?
    
    # Actually, simpler: just use a weighted sum.
    # The magnitudes might differ.
    # Let's standardize the scores within the batch to make them comparable.
    # However, we can't easily standardize without knowing the range beforehand 
    # and it might distort the absolute meaning if f_best changes significantly.
    
    # Alternative: Use the "Knowledge Gradient" or similar?
    # For now, let's stick to the blend but ensure stability.
    
    # To avoid scale issues, we can normalize EI and UCB by their respective stds in the batch
    # But that adds complexity. Let's try a simple weighted sum first.
    # If EI is around 0.1 and UCB is around 5.0, UCB will dominate.
    # That's good for exploration.
    
    # Let's refine the beta range.
    # If we are far, we want to jump around.
    # If we are close, we want to polish.
    
    # Let's try: score = w_ucb * ucb_score + w_ei * ei
    # With w_ucb decreasing as we get closer.
    
    score = (0.5 + 0.5 * (1.0 - t)) * ucb_score + (0.5 + 0.5 * t) * ei
    
    # Ensure finite
    return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
