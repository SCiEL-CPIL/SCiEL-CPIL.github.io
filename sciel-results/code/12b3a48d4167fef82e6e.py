"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are close to the limit, return a safe, fast result
    # This is a heuristic; the main protection is efficient computation.
    if time.time() - _start_time > 330.0 - _DEADLINE_MARGIN:
        # Return a simple, fast score to avoid hanging in complex calculations
        # Using a simple ratio as fallback
        return (f_best - mu) / (cost + 1e-9)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration Term
    # The knowledge suggests an "Adaptive UCB/EI blend with tanh-scaled beta".
    # We interpret this as adding an exploration component that is scaled by uncertainty
    # and modulated by the distance to the best point.
    
    # A common effective strategy is to blend EI with a UCB-like term.
    # For minimization, a UCB-like term that encourages exploration is proportional to sigma.
    # We use a tanh scaling to make the exploration adaptive.
    # If the point is far from the best (large z), we might explore more?
    # Or if the point is close, we exploit?
    
    # Let's try a blend: Score = (EI + beta * sigma * tanh(z)) / cost
    # If z is large positive (mu << f_best), tanh(z) ~ 1, so we add beta * sigma.
    # If z is large negative (mu >> f_best), tanh(z) ~ -1, so we subtract beta * sigma.
    # This effectively boosts points with high uncertainty that are potentially good.
    
    beta = 1.0  # Tunable parameter, 1.0 is a reasonable start
    exploration_term = beta * sigma_safe * np.tanh(z)
    
    # Combine EI and Exploration
    # We want higher score to be better.
    # EI is higher when improvement is expected.
    # Exploration term is higher when sigma is high and z is positive (potentially good).
    
    combined_score = ei + exploration_term
    
    # Cost-aware normalization
    # Dividing by cost penalizes expensive points.
    scores = combined_score / cost_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
