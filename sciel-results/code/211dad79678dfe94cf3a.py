"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Returns 2000 scores, higher meaning evaluate next.
    """
    global _start_time
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a simple, fast acquisition to avoid timeout
        # Use raw EI as fallback
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate adaptive exploration bonus
    # We want to explore more early on.
    # Use a progress metric based on f_best relative to the domain max (25 for [-5,5]^2? No, 1D sphere is x^2, max is 25).
    # f_best starts around ~12.5 (uniform mean of x^2 on [-5,5] is 8.33? Variance is 8.33. Mean is 8.33).
    # Let's use a simple decay based on the number of evaluations or f_best.
    # Since we don't have eval count, use f_best.
    # As f_best -> 0, we should stop exploring.
    
    # Estimate progress: 1.0 when f_best is high, 0.0 when f_best is low.
    # f_best is in [0, 25].
    # Let's map f_best to [0, 1] progress where 1 is "start" and 0 is "end".
    # If f_best > 10, we are early. If f_best < 1, we are late.
    # progress = max(0, min(1, (f_best - 0.5) / 10.0))
    
    # Alternatively, use the uncertainty level.
    # If sigma is high everywhere, we are early.
    sigma_mean = np.mean(sigma)
    # If sigma_mean is high, explore more.
    # Let's set beta (exploration weight) based on sigma_mean.
    # beta = 1.0 * (sigma_mean / (sigma_mean + 1.0))
    # This makes beta ~1 when sigma is large, and ~0 when sigma is small.
    
    # Let's try a fixed structure with adaptive beta.
    # score = EI + beta * sigma
    # This encourages high uncertainty.
    
    # To avoid over-exploration late game, decay beta as f_best improves.
    # If f_best is close to 0, beta should be small.
    # beta = 0.5 * (1.0 - min(f_best / 5.0, 1.0))
    
    # Let's combine these insights.
    # Early: high beta. Late: low beta.
    # f_best starts ~8. Ends ~0.
    # progress = 1.0 - min(f_best / 8.0, 1.0)  # 0 at start, 1 at end (roughly)
    # Actually, let's just use f_best directly.
    
    # If f_best is 10, we want high exploration.
    # If f_best is 0.1, we want low exploration.
    # beta = 0.8 * (1.0 - min(f_best / 10.0, 1.0))
    
    # Let's use a simpler, robust heuristic:
    # The "low beta" finding for UCB suggests that exploitation is key.
    # So we should keep beta LOW.
    # Let's try beta = 0.5 * (1.0 - min(f_best / 5.0, 1.0))
    
    # Actually, let's look at the "low beta" finding again.
    # "UCB with low beta favors exploitation... converging faster".
    # This implies we should use a small kappa in UCB.
    # In our `score = EI + beta * sigma`, `beta` acts like `kappa`.
    # So we should use a small `beta`.
    
    # Let's try a very small beta, or even just EI.
    # But EI alone scored 1.0.
    # The current program (with broken UCB) scored 2.1.
    # If we fix the UCB direction and make it small, we might get > 2.1.
    
    # Let's use: score = EI + 0.2 * sigma
    # This is a mild exploration bonus.
    
    # Adaptive:
    # If f_best is very low, reduce beta.
    # If f_best is high, increase beta.
    
    # Let's use a sigmoid-like transition.
    # beta = 0.5 / (1.0 + np.exp(f_best - 2.0))
    # If f_best < 2, beta is small. If f_best > 2, beta is larger.
    
    # Let's try this.
    beta = 0.5 / (1.0 + np.exp(f_best - 2.0))
    
    # Add a small constant to ensure some exploration always?
    # No, let's stick to the adaptive one.
    
    scores = ei + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
