"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using a dynamic blend of Expected Improvement (EI) and
    Probability of Improvement (PI) with an adaptive exploration component.
    
    Strategy:
    - Use EI as the primary exploitation signal.
    - Add a small, adaptive exploration bonus based on sigma to encourage
      sampling in uncertain regions, but scale it down as the search progresses
      (estimated via f_best improvement).
    - Do NOT divide by cost, as the shifted minimum can be anywhere in the domain,
      and cost-based biasing can lead to missing the global optimum.
    - Use a hybrid score: EI + kappa * sigma * (1 - normalized_improvement)
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: ensure finite inputs and handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Probability of Improvement (PI)
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Adaptive exploration: 
    # We want to explore more when f_best is still high (early in search)
    # and exploit more when f_best is low (later in search).
    # Normalize f_best to [0, 1] range? We don't know the max possible f_best.
    # Instead, use a simple heuristic: if f_best is still large, explore more.
    # For Rosenbrock, f(0,0)=0, and max f on [-2,2]^2 is around 2000+.
    # Let's use a soft decay based on f_best.
    # If f_best > 10, we are likely far from the minimum, so explore more.
    # If f_best < 1, we are close, so exploit more.
    
    # A simple adaptive kappa:
    # Start with high exploration, decay as f_best decreases.
    # kappa = 1.0 / (1.0 + f_best)
    # This gives kappa ~1.0 when f_best is small, and kappa ~0 when f_best is large.
    # Wait, we want the opposite: more exploration when f_best is large.
    # kappa = 1.0 / (1.0 + 1.0/f_best) ? No.
    # Let's use: kappa = min(1.0, f_best / 10.0)
    # If f_best = 0, kappa = 0 (pure exploitation).
    # If f_best = 10, kappa = 1 (max exploration).
    # If f_best = 100, kappa = 1 (max exploration).
    
    kappa = np.clip(f_best / 10.0, 0.0, 1.0)
    
    # Exploration term: sigma * kappa
    # But we also want to penalize points with very high mu (likely bad regions).
    # EI already does this implicitly.
    
    # Combine EI and exploration
    # score = ei + kappa * sigma_safe
    # However, EI can be negative if mu > f_best. We want to avoid those.
    # Let's ensure the score is non-negative by adding a small base or clipping.
    # Actually, EI is always >= 0 for the standard definition? 
    # No, if mu > f_best, (f_best - mu) is negative, Phi(z) is small, sigma*phi(z) is positive.
    # It can be slightly negative. Let's clip to 0.
    ei = np.maximum(ei, 0.0)
    
    score = ei + kappa * sigma_safe
    
    # Add a very small PI term to break ties and encourage improvement
    score += 0.1 * pi
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
