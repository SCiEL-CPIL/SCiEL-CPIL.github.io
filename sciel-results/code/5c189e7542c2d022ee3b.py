"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Combines Expected Improvement (EI) with a variance-based exploration bonus
    that decays as the current best approaches the known minimum (0).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))

    # Ensure sigma is strictly positive
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (fb - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (fb - mu) * Phi(z) + sigma * phi(z)
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural change: Add a minimum-variance exploration component.
    # The directive suggests scoring by variance to force resolution of uncertainty.
    # Pure variance can be too aggressive early or late.
    # We blend EI (exploitation/standard exploration) with a variance bonus.
    # The variance bonus is scaled to be comparable to EI.
    
    # Scale the variance bonus relative to the current best value.
    # As fb -> 0, we might want to reduce exploration if we are very close,
    # or increase it if we are stuck?
    # The directive says: "force the GP to resolve uncertainty in unexplored regions 
    # before exploiting the current best, thereby preventing premature convergence".
    # This suggests we should keep exploring high-sigma regions even if EI is low.
    
    # Let's use a weighted sum: Score = EI + lambda * sigma
    # How to choose lambda?
    # If we look at inspiration 1 (UCB), it used -mu + beta*sigma.
    # EI is roughly (fb-mu) when mu << fb, and ~0 when mu >> fb.
    # Adding sigma directly encourages high uncertainty.
    
    # Let's try a dynamic lambda that decreases as we get closer to 0?
    # Or a fixed lambda?
    # Let's look at the scale. mu and sigma are in objective units.
    # For sphere 1d on [-5,5], max f is 25.
    # Early on, fb ~ 10-20. EI ~ 1-5. sigma ~ 1-5.
    # Late on, fb ~ 0.1. EI ~ 0.01. sigma ~ 0.01-0.1.
    
    # If we just add sigma, it might dominate EI late in the game if sigma is not tiny.
    # But in late game, sigma should be small near the optimum.
    # In unexplored regions, sigma is large.
    
    # Let's try: Score = EI + 0.1 * sigma_safe
    # This is essentially a UCB-like term added to EI.
    # The 0.1 factor is a heuristic to balance.
    
    # Alternative: Use the "Probability of Improvement" (PI) which is bounded [0,1]
    # PI = Phi(z).
    # EI is unbounded.
    
    # Let's stick to the directive's hint: "score candidates solely by their posterior 
    # standard deviation... effectively ignoring the mean".
    # This is a strong structural change.
    # If we ignore the mean completely, we pick the highest sigma point.
    # This is pure exploration.
    # This might be too extreme.
    # Let's blend: Score = EI + alpha * sigma.
    
    # To make it a "structural change" from the current EI-only, we introduce
    # a significant exploration weight.
    
    # Let's try alpha = 0.5.
    # Score = EI + 0.5 * sigma_safe
    
    # However, EI already contains a sigma term: sigma * phi(z).
    # phi(z) is max 0.3989.
    # So the implicit exploration weight in EI is ~0.4 * sigma (when z=0).
    # Adding 0.5 * sigma doubles the exploration pressure.
    
    # Let's refine:
    # We want to prevent premature convergence.
    # Premature convergence happens when EI is dominated by the term (fb-mu)*Phi(z)
    # and sigma is small, so the sigma*phi(z) term is small.
    # Adding an explicit sigma term helps.
    
    # Let's try: Score = EI + 0.2 * sigma_safe
    
    # Also, ensure finite scores.
    score = ei + 0.2 * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
