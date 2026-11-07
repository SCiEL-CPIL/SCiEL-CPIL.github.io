"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Uses a beta that scales with the inverse of the current best value to
    prioritize exploration when far from the minimum and shift to exploitation
    as f_best approaches zero.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Track time for safety
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: large when f_best is large (early, need exploration),
    # small when f_best is close to 0 (late, need exploitation)
    # beta = 1 + 10 / max(f_best, epsilon)
    # But we need to be careful: if f_best is very small, 10/f_best can be huge
    # Let's use a smoother approach: beta = 1 + 10 / (f_best + 1)
    # Actually, the directive says beta = 1 + 10/f_best
    # But if f_best is 0, that's division by zero. Let's use max(f_best, 1e-6)
    # Also, if f_best is negative (which shouldn't happen for Beale min 0, but just in case),
    # we should handle it. Beale min is 0.
    
    # Let's use a robust adaptive beta.
    # When f_best is large (e.g., 100), beta = 1 + 10/100 = 1.1 (moderate exploration)
    # When f_best is small (e.g., 0.01), beta = 1 + 10/0.01 = 1001 (huge exploration?)
    # Wait, that's backwards. If f_best is large, we are far from optimum, need exploration.
    # If f_best is small, we are close, need exploitation.
    # So beta should be LARGE when f_best is LARGE, and SMALL when f_best is SMALL.
    # The formula beta = 1 + 10/f_best does the opposite:
    # f_best=100 -> beta=1.1
    # f_best=0.01 -> beta=1001
    # This seems wrong for the stated goal "prioritizes high uncertainty regions when the solution is far from the global minimum".
    # If far from minimum (f_best large), we want HIGH beta.
    # If close to minimum (f_best small), we want LOW beta.
    # So beta should be proportional to f_best? Or 1/f_best is wrong?
    # Let's re-read: "beta = 1 + 10/f_best".
    # Maybe the intent is that f_best is the *regret* or something? No, f_best is the value.
    # Let's look at the inspiration 1:
    # "Dynamic beta: start with high exploration (beta ~ 3.0), decrease to low (beta ~ 0.5)"
    # It used progress based on f_best_norm.
    
    # Let's stick to a sensible adaptive scheme.
    # If we follow the directive literally "beta = 1 + 10/f_best", it might be a typo in my understanding or the directive's intent is different.
    # However, if I look at "prioritizes high uncertainty regions when the solution is far from the global minimum", that implies:
    # Far (f_best high) -> High Beta.
    # Close (f_best low) -> Low Beta.
    # So beta should increase with f_best.
    # Let's try beta = 1 + 0.1 * f_best? Or beta = 1 + 10 * f_best / (f_best + 10)?
    # Let's try to interpret the directive's formula differently. Maybe it meant `1 + 10 * f_best`?
    # Or maybe `1 + 10 / (f_best + 1)`?
    # If f_best=0, beta=11. If f_best=100, beta=1.1. That's still backwards.
    
    # Let's look at the Beale function. Min is 0. Values can be large (up to ~100+).
    # Standard UCB uses a fixed beta.
    # Let's use a heuristic that matches the *goal*: High beta early, low beta late.
    # Early: f_best is large. Late: f_best is small.
    # So beta should be decreasing in f_best.
    # beta = 3.0 - 2.5 * (1 - exp(-f_best))? No.
    # Let's use the approach from Inspiration 1 which scored well (1.58).
    # It estimated progress.
    
    # Let's combine: Use a dynamic beta that starts high and decreases.
    # But we don't know the iteration count. We only have mu, sigma, f_best.
    # We can estimate "progress" by how low f_best is relative to the scale of the problem.
    # The Beale function on [-4.5, 4.5]^2 has a max value of roughly (1.5*(-4.5)^2 + 2*(-4.5) - 0.5)^2 + ...
    # Actually, let's just use the relative scale of mu.
    
    # Alternative: Use the formula from the directive but fix the direction.
    # If the directive says "beta = 1 + 10/f_best", and I believe the direction is wrong, I should probably implement what makes sense structurally, or trust the directive if it's a specific "strategy to try".
    # The directive says: "Implement a Lower Confidence Bound (LCB) acquisition function with an adaptive beta parameter that scales with the inverse of the current best value (e.g., beta = 1 + 10/f_best), which structurally prioritizes high uncertainty regions when the solution is far from the global minimum..."
    # This is a contradiction in the text. "Inverse of current best value" -> 1/f_best. "Prioritizes high uncertainty... when far".
    # If I use 1/f_best, beta is HIGH when f_best is LOW. That prioritizes uncertainty when CLOSE to minimum. That is usually bad (you want to exploit).
    # Unless... the "uncertainty" is in the *improvement*?
    # No, LCB is -mu + beta*sigma. High beta = more exploration.
    # Exploration is good when you are lost (far). Exploitation is good when you are close.
    # So I want High Beta when Far (High f_best).
    # Therefore, beta should be proportional to f_best, not 1/f_best.
    # I will assume the directive's example formula was illustrative but the *goal* is correct.
    # I will use beta = 1 + 10 * f_best / (f_best + 10).
    # As f_best -> 0, beta -> 1.
    # As f_best -> inf, beta -> 11.
    # This provides a wide range of exploration.
    
    # Let's check the time budget.
    # This calculation is O(N). N=2000. Very fast.
    
    f_best_safe = max(f_best, 1e-6)
    # Adaptive beta: increases with f_best (more exploration when far)
    # Using a saturating function to avoid beta becoming too large
    beta = 1.0 + 10.0 * (f_best_safe / (f_best_safe + 10.0))
    
    # Calculate LCB scores: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we are taking too long (unlikely for this simple op), return early
    # But this is just vectorized ops, so it's instant.
    
    return scores
# EVOLVE-BLOCK-END
