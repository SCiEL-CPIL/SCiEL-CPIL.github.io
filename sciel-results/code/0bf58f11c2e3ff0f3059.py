"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Weierstrass 1D parameters (standard)
_W_FREQ = 10.0
_W_GAMMA = 0.5


def _weierstrass(x):
    """Weierstrass 1D function. x in [-0.5, 0.5]."""
    x = np.asarray(x, dtype=np.float64)
    val = np.zeros_like(x, dtype=np.float64)
    for k in range(30):
        val += (_W_GAMMA ** k) * np.cos(2.0 * np.pi * _W_FREQ ** k * (x + 0.5))
    # Standard Weierstrass is shifted so minimum is 0; the raw sum has min ~ sum(gamma^k) * cos(...)
    # Actually, the standard form w(x) = sum gamma^k cos(2 pi freq^k (x+0.5))
    # The minimum value of the raw sum is not 0. The task says "minimum value 0".
    # Usually the Weierstrass function is defined such that it's bounded.
    # Let's check: if we use the standard definition, the minimum is not 0.
    # However, the task says "minimizes weierstrass 1d ... minimum value 0".
    # This implies the function used by the evaluator is likely shifted or scaled.
    # But wait, the acquisition function only receives mu, sigma, f_best.
    # The evaluator handles the function. I do NOT need to implement the function for the acquisition.
    # The acquisition just returns scores.
    # So I don't need to know the exact function value, just the relative improvement.
    # However, knowing the landscape might help design a better acquisition.
    # The Weierstrass function is continuous but nowhere differentiable. It has many local minima.
    # Standard EI might struggle with multimodality if the GP hasn't seen enough.
    # A common improvement is to use a "local" acquisition or a mixture.
    # But with only 2000 points and 25 evaluations, the GP is quite sparse.
    # Let's stick to a robust EI variant.
    # One issue with standard EI is that it can be too greedy.
    # Let's try Expected Hypervolume Improvement or a simple EI with a small exploration term.
    # Actually, for 1D, EI is usually quite good.
    # The problem might be that sigma is small in flat regions, causing numerical issues or overconfidence.
    # Let's ensure sigma is bounded away from 0.
    
    # Let's try a different acquisition: UCB (Upper Confidence Bound) with minimization.
    # For minimization, we want to maximize (f_best - mu) + kappa * sigma? No.
    # We want to find low values.
    # UCB for minimization: score = -(mu - kappa * sigma) = -mu + kappa * sigma.
    # Or: score = (f_best - mu) / (sigma + eps) * sigma? No.
    # Standard UCB for minimization: maximize -mu + beta * sigma.
    # Let's compare EI and UCB.
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.
    # This is always >= 0.
    # UCB: -mu + beta * sigma.
    # UCB can be negative if mu is very low. But we want higher scores to be better.
    # If mu is very low (good), -mu is large positive.
    # If sigma is high, UCB adds exploration.
    # EI is theoretically grounded. Let's stick to EI but maybe tweak it.
    
    # One known issue: if f_best is not updated frequently or is far from current mu, EI can be misleading.
    # But f_best is the global best so far.
    
    # Let's try a "Local" EI or a combination.
    # Actually, for 1D Weierstrass, the function is very oscillatory at high frequencies.
    # With length scale 0.2, the GP might not capture the high-frequency oscillations well.
    # This means the GP posterior might be overconfident in areas it hasn't sampled densely.
    # This is a classic problem.
    # A solution is to use a shorter length scale, but we can't change the GP.
    # We can only change the acquisition.
    # Perhaps a more aggressive exploration strategy is needed.
    # Let's try UCB with a tuned beta.
    # Or, a mixture of EI and a random exploration term.
    
    # Let's look at the "fresh uniform candidates" part.
    # The evaluator draws 2000 fresh uniform candidates each step.
    # This means the points are not adaptive in their placement, only in which one is chosen.
    # So the GP is updated with the chosen point.
    # The 2000 candidates are random.
    # This is a "random sampling" BO approach.
    # In this case, EI is often effective because it picks the most promising random point.
    
    # Let's try to improve the numerical stability of EI.
    # And maybe add a small constant to sigma to prevent division by zero or extreme values.
    
    # Another idea: Use the "Probability of Improvement" (PI) or "Expected Improvement" (EI) with a small epsilon.
    # EI is generally better than PI.
    
    # Let's try a different approach: "Knowledge Gradient" is complex.
    # How about "Expected Improvement" with a local mean?
    # No, we don't have coordinates.
    
    # Let's try "UCB" with a dynamic beta.
    # beta = 2 * log(t) / t? No, t is not passed.
    # We can estimate t from the number of evaluations? No, we don't have it.
    # We only have mu, sigma, f_best.
    # So we must use a static or simple heuristic.
    
    # Let's try a mixture of EI and UCB.
    # score = alpha * EI + (1 - alpha) * UCB
    # But UCB scale is different.
    
    # Let's stick to EI but ensure robustness.
    # The current code:
    # z = (f_best - mu) / (sigma + 1e-9)
    # return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # This is standard EI.
    # Why might it fail?
    # 1. If sigma is very small, z is large, cdf(z) ~ 1, pdf(z) ~ 0. EI ~ f_best - mu.
    #    This favors points where mu < f_best.
    # 2. If sigma is large, z is small, EI ~ (f_best - mu) * 0.5 + sigma * 0.4.
    #    This favors points with high sigma (uncertainty).
    
    # One issue: If the GP is overconfident (sigma too small) in unexplored regions, EI will be low there.
    # But the GP is fitted to the data. If the data is sparse, sigma should be large in unexplored regions.
    # Unless the length scale is too long, making the GP interpolate smoothly.
    # With length scale 0.2 in [0,1] (which is 0.1 in [-0.5,0.5]), the GP is quite wiggly.
    # It should capture some structure.
    
    # Let's try to add a small exploration bonus to EI.
    # score = EI + lambda * sigma
    # This encourages exploring high-uncertainty regions even if EI is low.
    # Let's try lambda = 0.1 * (f_best - min(mu)) / max(sigma)? No, scale dependent.
    # Let's try a fixed small lambda, e.g., 0.01.
    # But we don't know the scale of the function.
    # The Weierstrass function values are O(1).
    # So lambda = 0.01 might be reasonable.
    
    # Alternatively, use "Expected Improvement" with a "lookahead" or "local" term? No.
    
    # Let's try "UCB" with beta = 2.0.
    # score = -mu + 2.0 * sigma
    # This is simple and often works well.
    # Let's compare EI and UCB.
    # EI is more focused on improvement. UCB is more exploratory.
    # For multimodal functions, UCB might be better.
    
    # Let's try a hybrid:
    # score = EI + 0.5 * sigma
    # This adds a moderate exploration bonus.
    
    # Let's also consider the "fresh uniform candidates" aspect.
    # Since the candidates are random, the set of mu/sigma values changes every step.
    # The acquisition must be robust to this.
    
    # I will implement a hybrid EI + exploration bonus.
    # I will also add a deadline guard as requested.
    
    # Deadline guard:
    # The function is called 25 times (30 - 5).
    # Each call must be fast.
    # EI computation is O(N) where N=2000. This is very fast (<1ms).
    # So deadline is not a concern for the acquisition itself.
    # But the directive says to keep an explicit deadline guard.
    # I will add a check, but it will likely never trigger.
    
    # Let's refine the hybrid.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Exploration = sigma
    # Score = EI + beta * sigma
    # What beta?
    # If beta is too high, we explore too much.
    # If beta is too low, we exploit too much.
    # Let's try beta = 0.5.
    
    # Another idea: "Expected Improvement" with a "local" sigma? No.
    
    # Let's try "Probability of Improvement" (PI) with a small epsilon?
    # PI = Phi(z)
    # This is less sensitive to the magnitude of improvement.
    # EI is generally better.
    
    # I'll go with EI + 0.5 * sigma.
    
    # Wait, the task says "minimum value 0".
    # If the function is Weierstrass, the minimum is 0.
    # f_best starts high and decreases.
    # As f_best approaches 0, EI becomes small.
    # The exploration term sigma might dominate, leading to random search near the end.
    # This might be good or bad.
    # If we are near the global minimum, we want to exploit.
    # So maybe the exploration term should decay?
    # But we don't know the step count.
    # We can use f_best as a proxy.
    # If f_best is close to 0, reduce exploration.
    # beta = 0.5 * max(0, 1 - f_best) ?
    # If f_best is 1, beta=0. If f_best is 0, beta=0.5.
    # This is the opposite of what I want. I want less exploration when f_best is good.
    # So beta = 0.5 * f_best ?
    # If f_best is 1, beta=0.5. If f_best is 0, beta=0.
    # This seems reasonable.
    
    # Let's try:
    # z = (f_best - mu) / (sigma + 1e-9)
    # ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # beta = 0.5 * np.clip(f_best, 0, 1)  # Assuming f_best is in [0,1]
    # score = ei + beta * sigma
    
    # But what if f_best is not in [0,1]? The Weierstrass function values are positive.
    # The maximum value is sum(gamma^k) = 1/(1-gamma) = 2.
    # So f_best is in [0, 2].
    # Let's use beta = 0.25 * f_best.
    
    # Let's also consider that sigma might be very small.
    # I will clip sigma to a minimum value to avoid numerical issues.
    # sigma = np.maximum(sigma, 1e-6)
    
    # Let's write the code.
    
    # One more thought: The "fresh uniform candidates" means that the GP is not updated with all 2000 points, only the best one.
    # This is a "best of N" strategy.
    # EI is well-suited for this.
    
    # I will implement the hybrid EI + exploration.
    
    pass  # Placeholder for actual code

# EVOLVE-BLOCK-END

import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 300.0  # 300s limit, use 290s for safety

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _START_TIME > _TIMEOUT - 10:
        # Return a safe default (e.g., EI) to avoid timeout
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure sigma is positive and not too small
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus
    # Scale beta by f_best to reduce exploration as we get better
    # Assume f_best is in [0, 2] for Weierstrass
    beta = 0.25 * np.clip(f_best, 0, 2)
    
    score = ei + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
