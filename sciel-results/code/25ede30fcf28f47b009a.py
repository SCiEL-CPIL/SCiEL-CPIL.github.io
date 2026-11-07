import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = 0.5  # 0.5 second limit per call, plenty for N=2000
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive LCB (Lower Confidence Bound) for minimization
    # We want to select the point with the lowest predicted value minus an uncertainty term.
    # Since the task requires returning scores where "higher meaning evaluate next",
    # we maximize the negative of the LCB value: Score = - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Strategy:
    # When f_best is large (early in optimization, far from optimum), we want high exploration (high beta).
    # When f_best is small (late in optimization, close to optimum), we want high exploitation (low beta).
    
    # The strategy suggests beta = 1 + 10/f_best.
    # However, we must be careful if f_best is very small or zero to avoid division by zero or huge beta.
    # Also, if f_best is very large, beta becomes ~1, which might be too conservative for early exploration.
    
    # Let's refine the adaptive beta.
    # A common heuristic is to scale beta based on the number of iterations or the current best value.
    # Since we don't have iteration count, we use f_best.
    # If f_best is close to the global minimum (0), we want to stop exploring.
    # If f_best is large, we want to explore.
    
    # Let's try a smooth transition.
    # beta = 1.0 + 5.0 * (1.0 / (1.0 + f_best))
    # If f_best = 0, beta = 1 + 5 = 6 (High exploration? Wait, if f_best is 0, we are done. We should exploit).
    # Actually, if f_best is 0, we have found the global min. We should just pick the point with lowest mu.
    # So beta should be small when f_best is small.
    
    # Let's reverse the logic:
    # High f_best -> High Beta (Explore)
    # Low f_best -> Low Beta (Exploit)
    
    # beta = 1.0 + 4.0 * (f_best / (f_best + 1.0))
    # If f_best -> 0, beta -> 1.0
    # If f_best -> inf, beta -> 5.0
    
    # This seems reasonable.
    # Let's check the inspirations.
    # Inspiration 1: beta = 5.0 * (f_best / (f_best + 1.0)) + 1.0
    #   f_best=0 -> beta=1.0
    #   f_best=1 -> beta=3.5
    #   f_best=10 -> beta=4.54
    #   f_best=inf -> beta=6.0
    
    # Inspiration 3: beta = 1.5 + 2.5 * (f_best / (f_best + 1.0))
    #   f_best=0 -> beta=1.5
    #   f_best=1 -> beta=2.75
    #   f_best=inf -> beta=4.0
    
    # Current program: beta = 5.0 (fixed)
    #   This scores 6.79, which is quite good.
    #   The inspirations score ~3.7, which is worse.
    #   Why? Maybe the adaptive beta was too low in the middle range?
    #   Or maybe the fixed beta=5.0 was just well-tuned for the Beale function's landscape.
    
    # The directive says: "Implement a Lower Confidence Bound (LCB) acquisition function with an adaptive beta parameter that scales with the inverse of the current best value (e.g., beta = 1 + 10/f_best)..."
    # Wait, the directive text in the prompt is: "Strategy to try in this change: Implement a Lower Confidence Bound (LCB) acquisition function with an adaptive beta parameter that scales with the inverse of the current best value (e.g., beta = 1 + 10/f_best), which structurally prioritizes high uncertainty regions when the solution is far from the global minimum to escape local basins, while tightening the confidence bound as f_best approaches zero to focus exploitation on the"
    
    # Let's follow the directive's specific formula: beta = 1 + 10 / f_best.
    # But we need to handle f_best near 0.
    # If f_best is very small, 10/f_best is huge. That would mean HUGE exploration when we are close to the optimum? That seems counter-intuitive.
    # "tightening the confidence bound as f_best approaches zero" -> This implies beta should DECREASE as f_best approaches zero.
    # The formula beta = 1 + 10/f_best INCREASES as f_best approaches zero.
    # There is a contradiction in the directive text description vs the formula.
    # "prioritizes high uncertainty regions when the solution is far from the global minimum" -> f_best is large.
    # "tightening ... as f_best approaches zero" -> f_best is small.
    
    # If f_best is large (far from min), we want high beta.
    # If f_best is small (close to min), we want low beta.
    
    # Formula: beta = 1 + 10/f_best.
    # If f_best = 100, beta = 1.1. (Low exploration when far?)
    # If f_best = 0.1, beta = 101. (High exploration when close?)
    # This contradicts the text description.
    
    # Let's look at the inspirations again. They used:
    # beta = 5.0 * (f_best / (f_best + 1.0)) + 1.0
    # This matches the text description:
    # f_best large -> beta large.
    # f_best small -> beta small.
    
    # However, the current program (fixed beta=5.0) has a HIGHER score (6.79) than the adaptive ones (3.7).
    # This suggests that for this specific task (Beale 2D, 30 evals), the fixed aggressive exploration (beta=5.0) worked better than the adaptive ones which started with lower beta.
    # The adaptive ones started with beta=1.0 or 1.5 when f_best was small? No, f_best starts large (random points).
    # Wait, if f_best starts large, say 100.
    # Insp 1: beta = 5*(100/101)+1 ~ 5.5.
    # Insp 3: beta = 1.5+2.5*(100/101) ~ 4.0.
    # Current: beta = 5.0.
    
    # So Insp 1 was slightly more aggressive than current in the beginning.
    # Insp 3 was less aggressive.
    # Both scored worse.
    
    # Maybe the issue is that once f_best drops, the beta drops too fast, causing premature convergence to a local minimum?
    # The Beale function has local minima.
    
    # Let's try to combine the "fixed high beta" strength with a slight adaptation that doesn't drop as aggressively.
    # Or, perhaps the directive's "inverse" logic was actually meant to be:
    # beta = 1 + 10 * (1 / (1 + f_best)) ? No, that's low beta for large f_best.
    
    # Let's stick to the directive's likely intent: Adaptive beta that is high when f_best is large and low when f_best is small.
    # But since the fixed beta=5.0 is the best so far, let's try a beta that stays closer to 5.0 but adapts slightly.
    
    # Proposed:
    # beta = 1.0 + 4.0 * (f_best / (f_best + 0.5))
    # If f_best = 0, beta = 1.0
    # If f_best = 0.5, beta = 3.0
    # If f_best = 5, beta = 4.54
    # If f_best = 100, beta = 4.98
    
    # This keeps beta high (near 5) for most of the optimization, only dropping significantly when we are very close to the optimum (f_best < 1).
    # This might preserve the exploration strength of the current program while allowing for finer exploitation at the end.
    
    beta = 1.0 + 4.0 * (f_best / (f_best + 0.5))
    
    # Compute LCB scores (maximize -LCB)
    scores = -mu + beta * sigma_safe
    
    # Check if computation took too long
    if time.time() - start > limit:
        # Fallback to EI if something goes wrong
        z = (f_best - mu) / sigma_safe
        scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
