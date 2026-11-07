"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track evaluation count and start time for deadline guard
_eval_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    global _eval_count
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
        
    _eval_count += 1
    
    # Deadline guard: The evaluator kills after 330s.
    # We have 30 evaluations. If we are close to the deadline, 
    # we should return a safe, fast result. 
    # However, the acquisition function itself is called 30 times. 
    # Each call must be fast. 
    # Thompson Sampling on 2000 points is O(N), so it's very fast.
    # LCB is also O(N).
    # No complex iterative optimization is performed here, so timeout risk is low.
    
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9)
    
    # Avoid division by zero or negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the lowest observed value.
    # In Bayesian Optimization for minimization, we want to find points with low mu and high sigma.
    # LCB = mu - beta * sigma
    
    # Adaptive beta strategy:
    # When f_best is large (far from optimum, e.g., initial random points), we want high exploration.
    # When f_best is small (close to optimum), we want high exploitation (low beta).
    # 
    # The directive suggests: beta = 1 + 10 / f_best
    # However, f_best can be very close to 0, causing beta to explode.
    # Also, f_best could be negative? Beale function minimum is 0. 
    # Beale function: (1.5 - 2x + x^2)(2.25 - 1.5y + y^2)(2.625 - y)
    # Min at (3, 0.5) is 0.
    # Values are generally positive.
    # If f_best is very small, 10/f_best is huge. 
    # We should cap beta or use a safe denominator.
    # Let's use max(f_best, epsilon) to prevent division by zero.
    # But if f_best is 1e-5, beta = 1 + 10/1e-5 = 100001. 
    # This would make LCB = mu - 100001 * sigma. 
    # This is extremely aggressive exploration. 
    # Is that what we want? 
    # The inspiration said "fixed high-beta LCB (beta=5) ... scored highest".
    # The directive says "adaptive beta ... scales with inverse of current best value".
    # If f_best is small, we are close to the minimum. 
    # Usually, we want to EXPLOIT when close. 
    # So beta should DECREASE as f_best decreases? 
    # "prioritizes high uncertainty regions when the solution is far from the global minimum" -> High beta when f_best is large.
    # "tightening the confidence bound as f_best approaches zero" -> Low beta when f_best is small.
    # 
    # Formula: beta = 1 + 10 / f_best implies:
    # f_best = 10 -> beta = 2
    # f_best = 1 -> beta = 11
    # f_best = 0.1 -> beta = 101
    # This is the OPPOSITE of what the text describes.
    # "scales with the inverse of the current best value" -> 1/f_best.
    # If f_best is large (far), 1/f_best is small -> Low beta? 
    # If f_best is small (close), 1/f_best is large -> High beta?
    # 
    # Let's re-read carefully: 
    # "prioritizes high uncertainty regions when the solution is far from the global minimum" 
    # Far => f_best is large (assuming f_min=0). 
    # So we want HIGH beta when f_best is LARGE.
    # "tightening the confidence bound as f_best approaches zero" 
    # Close => f_best is small. 
    # So we want LOW beta when f_best is SMALL.
    # 
    # So we want beta to be INVERSELY proportional to f_best? No.
    # We want beta to be DIRECTLY proportional to f_best?
    # Or maybe the text meant "scales with the value" not "inverse"?
    # "scales with the inverse of the current best value" usually means y = 1/x.
    # If y = 1/x:
    # x large -> y small.
    # x small -> y large.
    # This contradicts the goal description.
    # 
    # Let's look at the inspiration: "fixed high-beta LCB (beta=5) ... empirically scored highest".
    # This suggests that a constant, reasonably high beta is good.
    # 
    # Let's try to interpret "adaptive beta ... scales with the inverse of the current best value" in a way that makes sense for the goal.
    # Maybe it means beta = 1 + k * f_best? 
    # If f_best is large, beta is large. If f_best is small, beta is small.
    # This matches the goal: High exploration when far, low exploration when close.
    # 
    # Let's try: beta = 1 + 0.5 * f_best
    # Initial f_best ~ 100? Beale range is roughly 0 to 1000?
    # Beale function values:
    # At (0,0): (1.5)(2.25)(2.625) ~ 8.8
    # At (-4.5, -4.5): (1.5 + 9 + 20.25)(2.25 + 6.75 + 20.25)(2.625 + 4.5) = 30.75 * 29.25 * 7.125 ~ 6400
    # So f_best starts around 10-1000.
    # If f_best = 100, beta = 1 + 50 = 51. Very high exploration.
    # If f_best = 1, beta = 1 + 0.5 = 1.5. Moderate.
    # If f_best = 0.01, beta = 1.005. Low exploration.
    # 
    # This seems reasonable.
    # 
    # However, the text explicitly says "inverse". 
    # Could it be that f_best is negative? No, min is 0.
    # Could it be that "current best value" refers to something else? No, f_best is defined.
    # 
    # Let's assume the text has a typo and meant "scales with the value" or "scales linearly".
    # Or maybe "inverse" refers to the relationship between beta and the DISTANCE to optimum?
    # Distance ~ f_best.
    # If we want High Beta when Distance is High, Beta should increase with Distance.
    # 
    # Let's try: beta = 1 + f_best / 10
    # If f_best = 100, beta = 11.
    # If f_best = 10, beta = 2.
    # If f_best = 1, beta = 1.1.
    # 
    # Let's check the inspiration again: "fixed high-beta LCB (beta=5) ... scored highest".
    # This implies that beta=5 is a good sweet spot.
    # If we start with f_best=100, beta=11 might be too high? Or just right?
    # If we end with f_best=0.1, beta=1.01 might be too low? 
    # 
    # Let's try to map the range of f_best to a beta range of [1, 5] or [1, 10].
    # 
    # Alternative interpretation:
    # Maybe the "inverse" is correct, but the goal description is "prioritizes high uncertainty ... when far".
    # If I use beta = 1 + 10/f_best:
    # Far (f_best=100): beta = 1.1. (Low exploration)
    # Close (f_best=0.1): beta = 101. (High exploration)
    # This would mean we explore MORE when we are CLOSE. This is usually bad (we should exploit).
    # 
    # So I will bet on the goal description being the intent, and the "inverse" wording being a mistake in the prompt's directive or my interpretation of "inverse of current best value" (maybe it meant inverse of the *error*? No).
    # 
    # I will implement: beta = 1 + 0.1 * f_best
    # This ensures beta grows as we are further away.
    # 
    # Let's add a cap to beta to prevent it from becoming absurdly large if f_best is huge.
    # beta = min(1 + 0.1 * f_best, 10.0)
    
    if f_best <= 0:
        beta = 1.0
    else:
        beta = 1.0 + 0.1 * f_best
        # Cap beta to avoid numerical issues or overly aggressive exploration
        if beta > 10.0:
            beta = 10.0
        # Floor beta at 1.0 to ensure some exploitation
        if beta < 1.0:
            beta = 1.0

    # Compute LCB
    # We want to MINIMIZE the function, so we want LOW mu and HIGH sigma.
    # Score = mu - beta * sigma
    # We return scores where HIGHER means evaluate next.
    # So we should return -LCB = -(mu - beta * sigma) = -mu + beta * sigma
    # Or just return LCB and let the evaluator pick the minimum?
    # The contract says: "Return 2000 finite scores, higher meaning evaluate next"
    # So we must return a value where higher is better.
    # If we want to minimize f, and LCB = mu - beta*sigma, then lower LCB is better.
    # So we should return -LCB.
    
    lcb = mu - beta * sigma
    score = -lcb
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score

# EVOLVE-BLOCK-END
