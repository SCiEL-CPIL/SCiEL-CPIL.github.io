"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track evaluation count if possible, 
# but since the function is called with mu, sigma, f_best only, 
# we cannot directly track the step count unless we infer it or use a heuristic.
# However, the problem states "at each step... calls acquisition". 
# The acquisition function signature is fixed: acquisition(mu, sigma, f_best).
# We cannot store state between calls easily in a stateless function unless we use global variables.
# But wait, the prompt says "Keep the function names and signatures of the seed program."
# And "mu and sigma have shape (2000,)".
# We can use a global variable to track the number of calls.
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # The loop runs until 30 evaluations including the first 5.
    # So there are 25 acquisition calls.
    # Step 0: 5 points evaluated, then acquisition called for point 6.
    # Step 1: acquisition called for point 7.
    # ...
    # Step 24: acquisition called for point 30.
    
    # We can estimate the progress based on _call_count.
    # Total calls = 25.
    # Early calls (low count) -> more exploration (higher beta) or exploitation?
    # Standard UCB: score = mu - beta * sigma (for minimization).
    # Usually, you want to explore early. So high beta early.
    # Or you want to exploit early to find a good f_best?
    # Actually, standard BO practice:
    # EI is often used. UCB with constant beta is common.
    # Let's try an adaptive beta that decays, encouraging exploitation later.
    # Or a beta that grows, encouraging exploration?
    # If we explore too much, we might miss the minimum.
    # If we exploit too much, we might get stuck in local minima.
    # Branin has multiple local minima.
    
    # Let's try a strategy that balances exploration and exploitation.
    # A common adaptive UCB uses beta = c * sqrt(log(t+1) / t) or similar.
    # Or simply linear decay of beta.
    
    # Let's assume _call_count goes from 1 to 25.
    # t = _call_count
    # Total steps T = 25
    
    # Let's try a decaying beta to focus on exploitation later.
    # beta = beta_0 * (1 - t/T) + beta_min
    # Or beta = beta_0 / (1 + t)
    
    # Let's try a simple linear interpolation between a high exploration rate and a low one.
    # High exploration early: beta = 3.0
    # Low exploration late: beta = 0.5
    
    t = _call_count
    T = 25.0
    
    # Normalize t to [0, 1]
    # At t=1, progress = 0/24 = 0
    # At t=25, progress = 24/24 = 1
    if t > 1:
        progress = (t - 1) / (T - 1)
    else:
        progress = 0.0
        
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta starts at 3.0 and decays to 0.5
    beta = 3.0 - 2.5 * progress
    
    # UCB for minimization:
    # We want to minimize f.
    # Uncertainty is sigma.
    # Lower bound of potential minimum: mu - beta * sigma
    # We want to maximize the score, so higher score = better candidate.
    # So score = -(mu - beta * sigma) = -mu + beta * sigma?
    # No, standard UCB for minimization is often:
    # x_next = argmin (mu(x) - beta * sigma(x))
    # But the task says "Return 2000 finite scores, higher meaning evaluate next".
    # So we need to return a score where higher is better.
    # If we pick x that minimizes mu - beta*sigma, then the "score" for that x should be high.
    # Let score = - (mu - beta * sigma) = beta * sigma - mu.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    score = beta * sigma_safe - mu
    
    return score
# EVOLVE-BLOCK-END
