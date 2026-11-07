"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # beta scales with the current best value to encourage exploration early
    # and exploitation later.
    # f_best is the minimum observed value. As f_best -> 0, beta -> 0.
    # As f_best is large, beta is large.
    
    # Ensure f_best is non-negative for log
    f_best_safe = max(f_best, 1e-6)
    
    # Calculate adaptive beta
    # Using a base of 2.0 to ensure sufficient exploration initially
    beta = 2.0 * np.sqrt(np.log(1.0 + f_best_safe))
    
    # Calculate UCB score: mu - beta * sigma
    # Higher score is better for minimization (we want low mu and high sigma)
    # Note: In minimization, we typically minimize mu - beta*sigma.
    # The task says "Return 2000 finite scores, higher meaning evaluate next".
    # So we return the UCB value itself. The evaluator picks the argmax.
    # Since we are minimizing f(x), a low mu is good. A high sigma is good for exploration.
    # The quantity mu - beta*sigma will be very negative where sigma is high.
    # Wait. If we return mu - beta*sigma, the argmax will pick the point with the HIGHEST mu - beta*sigma.
    # This is WRONG for minimization if we just use standard UCB definition.
    
    # Standard UCB for minimization:
    # We want to find x that minimizes f(x).
    # The posterior mean is mu. The uncertainty is sigma.
    # A good candidate has low mu (likely low f) OR high sigma (potential for lower f).
    # The UCB acquisition function for minimization is often defined as:
    # alpha(x) = mu(x) - beta * sigma(x)
    # We want to MAXIMIZE alpha(x)? No.
    # If mu is low, alpha is low. If sigma is high, alpha is low (more negative).
    # So maximizing alpha(x) would pick the point with the HIGHEST mu and LOWEST sigma.
    # This is EXPLOITATION of the WORST points. That's wrong.
    
    # Correction:
    # For MINIMIZATION, the acquisition function should be high where we expect low f.
    # Let's look at EI for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # This is high when f(x) is likely lower than f_best.
    
    # For UCB in minimization:
    # We want to balance low mu and high sigma.
    # A common formulation is to MINIMIZE mu - beta * sigma? No.
    # Let's think.
    # If mu = 0, sigma = 0. Score should be high.
    # If mu = 10, sigma = 0. Score should be low.
    # If mu = 5, sigma = 10. Score should be medium/high (exploration).
    
    # If we use Score = - (mu - beta * sigma) = -mu + beta * sigma.
    # High mu -> Low score. Good.
    # High sigma -> High score. Good.
    # So Score = -mu + beta * sigma.
    # This is equivalent to maximizing -mu + beta * sigma.
    # Or minimizing mu - beta * sigma.
    
    # Let's verify with the task description: "higher meaning evaluate next".
    # So we want the score to be HIGH for good candidates.
    # Good candidate: Low mu, High sigma.
    # Score = -mu + beta * sigma.
    # If mu is small (good), -mu is large (positive).
    # If sigma is large (good), beta*sigma is large (positive).
    # So Score = -mu + beta * sigma works.
    
    # Alternatively, some define UCB for minimization as mu - beta * sigma and then take the MINIMUM.
    # But the interface asks for scores where HIGHER is better.
    # So we must return a value that is large for good points.
    # Therefore: Score = -mu + beta * sigma.
    
    # Let's double check EI logic.
    # EI is positive when f(x) < f_best.
    # So EI is high for good points.
    
    # So for UCB:
    # We want to maximize the potential for improvement.
    # Potential lower bound is mu - beta * sigma.
    # We want this lower bound to be as LOW as possible.
    # So we want to MINIMIZE (mu - beta * sigma).
    # To make "higher score" correspond to "lower value", we can return:
    # Score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's use this formula.
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
