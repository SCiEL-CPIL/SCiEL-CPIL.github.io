"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track time to enforce the 330s limit.
# However, the function is called repeatedly. We must be careful with global state.
# The task says "Each evaluation must finish within 300 s".
# The evaluator loop:
# 1. Start with 5 points.
# 2. While cost < 30:
#    a. Score 2000 candidates.
#    b. Evaluate best.
#
# The acquisition function is called once per step (b) with 2000 candidates.
# It must return 2000 scores.
#
# Strategy:
# The current baseline is EI / cost.
# The inspiration suggests Adaptive LCB with dynamic beta.
# Let's try a hybrid or a refined LCB.
# LCB: mu - beta * sigma
# Cost-aware: (mu - beta * sigma) / cost ? Or just LCB?
# The cost is 1 + mean(x). Since x is uniform in [0,1], mean(x) is around 0.5.
# So cost is roughly 1.5.
# Dividing by cost penalizes expensive points.
#
# Let's look at the "Adaptive LCB" idea.
# beta can be tuned. A common adaptive beta is based on the number of iterations or the uncertainty.
# beta = sqrt(2 * log(n)) or similar.
# Here n is the number of evaluations so far.
# We don't have n directly, but we can infer it from f_best? No.
# We can infer it from the cost? The total cost is passed? No, `cost` is the cost of the *candidates*.
# Wait, the signature is `acquisition(mu, sigma, f_best, cost)`.
# `cost` has shape (2000,). It is the cost of evaluating each candidate.
# The total cost of the run is not passed.
# However, `f_best` is the lowest observed value.
#
# Let's stick to a cost-aware LCB.
# Score = (mu - beta * sigma) / cost
#
# What is a good beta?
# For Hartmann 3D, the function is smooth but has multiple local minima.
# A larger beta encourages exploration.
# The baseline EI is equivalent to LCB with a specific beta depending on the distance to f_best.
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
#
# Let's try a fixed beta LCB, but tuned for cost.
# Or perhaps a "Cost-Adjusted LCB" where beta is increased as we get closer to the optimum?
# We don't know how close we are.
#
# Alternative: Use the "Knowledge Gradient" or "Probability of Improvement"?
# POI = Phi(z). Cost-aware: POI / cost.
#
# The prompt mentions: "Adaptive LCB replaces EI, using a dynamic beta that scales with relative uncertainty to avoid EI's premature convergence".
# Let's implement Adaptive LCB.
# beta = beta_0 * (1 + log(1 + uncertainty_ratio)) ?
# Or simply a beta that is higher than typical EI.
#
# Let's try:
# z = (f_best - mu) / sigma
# EI is optimal for a specific risk profile.
# LCB with beta=2 is a common choice.
# Let's try beta=2.5 and divide by cost.
#
# Also, consider that `cost` varies.
# cost = 1 + mean(x).
# Points with smaller x have lower cost.
# The minimizer is in the inner 80% [0.1, 0.9].
# So the minimum cost is around 1 + 0.3 = 1.3 (if x is around 0.1,0.1,0.1? No, mean of coords).
# If x is (0.1, 0.1, 0.1), cost = 1 + 0.1 = 1.1.
# If x is (0.9, 0.9, 0.9), cost = 1 + 0.9 = 1.9.
# So there is a significant cost difference.
#
# Let's refine the LCB.
# Score = (mu - beta * sigma) / cost
#
# To make it adaptive, we can use a beta that depends on the "spread" of mu or sigma?
# Or just a fixed beta that is higher than the effective beta of EI.
# EI's effective beta varies. Near f_best, it's small. Far away, it's larger.
#
# Let's try a simple improvement:
# Use LCB with beta=2.0, divided by cost.
#
# Another idea: The "cost" term in the denominator might be too aggressive if sigma is large.
# If sigma is large, LCB is very low (good).
#
# Let's try:
# beta = 2.0
# score = (mu - beta * sigma) / cost
#
# To ensure stability, clip sigma.
#
# Let's also consider the deadline.
# The function is called with 2000 points.
# Computing norm.cdf and norm.pdf is fast.
# There is no heavy computation here.
# The deadline guard is mostly for safety, but this function is O(N).
#
# Let's write the code.

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Safety margin: 330s total, we are inside the loop.
    # Each call should be fast.
    
    # Clip sigma to avoid division by zero or extreme values
    sigma_safe = np.clip(sigma, 1e-6, None)
    
    # Cost-aware Lower Confidence Bound
    # Beta controls exploration. 
    # EI is roughly equivalent to LCB with beta ~ 1-2 depending on z.
    # Let's use a slightly higher beta to encourage exploration of multiple minima.
    beta = 2.0
    
    lcb = mu - beta * sigma_safe
    
    # Normalize by cost
    # cost is (1 + mean(x))
    # Higher cost -> lower score
    scores = lcb / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    elapsed = time.time() - start_time
    if elapsed > 290: # Safety check, though this function is very fast
         pass
         
    return scores
# EVOLVE-BLOCK-END
