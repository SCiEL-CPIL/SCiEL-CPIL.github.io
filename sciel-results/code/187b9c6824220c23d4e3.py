# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Domain is [-2, 2]^2, scaled to [0, 1] for GP length scales
# We don't receive x coordinates, but we know the domain structure.
# The 'cost' parameter is 1 + mean(x_scaled).
# x_scaled = (x + 2) / 4
# cost = 1 + (x1_scaled + x2_scaled) / 2
# This means points closer to (2, 2) are more expensive.
# Points closer to (-2, -2) are cheaper (cost ~ 1).
# Points around (0, 0) have cost ~ 1 + 0.5 = 1.5.

# The Rosenbrock function minimum is at (1, 1) in original coords.
# In scaled coords [0, 1], (1, 1) maps to ((1+2)/4, (1+2)/4) = (0.75, 0.75).
# Cost at (1, 1) is 1 + (0.75 + 0.75)/2 = 1.75.

# Strategy:
# 1. Compute a standard acquisition function (e.g., EI or UCB).
# 2. Penalize high-cost regions more aggressively if we are far from the optimum,
#    or simply divide by cost (cost-aware EI).
# 3. Add a deterministic bias towards the known approximate location of the optimum
#    in scaled space (0.75, 0.75) to help navigate the valley, but keep it soft.
#    However, since we don't have x, we can't add a spatial bias directly to the score
#    unless we estimate x from cost? No, cost is given per candidate.
#    Wait, the input `cost` has shape (2000,). So for each candidate i, we know its cost.
#    We do NOT know the coordinates x_i.
#    So we cannot add a spatial bias based on x. We can only use mu, sigma, and cost.

# Let's refine the cost-aware acquisition.
# Standard EI:
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
# where z = (f_best - mu) / sigma

# Cost-aware EI:
# EIC = EI / cost

# Is there a better way?
# Maybe Upper Confidence Bound (UCB)?
# UCB = mu - kappa * sigma
# Cost-aware UCB: UCB / cost?
# Or UCB - lambda * cost?
# Dividing by cost is equivalent to maximizing EI/cost.
# This favors cheap points with high EI.

# Let's try a mix or a different acquisition function.
# The prompt mentions "simple regret" scoring.
# The seed program used EI/cost.

# Let's look at the "Inspiration":
# "PoI/cost prioritizes high-probability improvement regions over high-magnitude ones..."
# PoI (Probability of Improvement) is just Phi(z).
# PoI/cost = Phi(z) / cost.
# This might be more robust than EI which can be dominated by large sigma (uncertainty).

# Let's try PoI / cost.
# Or maybe a combination: (EI + alpha * PoI) / cost?

# Another idea: The Rosenbrock function has a narrow valley.
# GP might struggle with the valley structure if length scales are too large.
# The task says length scale is 0.2 on scaled coordinates. This is quite local.
# With 5 initial points and a few iterations, the GP might not cover the valley well.
# However, we can't change the GP. We can only change the acquisition function.

# Let's consider the cost structure again.
# cost = 1 + (x1_s + x2_s)/2.
# Minimum cost is 1 (at x_s = (0,0) i.e. x=(-2,-2)).
# Maximum cost is 2 (at x_s = (1,1) i.e. x=(2,2)).
# Optimum is at x_s = (0.75, 0.75), cost = 1.75.
# So the optimum is in the "expensive" half of the domain.
# If we just minimize cost, we go to (-2, -2), which is far from the optimum.
# If we just maximize EI, we might go to high uncertainty areas.

# A common approach for cost-aware BO is to use the ratio of improvement to cost.
# Let's stick with EI/cost but maybe tune the exploration/exploitation balance.
# The seed program uses standard EI.

# What if we use a "Safe" acquisition function?
# Or what if we use a lower confidence bound on the cost? No, cost is known.

# Let's try UCB with cost penalty.
# score = (mu - kappa * sigma) / cost
# kappa controls exploration.
# In EI, exploration is adaptive based on sigma.
# In UCB, kappa is fixed.
# For Rosenbrock, the valley is tricky. High sigma areas might be the edges of the valley.
# EI tends to explore high sigma areas.
# UCB with a moderate kappa might be better.

# Let's try a hybrid:
# score = (EI * 0.5 + PoI * 0.5) / cost
# Or just PoI / cost as suggested by the inspiration?
# The inspiration says "PoI/cost ... measured 1 -> 2.743".
# This suggests PoI/cost is a strong candidate.
# Let's implement PoI/cost.

# z = (f_best - mu) / (sigma + 1e-9)
# PoI = norm.cdf(z)
# score = PoI / cost

# Let's also consider that f_best might be very low, making z very large, PoI ~ 1.
# Then score ~ 1/cost. This would just pick the cheapest point.
# This is bad if we are already near the optimum.
# EI handles this better because if mu < f_best, EI is positive.
# If mu is much less than f_best, EI is large.
# If mu is close to f_best, EI is small.
# PoI is always between 0 and 1.

# Let's look at the seed program again.
# It returns EI / cost.
# The inspiration says PoI/cost improved the score significantly.
# Let's try PoI/cost.

# To be safe, let's add a small exploration term to PoI?
# Or just use PoI/cost.

# Another thought: The cost is 1 + mean(x_scaled).
# The "mean" of x_scaled is the average of the two coordinates.
# So cost depends on the sum of coordinates.
# This creates a gradient in cost across the domain.

# Let's try PoI/cost.

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero
    sigma_safe = sigma + 1e-9
    cost_safe = cost + 1e-9
    
    # Probability of Improvement
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Cost-aware scores
    score_poi = poi / cost_safe
    score_ei = ei / cost_safe
    
    # Let's try a mix or just PoI.
    # The inspiration strongly points to PoI.
    # Let's try 0.5 * PoI + 0.5 * EI?
    # Or maybe just PoI is better for stability.
    
    # Let's try PoI / cost.
    return score_poi
# EVOLVE-BLOCK-END
