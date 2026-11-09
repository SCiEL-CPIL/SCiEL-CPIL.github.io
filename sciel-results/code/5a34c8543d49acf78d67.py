"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# We need to track state across calls to implement a hybrid strategy or diversity
# However, the signature is stateless. We must rely on the inputs mu, sigma, f_best.
# Since we cannot store state in the function itself easily without globals,
# and globals are reset per evaluation run? No, the python process persists.
# But "fresh uniform candidates" are drawn each step.
# We can use a global variable to track previous best or number of calls?
# The prompt says: "Write acquisition(mu, sigma, f_best) for a loop..."
# It implies the function is called repeatedly.
# Using globals is risky if the evaluator restarts the interpreter for each seed.
# "For each seed 0, 1 and 2 the evaluator starts..."
# Usually, the module is imported once, and the loop runs.
# Let's assume we can use a simple global counter or just a robust stateless heuristic.

# A robust stateless heuristic for Michalewicz 2D:
# 1. Expected Improvement (EI) is the standard.
# 2. Adding an exploration term based on uncertainty (sigma) helps escape local optima.
# 3. Michalewicz 2D has a known global minimum near (2.20, 1.57).
#    However, we shouldn't hardcode the solution.
# 4. The "diversity bonus" mentioned in the prompt (0.1*sqrt(sigma_norm)) improved a score.
#    Let's try a weighted combination of EI and UCB (Upper Confidence Bound).

# To be safe and fast:
# score = EI + beta * sigma
# beta controls exploration.
# For 30 evaluations, we want to explore early and exploit later.
# But we don't know the step number.
# We can estimate the "age" of the search by looking at how tight sigma is relative to the range?
# Or simply use a fixed beta that works well for small budgets.

# Let's look at the measured knowledge:
# "Adding a diversity/exploration bonus (0.1*sqrt(sigma_norm)) to EI ... 1 -> 0.6944"
# Note: The score is "simple regret" where lower is better?
# "Scoring (lower is better): mean over the seeds of the simple regret..."
# "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
# Wait, the contract says:
# "Scoring (lower is better): mean over the seeds of the simple regret, lowest observed value minus the minimum value."
# "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
# This is slightly contradictory phrasing. Let's clarify:
# Simple Regret = f_best - f_min. Lower is better.
# Search Score = seed_value / value? No, "seed/value".
# If seed value is 1.0, and my value is 0.5 (better, lower regret), score = 1.0 / 0.5 = 2.0.
# So higher score is better.
# The measured improvement "1 -> 0.6944" likely refers to the *regret* or a normalized metric?
# If the score went from 1.0 to 0.6944, and "above 1.0 is an improvement", then 0.6944 would be a *worsening* if it's the score.
# BUT, if the metric is "simple regret", then 0.6944 is better than 1.0.
# The prompt says: "Measured: 1 -> 0.6944. (strategy of a program that improved island 6)"
# And "above 1.0 is an improvement" for the *search score*.
# It's highly probable that "1 -> 0.6944" refers to the *regret* dropping from 1.0 to 0.6944, which means the *score* increased from 1.0 to 1/0.6944 ≈ 1.44.
# So, adding the exploration bonus worked.

# Let's implement a hybrid acquisition function:
# EI + alpha * sigma
# We need to choose alpha.
# sigma is in objective units.
# For Michalewicz, values are around -1.8 to 0.
# sigma might be small (1e-6 noise) or larger (uncertainty).
# If sigma is very small, EI dominates. If sigma is large, exploration dominates.
# A dynamic alpha based on the number of evaluations would be ideal, but we are stateless.
# We can approximate the "age" by the magnitude of sigma? No.
# We can use a fixed alpha that encourages exploration.
# The prompt suggests 0.1 * sqrt(sigma_norm).
# sigma_norm might be sigma / (max(sigma) + eps) or similar?
# Or just sqrt(sigma)?
# Let's try: score = EI + 0.1 * sigma
# Or score = EI + 0.5 * sigma (stronger exploration)

# Let's refine the EI calculation for stability.
# z = (f_best - mu) / sigma
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)

# Potential issue: if sigma is very small, z is huge, EI is 0 or very small.
# The exploration term 0.1 * sigma will be tiny.
# This might lead to pure exploitation too early.

# Alternative: Use UCB directly?
# UCB = mu + beta * sigma
# For minimization, we want low mu and high sigma?
# No, UCB for minimization is usually: mu - beta * sigma (if we want to find low values)
# Wait, standard UCB for maximization is mu + beta * sigma.
# For minimization, we can maximize -f, so UCB = -mu + beta * sigma.
# Or we can define an acquisition function that we want to MAXIMIZE.
# The prompt says: "Return 2000 finite scores, higher meaning evaluate next".
# So we need to maximize the score.
# If we use EI, higher EI is better.
# If we use UCB for minimization, we want to maximize the chance of finding a low value.
# A common approach for minimization is:
# score = -mu + beta * sigma
# This encourages low mu (good) and high sigma (exploration).

# Let's compare EI and UCB.
# EI is generally better for Bayesian Optimization.
# The "diversity bonus" in the prompt was added to EI.
# So: score = EI + gamma * sigma

# What should gamma be?
# The prompt mentioned 0.1 * sqrt(sigma_norm).
# If sigma is in objective units, and objective range is ~2 (from 0 to -2),
# sigma might be ~0.1. sqrt(0.1) ~ 0.3. 0.1 * 0.3 = 0.03.
# If we just use 0.1 * sigma, it's 0.01.
# Let's try a slightly stronger exploration term.
# score = EI + 0.5 * sigma

# Also, we should handle the case where sigma is very small.
# If sigma < 1e-8, set sigma = 1e-8 to avoid division by zero in z.

# Let's also consider the deadline.
# The function is called 30 times. Each call processes 2000 points.
# This is very fast. The deadline is not a concern for the function itself.
# The deadline applies to the whole evaluation (300s).
# 30 calls * 2000 points * (GP prediction time) is likely well within 300s.
# GP prediction is the bottleneck, but that's done by the evaluator, not us.
# We just compute the score.

# So, a simple, robust function:

def acquisition(mu, sigma, f_best):
    # Ensure sigma is not zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus
    # We want to encourage exploration in areas of high uncertainty.
    # A linear term in sigma is a common UCB-like bonus.
    # Let's use a coefficient that balances exploration and exploitation.
    # For a small budget (30 evals), exploration is crucial.
    # Let's try 0.5 * sigma.
    exploration_bonus = 0.5 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
