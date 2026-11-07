"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
# Styblinski-Tang 2D: f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i) for x in [-5, 5]^2
# Global minimum at x* = (-2.903534, -2.903534), f* = -78.3323
# The function is symmetric in each coordinate.
# f(x) = 0.5 * [x1^4 - 16*x1^2 + 5*x1 + x2^4 - 16*x2^2 + 5*x2]
# g(t) = t^4 - 16*t^2 + 5*t
# g'(t) = 4*t^3 - 32*t + 5 = 0
# Roots: t ≈ -2.9035, t ≈ 0.1582, t ≈ 2.7453
# g(-2.9035) ≈ -156.664 -> f = 0.5 * (-156.664 * 2) = -156.664? No.
# f(x*, x*) = 0.5 * (g(x*) + g(x*)) = g(x*)
# g(-2.903534) = (-2.903534)^4 - 16*(-2.903534)^2 + 5*(-2.903534)
# = 70.79 - 134.86 - 14.52 = -78.59? Let's calculate precisely.
# x = -2.903534027848996
# x^2 = 8.43051011887902
# x^4 = 71.0735236055534
# 16*x^2 = 134.888161902064
# 5*x = -14.5176701392449
# g(x) = 71.0735236055534 - 134.888161902064 - 14.5176701392449 = -78.3323084357555
# So f(x*, x*) = -78.3323084357555

# The task says: "Each function runs on five seeded instances f_k(x) = f(x - s_k) whose minimizer is moved to a uniform point in the inner 80% of each range"
# This means the global minimum is NOT at a fixed location in the scaled [0,1]^2 space. It moves per instance.
# Therefore, we cannot hard-code the location of the minimum.
# We must rely on the GP and the acquisition function.

# However, the cost is 1 + mean(x_scaled).
# x_scaled is in [0, 1]^2.
# So cost is between 1 and 2.
# The budget is 30.
# We start with 5 points. Cost ~ 1.5 each -> 7.5.
# Remaining budget ~ 22.5.
# Each eval costs ~ 1.5.
# So we can do ~ 15 more evals.
# Total evals ~ 20.

# The acquisition function receives mu, sigma, f_best, cost.
# cost is the cost of evaluating each candidate point.
# cost = 1 + mean(x_scaled).
# So cost varies per point.
# We want to maximize (expected improvement) / cost.

# Current program uses EI / cost.
# This is a standard approach.
# Why is it scoring 1.0?
# Maybe it's not exploring enough or exploiting enough.
# Or maybe the GP is not well-calibrated.

# Let's try a different acquisition function.
# Instead of EI, let's try UCB (Upper Confidence Bound) or PI (Probability of Improvement).
# Or a combination.
# UCB: mu + beta * sigma.
# PI: norm.cdf((f_best - mu) / sigma).

# Since we want to minimize, EI is:
# EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma.

# Let's try to improve the GP model or the acquisition function.
# The task says: "GP: Matern-5/2, length scale 0.2 on coordinates scaled to [0, 1], noise 1e-6, standardized targets"
# This is fixed. We cannot change the GP.

# So we must change the acquisition function.
# Let's try a more aggressive exploration strategy.
# Or a more exploitation-focused one.

# Let's try: EI / (cost + epsilon) where epsilon is small.
# Or maybe: EI * (1 / cost) is not the best.
# What if we use: (f_best - mu) / cost? This is deterministic improvement per cost.
# But it ignores uncertainty.

# Let's try: (f_best - mu - alpha * sigma) / cost? No, that's not right.

# Let's try a hybrid:
# score = EI / cost + gamma * sigma / cost
# This balances exploitation and exploration.

# Or simply:
# score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# score = score / (cost ** 2)  # More penalty for high cost

# Let's try cost ** 2.

# Also, we should ensure numerical stability.
# sigma can be very small.
# z = (f_best - mu) / (sigma + 1e-9)

# Let's try:
# EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# return EI / (cost ** 1.5)

# Or maybe the issue is that the GP is not capturing the function well.
# But we can't change the GP.

# Let's try a different acquisition function:
# Expected Improvement per unit cost, but with a different scaling.

# Let's try:
# z = (f_best - mu) / (sigma + 1e-9)
# EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# return EI / cost

# This is the current program. It scores 1.0.

# Let's try:
# return (f_best - mu) / cost
# This is pure exploitation.

# Or:
# return sigma / cost
# This is pure exploration.

# Let's try a mix:
# return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost * (1 + sigma))

# This penalizes high uncertainty more.

# Let's try:
# return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost + sigma)

# This seems reasonable.

# Let's try:
# return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost ** 2)

# Let's try:
# return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / (cost * (1 + 2*sigma))

# I'll try a few variations and see which one is most promising.
# Since I can't run the code, I'll have to guess.
# The most common improvement is to adjust the balance between exploitation and exploration.
# Let's try:
# score = EI / cost
# But with a different EI formula? No, EI is standard.

# Maybe the issue is that the GP is not well-calibrated, so sigma is not a good measure of uncertainty.
# But we can't change the GP.

# Let's try:
# return (f_best - mu) / (cost * (sigma + 1e-9))
# This is like a z-score.

# Or:
# return norm.cdf((f_best - mu) / (sigma + 1e-9)) / cost
# This is the probability of improvement per cost.

# Let's try PI / cost.

# I'll go with a hybrid of EI and PI.
# score = 0.5 * EI / cost + 0.5 * PI / cost

# Let's try:
# z = (f_best - mu) / (sigma + 1e-9)
# EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
# PI = norm.cdf(z)
# return (EI + 0.1 * PI) / cost

# This adds a small exploration bonus.

# Let's try this.

def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit cost, with a small exploration bonus.
    z = (f_best - mu) / (sigma + 1e-9)
    EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    PI = norm.cdf(z)
    return (EI + 0.1 * PI) / cost
# EVOLVE-BLOCK-END
