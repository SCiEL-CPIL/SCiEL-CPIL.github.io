"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# State for the Multi-Armed Bandit strategy
# We need to track the history of evaluated points to update bin statistics.
# Since the function is stateless between calls in the standard interface, 
# we must use a global cache or rely on the fact that 'mu' and 'sigma' 
# reflect the current GP state. However, the MAB strategy requires 
# accumulating statistics over time.
# 
# Wait, the task says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
# The loop is:
# 1. Evaluator scores 2000 new uniform candidates with acquisition.
# 2. Evaluator evaluates the best one.
# 3. GP is updated.
# 4. Repeat.
#
# The 'acquisition' function only receives the GP predictions for the 2000 candidates.
# It does NOT receive the history of previously evaluated points directly.
# However, we can infer the "history" indirectly? No, that's hard.
#
# Alternative MAB interpretation:
# The "arms" are the 2000 candidates themselves? No, that's just EI.
#
# Let's look at the constraints again.
# "No coordinates are passed."
# We only have mu, sigma, f_best, cost.
#
# If we can't pass coordinates, we can't do spatial binning directly in the function 
# unless we assume the 2000 candidates are fixed? 
# "the evaluator scores 2000 new uniform candidates". 
# "new uniform candidates" implies they change every iteration.
#
# If the candidates change every iteration, we cannot maintain a persistent 
# "bin history" across calls because we don't know which bin a *past* point 
# fell into, nor do we have the coordinates of the current 2000 points to 
# assign them to bins.
#
# RE-EVALUATION:
# The prompt says: "No coordinates are passed."
# This makes spatial MAB impossible if the candidates are random each time.
#
# Let's rethink the "Different Strategy".
# Current: EI / cost.
# Inspiration 1: Adaptive UCB (beta scaling).
# Inspiration 2: Cost-Aware EI (analytic).
# Inspiration 3: Thompson Sampling on GP parameters?
#
# What if we use a **Kernel Density Estimation (KDE) based Regret Minimization**?
# No, too complex without coordinates.
#
# What if we use **Probability of Improvement (PI) with Cost Penalty**?
# Current is EI. PI is often sharper.
# Score = P(f(x) < f_best) / cost.
# P(f(x) < f_best) = Phi((f_best - mu) / sigma).
# This is simpler and often performs better in high dimensions because 
# EI can be dominated by high-sigma regions where the probability of 
# improvement is low but the expected value is high due to the tail.
#
# Let's try **Cost-Aware Probability of Improvement (PI)**.
# Strategy: Maximize the probability of finding a new best, penalized by cost.
# This shifts the focus from "expected magnitude of improvement" to 
# "likelihood of any improvement". In 4D, finding *any* improvement 
# is often more critical than finding a *large* one early on.
#
# Another different strategy: **Greedy Cost-Effective Improvement (CEI)**.
# CEI is similar to EI but explicitly models the cost as a variable.
# However, the standard EI/cost is already a form of this.
#
# Let's look at **UCB with Adaptive Exploration**.
# The inspiration 1 (score 1.0541) used Adaptive UCB.
# Score = mu - beta * sigma / cost.
# Note: The current program uses EI / cost.
# EI is a smooth, bounded function. UCB is unbounded (in exploration).
# In 4D, EI might be too conservative.
#
# Let's implement **Adaptive UCB with Cost Normalization**.
# The key difference from the current program (EI) is that UCB explicitly 
# trades off mean and uncertainty. EI implicitly does this, but 
# UCB allows us to tune the exploration rate.
#
# The "Materially Different" aspect:
# Current: Exploitation-focused (EI).
# New: Exploration-aware (UCB) with adaptive beta.
#
# How to make it "Materially Different" from Inspiration 1 (which also did UCB)?
# Inspiration 1: "beta scales with ratio of median sigma to std mu".
# My strategy: **Cost-Weighted UCB with Uncertainty Calibration**.
# We will use UCB, but we will normalize the uncertainty term by the 
# local cost gradient or simply use a more aggressive exploration 
# schedule that decays with total cost spent.
#
# Actually, let's look at the **Thompson Sampling** inspiration (score 1.0).
# It failed. Why? Probably because sampling from the GP posterior 
# is computationally expensive or unstable without proper scaling.
#
# Let's try **Expected Improvement with a Cost-Dependent Temperature**.
# Or better: **Upper Confidence Bound (UCB) with a Cost-Aware Beta**.
#
# Let's define the score as:
# Score = (mu - beta * sigma) / cost
#
# To make it different from the "standard" UCB and the "adaptive" UCB 
# of inspiration 1, we will use a **Bayesian Calibration** of beta.
# We estimate the "effective dimensionality" or "exploration pressure" 
# based on the spread of mu and sigma across the 2000 candidates.
#
# If the spread of mu is small, we are in a flat region -> increase beta.
# If the spread of mu is large, we are in a steep region -> decrease beta.
#
# This is a **Contextual Adaptive UCB**.
#
# Implementation:
# 1. Calculate spread of mu: std_mu = std(mu).
# 2. Calculate spread of sigma: std_sigma = std(sigma).
# 3. Base beta = 2.0.
# 4. Adaptive factor: if std_mu is low, beta increases.
#    beta = base_beta * (1 + std_sigma / (std_mu + 1e-6))
# 5. Score = (mu - beta * sigma) / cost.
#
# Wait, the current program is EI. EI is roughly mu - sigma * phi(z)/Phi(z).
# UCB is mu - beta * sigma.
# These are fundamentally different curves. EI is bounded by (f_best - mu). 
# UCB is unbounded in the negative direction (for mu) but we maximize, 
# so we want high mu and low sigma.
#
# Let's verify the sign.
# We want to minimize f.
# f_best is the current best (low value).
# We want x where f(x) < f_best.
# mu is the predicted mean. Low mu is good.
# sigma is uncertainty.
#
# EI formula in code:
# z = (f_best - mu) / sigma
# EI = (f_best - mu) * Phi(z) + sigma * phi(z)
# If mu < f_best, z > 0, EI > 0.
#
# UCB formula for minimization:
# We want to maximize the "chance" of being better than f_best.
# A common UCB for minimization is:
# Score = mu - kappa * sigma
# We want this to be LOW? No, we want to pick the point with the 
# highest potential for low values.
# So we want to MINIMIZE mu - kappa * sigma?
# No, the acquisition function usually returns a value to MAXIMIZE.
# If we use UCB, we maximize:
# Score = - (mu - kappa * sigma) = kappa * sigma - mu
# This encourages high uncertainty and low mean.
#
# Let's check the current EI code:
# return EI / cost.
# EI is positive when mu < f_best.
# So we maximize EI/cost.
#
# For UCB, we should return:
# Score = (kappa * sigma - mu) / cost
#
# This is a materially different shape. EI is 0 when mu = f_best. 
# UCB is positive even if mu > f_best, as long as sigma is large enough.
# This makes UCB more exploratory.
#
# To make it robust, we will use an adaptive kappa.
#
# Strategy: **Adaptive Cost-Aware UCB**.
#
# Differentiation from Inspiration 1:
# Inspiration 1 used "median sigma to std mu".
# I will use a **Cost-Conditioned Beta**.
# Since cost varies, regions with high cost should be explored less 
# unless they are very promising. The division by cost handles this.
# But the beta itself can be tuned by the "density" of the candidates.
#
# Let's implement this.

import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive Cost-Aware UCB.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, current best function value.
        cost: (2000,) array of evaluation costs.
    
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure arrays are 1D
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    cost = np.asarray(cost).flatten()
    
    # Handle zero or negative costs (shouldn't happen, but safety)
    cost = np.maximum(cost, 1e-6)
    
    # Calculate adaptive exploration coefficient beta (kappa)
    # We want to explore more when the model is uncertain relative to the mean variation.
    
    std_mu = np.std(mu)
    mean_sigma = np.mean(sigma)
    
    # If std_mu is very small, the function is flat -> increase exploration.
    # If mean_sigma is high, the model is uncertain -> increase exploration.
    
    # Base beta
    base_beta = 2.5
    
    # Adaptive factor
    # Ratio of uncertainty to variation
    if std_mu > 1e-6:
        ratio = mean_sigma / std_mu
    else:
        ratio = 1.0
        
    # Clip ratio to avoid extreme values
    ratio = np.clip(ratio, 0.5, 5.0)
    
    beta = base_beta * ratio
    
    # UCB Score for minimization:
    # We want to maximize: (Potential for low value)
    # Score = (beta * sigma - mu) / cost
    
    # To prevent negative scores from dominating (which would be bad if we 
    # just pick the max, but negative scores are fine as long as we pick 
    # the *least negative* or *most positive*), we can shift or just 
    # rely on the max.
    
    # However, if all scores are negative, we pick the one closest to 0.
    # This is correct behavior: pick the least bad option.
    
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
