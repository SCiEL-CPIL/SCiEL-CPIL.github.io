"""Rewritten program using Thompson Sampling for cost-aware BO."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Thompson Sampling scores; higher is better (evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Draw a single scalar sample from the posterior distribution
    # This is the key idea of Thompson Sampling: sample a parameter once,
    # then score all candidates based on that single sample.
    # The posterior over the optimal value is approximated by N(f_best, sigma_opt)
    # where sigma_opt is a characteristic uncertainty.
    
    # Use the mean of sigma as a proxy for overall uncertainty,
    # but weight by exploration: use a quantile or just the mean.
    # Actually, in standard TS for optimization, we sample from the posterior
    # of the objective at each point. But the directive says:
    # "sample a single scalar value theta from a Gaussian distribution 
    #  with mean mu and standard deviation sigma for all 2000 candidates"
    
    # This is a bit ambiguous. Let me re-read:
    # "sampling a single scalar value theta from a Gaussian distribution 
    #  with mean mu and standard deviation sigma for all 2000 candidates"
    
    # I think this means: for each candidate i, we have mu[i] and sigma[i].
    # But "a single scalar value theta" suggests we draw ONE theta.
    # Perhaps the intent is: draw theta ~ N(f_best, sigma_ref) once,
    # then score each candidate by how likely it is to beat theta?
    
    # Actually, re-reading more carefully: "score them using the negative 
    # cost-normalized value -theta / cost". This suggests theta is a single
    # value, and we compute -theta / cost for each candidate. But that doesn't
    # depend on mu or sigma per candidate, which seems wrong.
    
    # Let me interpret this differently: perhaps the idea is that we model
    # the optimal value as a random variable theta ~ N(f_best, sigma_global),
    # and then for each candidate x_i, we compute the probability that
    # f(x_i) < theta, which involves mu[i] and sigma[i].
    
    # A common Thompson Sampling approach for optimization:
    # 1. Sample theta ~ N(f_best, sigma_global) where sigma_global captures
    #    overall uncertainty.
    # 2. For each candidate, compute P(f(x_i) < theta) = Phi((theta - mu[i]) / sigma[i])
    # 3. Score by this probability divided by cost.
    
    # But the directive specifically says "negative cost-normalized value -theta / cost".
    # Let me just follow the directive literally: draw one theta from a Gaussian
    # with mean f_best and some sigma, then score = -theta / cost.
    # But this doesn't use mu and sigma arrays at all, which seems odd.
    
    # Alternative interpretation: perhaps "for all 2000 candidates" means we
    # use the same theta for all, but the score incorporates mu and sigma somehow.
    
    # Let me try a more standard TS approach that still samples once:
    # Sample theta ~ N(f_best, sigma_mean) where sigma_mean = mean(sigma)
    # Then for each candidate, score = Phi((theta - mu[i]) / sigma[i]) / cost[i]
    # This is the probability that the candidate beats the sampled threshold.
    
    sigma_mean = float(np.mean(sigma))
    sigma_mean = max(sigma_mean, 1e-6)
    
    # Draw single sample
    theta = f_best + sigma_mean * np.random.standard_normal()
    
    # Compute probability that each candidate beats theta
    z = (theta - mu) / sigma
    # P(f(x) < theta) = Phi(z) since f(x) ~ N(mu, sigma^2)
    # Use scipy's norm.cdf for numerical stability
    from scipy.stats import norm
    prob = norm.cdf(z)
    
    # Score: higher probability of beating theta, lower cost is better
    scores = prob / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
