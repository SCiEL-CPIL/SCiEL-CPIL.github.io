"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: The evaluator kills after 330s.
# We are not doing a loop inside acquisition, so this is mostly a safety check 
# in case the GP evaluation or array operations are unexpectedly slow on some systems.
# However, the acquisition function itself is vectorized and should be fast.
# We will include a simple check to ensure we don't hang if something goes wrong, 
# though standard numpy/scipy ops on 2000 points are millisecond-scale.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Standard EI uses f_best as the threshold.
    # However, as f_best gets very close to the true minimum (0 for these functions),
    # the probability of improvement P(f(x) < f_best) becomes very small for most points,
    # and EI values shrink. This can cause the optimizer to stall or become overly greedy
    # in areas where the GP is uncertain but the mean is slightly above f_best.
    
    # To mitigate this "ratchet effect" and maintain exploration, we use a slightly
    # relaxed threshold. Instead of improving upon f_best, we aim to improve upon 
    # f_best - epsilon. This effectively adds a small exploration bonus.
    # Alternatively, we can just use standard EI but ensure it doesn't collapse entirely.
    # A common robust variant is to use a small epsilon in the improvement calculation.
    
    # Let's stick to a principled approach:
    # EI(x) = E[max(f_best - f(x), 0)]
    # Let Z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)  if sigma > 0
    # EI = 0 if sigma == 0 and mu >= f_best
    # EI = f_best - mu if sigma == 0 and mu < f_best
    
    # To avoid the collapse, we can use a "pseudo-best" that is slightly worse than f_best.
    # But this is heuristic. A more robust standard approach for cost-aware BO is 
    # simply EI / cost.
    
    # Let's try a hybrid: Use EI but with a floor on the exploration term?
    # Actually, the inspiration 3 mentioned "ratchet effect" and replacing it.
    # The current program uses TS. TS is stochastic. 
    # A deterministic EI is often more stable.
    
    # Let's calculate standard EI.
    # If sigma is very small, we should avoid numerical issues.
    eps = 1e-12
    
    # Z-score for the improvement
    # We are minimizing, so improvement is f_best - f(x)
    # Z = (f_best - mu) / sigma
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (f_best - mu) / sigma
    
    # Calculate EI
    # Phi(z) is CDF, phi(z) is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = np.zeros_like(mu)
    
    # Where sigma is significant
    mask = sigma > eps
    ei[mask] = (f_best - mu[mask]) * Phi_z[mask] + sigma[mask] * phi_z[mask]
    
    # Where sigma is negligible (deterministic region)
    # If mu < f_best, we have found a better point, EI is f_best - mu
    # If mu >= f_best, EI is 0
    ei[~mask] = np.maximum(f_best - mu[~mask], 0.0)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-awareness:
    # We want to maximize the expected improvement per unit cost.
    # This encourages finding improvements in cheap regions.
    scores = ei / cost
    
    # To prevent the "ratchet" where EI becomes zero everywhere and the algorithm
    # gets stuck, we can add a small exploration term based on uncertainty,
    # but weighted by the probability of improvement? 
    # The inspiration 1 (score 1.0) likely used a robust EI or UCB.
    # Let's stick to EI/cost as it is the most direct cost-aware metric.
    
    # However, if f_best is very close to 0, EI might be tiny.
    # Normalizing EI by the scale of the problem might help?
    # The problem functions have minima at 0 or -3.86.
    # The scale of f varies.
    
    # Let's add a small UCB-like exploration term to the EI to prevent stagnation.
    # Exploration bonus: sigma * beta
    # If we just use EI, we might miss regions where mu is slightly worse than f_best 
    # but sigma is large (potential for big improvement). EI captures this via the 
    # sigma * phi(z) term, but if z is very negative (mu >> f_best), phi(z) is tiny.
    # UCB: mu - beta * sigma (for minimization, we want low mu and high sigma? No, 
    # UCB for minimization is usually mu - beta * sigma is for maximization of -f.
    # For minimization, UCB is often defined as mu - beta * sigma is incorrect.
    # Standard UCB for maximization: mu + beta * sigma.
    # For minimization: mu - beta * sigma.
    # But we want to find low values. So we want candidates with low mu and high sigma?
    # Actually, high sigma means uncertainty. Low mu means current best guess is good.
    # We want to explore where the lower bound of the confidence interval is low.
    # Lower bound: mu - beta * sigma.
    # So an exploration score could be - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's blend EI and UCB?
    # Score = (EI / cost) + alpha * (sigma / cost) ?
    # This adds exploration.
    
    # Given the score 0.8363 for TS, and 1.0 for the best, let's try a pure, 
    # well-scaled EI/cost. If EI is too small, the relative differences between 
    # candidates might be drowned out by noise in the GP approximation.
    
    # Let's normalize EI by the range of the function? We don't know the range.
    # But we know the minimum is 0 (or -3.86).
    # f_best is the current best.
    
    # Let's try: scores = ei / cost
    # If this fails, we might need to add exploration.
    
    # To be safe against the ratchet, let's add a small, constant exploration bonus 
    # proportional to sigma/cost, but only if EI is very small?
    # No, let's keep it simple first. The "ratchet" is often solved by using 
    # a slightly higher threshold for f_best in EI calculation, e.g., f_best * (1 + 1e-4).
    # Or simply f_best + 1e-4.
    
    # Let's use a slightly relaxed f_best to encourage exploration.
    # If f_best is 0.001, we look for improvements over -0.001 (i.e., f < -0.001 is not possible if min is 0).
    # Wait, for minimization, if min is 0, f_best is positive.
    # We want to find f < f_best.
    # If we use threshold = f_best * 0.9 (for positive f_best), we require 10% relative improvement.
    # This makes EI larger because the "bar" is lower (easier to beat).
    # This effectively increases exploration.
    
    # Let's try this: threshold = f_best * 0.9 if f_best > 0 else f_best - 1e-4
    # But for Hartmann, min is -3.86. f_best will be negative.
    # If f_best = -3.8, threshold = -3.8 * 0.9 = -3.42.
    # We want f < -3.42. This is a much lower bar than f < -3.8.
    # This will generate larger EI values for points that are "good enough", 
    # encouraging exploration of the landscape rather than just local polishing.
    
    # Let's implement this adaptive threshold.
    if f_best > 0:
        threshold = f_best * 0.9
    else:
        # For negative minima, we want to allow for some slack.
        # If f_best is -3.86, we want to explore.
        # threshold = f_best - 0.1? 
        # Let's use a relative slack based on the magnitude.
        threshold = f_best - abs(f_best) * 0.1 - 1e-4
        
    # Recalculate EI with the relaxed threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        z_relaxed = (threshold - mu) / sigma
    
    phi_z_relaxed = norm.pdf(z_relaxed)
    Phi_z_relaxed = norm.cdf(z_relaxed)
    
    ei_relaxed = np.zeros_like(mu)
    
    mask = sigma > eps
    ei_relaxed[mask] = (threshold - mu[mask]) * Phi_z_relaxed[mask] + sigma[mask] * phi_z_relaxed[mask]
    ei_relaxed[~mask] = np.maximum(threshold - mu[~mask], 0.0)
    
    ei_relaxed = np.maximum(ei_relaxed, 0.0)
    
    scores = ei_relaxed / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
