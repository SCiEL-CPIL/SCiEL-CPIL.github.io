import numpy as np
from scipy.stats import norm
import time

# Strategy: Greedy Batch-Independent (GBI) Acquisition
# 
# The current program uses a pointwise UCB score: -mu + kappa * sigma.
# This treats all 2000 candidates independently. In Bayesian Optimization,
# candidates that are close to each other (in the latent space defined by the GP)
# are highly correlated. Selecting one provides little new information if another
# nearby candidate is also selected (or already selected in a batch context, though
# here we pick one, the GBI idea helps by penalizing points that are "redundant"
# with the high-probability mass of the other candidates).
#
# However, a simpler and more robust structural change that aligns with the 
# "Greedy Batch-Independent" inspiration is to use an acquisition function that
# better balances exploration and exploitation than raw UCB.
#
# The "Cell findings" note that switching from UCB to EI improved score significantly 
# (1.519 -> 6.097). This suggests that the specific balance of exploration/exploitation 
# in EI is superior for this problem.
#
# Let's implement a robust Expected Improvement (EI) for minimization.
# For minimization, we want to improve upon f_best.
# Improvement I(x) = max(0, f_best - f(x))
# E[I(x)] = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
# where Phi is the CDF and phi is the PDF of the standard normal distribution.
#
# This is a standard, well-validated acquisition function.
# The current program uses UCB. The inspiration suggests a shift to a method that 
# accounts for correlation or uses a different balance. EI is a strong candidate.
#
# Additionally, we can add a small diversity term or use a "Batch EI" variant 
# if we were doing batch selection, but since we pick one, standard EI is the baseline.
#
# To make it materially different from the current UCB, we switch to EI.
# We also add a safety check for numerical stability.

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition score for minimization.
    
    Parameters:
    mu: array of shape (N,) posterior means
    sigma: array of shape (N,) posterior standard deviations
    f_best: float, the best observed value so far
    
    Returns:
    scores: array of shape (N,) EI scores (higher is better)
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu
    # If mu > f_best, improvement is negative, EI should be 0.
    # If mu < f_best, improvement is positive.
    
    # Z-score for the normal distribution
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If f_best - mu is negative, Phi(z) is small and the term is negative, 
    # but EI is defined as max(0, ...). The formula naturally handles this 
    # because if z is very negative, Phi(z) ~ 0 and phi(z) ~ 0, so EI ~ 0.
    # However, strictly speaking, EI is 0 when f_best - mu <= 0.
    
    # Using scipy.stats.norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical precision might cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
