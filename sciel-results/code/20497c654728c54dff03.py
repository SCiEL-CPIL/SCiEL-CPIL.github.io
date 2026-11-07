"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    per unit cost for Bayesian optimization.
    
    This approach maximizes the expected improvement in the objective function
    relative to the cost of evaluation. It is generally more robust and
    sample-efficient than UCB for problems with varying costs.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero in Z calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)

    # Calculate the improvement factor Z
    # For minimization, improvement is f_best - mu
    d = f_best - mu
    Z = d / sigma_safe

    # Calculate CDF and PDF of the standard normal distribution
    # Phi(Z) = P(X <= Z)
    # phi(Z) = PDF at Z
    phi_Z = norm.pdf(Z)
    Phi_Z = norm.cdf(Z)

    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # Note: If mu > f_best, d is negative, Z is negative.
    # Phi(Z) will be small, phi(Z) is positive.
    # The term (f_best - mu) * Phi(Z) will be negative * small positive = negative.
    # The term sigma * phi(Z) is positive.
    # However, standard EI is defined as E[max(0, f_best - f(x))].
    # The formula d * Phi(Z) + sigma * phi(Z) is non-negative for all Z.
    # Let's verify: If Z -> -inf, Phi -> 0, phi -> 0, d -> -inf. 
    # Limit of d * Phi(Z) as Z -> -inf is 0 (from negative side).
    # So EI is always >= 0.
    
    EI = d * Phi_Z + sigma_safe * phi_Z
    
    # Ensure EI is non-negative to handle numerical precision issues
    EI = np.maximum(EI, 0.0)

    # Cost-aware score: Maximize expected improvement per unit cost
    score = EI / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
