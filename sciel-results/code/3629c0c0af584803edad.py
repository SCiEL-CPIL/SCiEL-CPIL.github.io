"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # 315s to leave 15s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient acquisition function for Bayesian optimization of Weierstrass 1D.
    
    The Knowledge Gradient (KG) computes the expected reduction in the best observed
    value if we were to evaluate at each candidate point. It explicitly models the
    global optimality of the search, directing evaluations toward regions that maximize
    the probability of containing the global minimum.
    
    For a single-objective minimization with a GP posterior N(mu, sigma^2):
    KG(x) = E[max(f_best, f(x))] - max(f_best, E[f(x)])
    
    This can be computed analytically:
    Let z = (f_best - mu) / sigma
    KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    We return negative KG scores so that the candidate with the highest expected
    improvement has the highest score (since the evaluator picks the highest score).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores where higher means evaluate next.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        # Fallback: use LCB with small kappa for speed
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is positive and numerically stable
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient computation:
    # KG(x) = E[max(f_best, f(x))] - max(f_best, mu(x))
    # 
    # For f(x) ~ N(mu, sigma^2):
    # E[max(f_best, f(x))] = f_best * P(f(x) >= f_best) + E[f(x) | f(x) < f_best] * P(f(x) < f_best)
    # 
    # Let z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z)
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z)
    # 
    # So E[max(f_best, f(x))] = f_best * (1 - Phi(z)) + (mu - sigma * phi(z) / Phi(z)) * Phi(z)
    #                          = f_best - (f_best - mu) * Phi(z) - sigma * phi(z)
    # 
    # max(f_best, mu) = f_best if mu >= f_best, else mu
    # 
    # KG(x) = E[max(f_best, f(x))] - max(f_best, mu)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # Handle numerical issues: where Phi_z is very small, use asymptotic approximation
    # For very negative z, Phi(z) ~ 0, phi(z)/Phi(z) ~ -z
    # For very positive z, Phi(z) ~ 1, phi(z)/Phi(z) ~ 0
    
    # E[max(f_best, f(x))]
    # = f_best * (1 - Phi(z)) + (mu - sigma * phi(z) / Phi(z)) * Phi(z)
    # = f_best - (f_best - mu) * Phi(z) - sigma * phi(z)
    
    # But we need to be careful: when Phi(z) is very small, phi(z)/Phi(z) can be unstable
    # Use a more stable formulation
    
    # Alternative stable computation:
    # KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z)  when mu < f_best
    # KG(x) = sigma * phi(z)  when mu >= f_best (since max(f_best, mu) = f_best, and
    #         E[max(f_best, f(x))] - f_best = sigma * phi(z) for the part where f(x) > f_best... 
    #         actually let me re-derive)
    
    # Let me re-derive more carefully:
    # E[max(f_best, f(x))] where f(x) ~ N(mu, sigma^2)
    # = integral_{-inf}^{f_best} f_best * p(f) df + integral_{f_best}^{inf} f * p(f) df
    # = f_best * P(f < f_best) + E[f * I(f >= f_best)]
    # = f_best * Phi(z) + integral_{f_best}^{inf} f * p(f) df
    # 
    # For f ~ N(mu, sigma^2):
    # E[f * I(f >= f_best)] = mu * P(f >= f_best) + sigma * phi(z)
    #                        = mu * (1 - Phi(z)) + sigma * phi(z)
    # 
    # So E[max(f_best, f(x))] = f_best * Phi(z) + mu * (1 - Phi(z)) + sigma * phi(z)
    #                          = mu + (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # max(f_best, mu) = max(f_best, mu)
    
    # KG(x) = E[max(f_best, f(x))] - max(f_best, mu)
    #       = mu + (f_best - mu) * Phi(z) + sigma * phi(z) - max(f_best, mu)
    
    # Case 1: mu >= f_best
    #   max(f_best, mu) = mu
    #   KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    #   Note: when mu >= f_best, z <= 0, so Phi(z) <= 0.5
    #   (f_best - mu) is negative or zero, Phi(z) is positive, so first term is negative
    #   But sigma * phi(z) is positive
    #   Actually this should still be non-negative... let me check
    #   When mu = f_best, z = 0, KG = 0 + sigma * phi(0) = sigma / sqrt(2*pi) > 0
    #   Hmm, that doesn't seem right. If mu = f_best, the expected max is f_best + sigma*phi(0)*... 
    #   wait, let me just use the formula directly.
    
    # Case 2: mu < f_best
    #   max(f_best, mu) = f_best
    #   KG(x) = mu + (f_best - mu) * Phi(z) + sigma * phi(z) - f_best
    #         = (f_best - mu) * Phi(z) + sigma * phi(z) - (f_best - mu)
    #         = (f_best - mu) * (Phi(z) - 1) + sigma * phi(z)
    #         = -(f_best - mu) * (1 - Phi(z)) + sigma * phi(z)
    #         = (mu - f_best) * (1 - Phi(z)) + sigma * phi(z)
    
    # Let's just compute it directly using the formula:
    # E[max(f_best, f(x))] = mu + (f_best - mu) * Phi(z) + sigma * phi(z)
    
    E_max = mu + (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # max(f_best, mu)
    max_fb_mu = np.maximum(f_best, mu)
    
    # KG = E_max - max_fb_mu
    kg = E_max - max_fb_mu
    
    # Ensure KG is non-negative (numerical stability)
    kg = np.maximum(kg, 0.0)
    
    # We want higher KG to mean "evaluate next", so return KG directly
    # (higher KG = more expected improvement in the best value)
    scores = kg
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
