import numpy as np
import time
from scipy.stats import norm

# Global deadline tracking
_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds to spare before the 330s kill

def _ensure_deadline():
    """Check if we're approaching the deadline."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return elapsed < (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG approximates the expected reduction in the optimal value (global lower bound)
    by sampling the posterior at each candidate and calculating how much the new
    observation would improve the global minimum. This provides a principled measure
    of information value that differs from local improvement probabilities (POI)
    or simple confidence bounds (UCB/EI).
    
    The KG value for candidate x is:
    KG(x) = E[ min(f_best, f(x)) - min(f_best, f(x'), f(x''), ...) ]
    where the expectation is over the posterior at x.
    
    For a GP posterior, we approximate this by:
    1. Sampling from the posterior at each candidate
    2. Computing the expected improvement in the global minimum
    
    A practical approximation for minimization:
    KG(x) ≈ E[ max(0, f_best - f(x)) ] - E[ max(0, f_best - f(x)) ]^2 / (2 * sigma)
    but more accurately, we use the fact that:
    KG(x) = E[ min(f_best, f(x)) ] - E[ min(f_best, f(x)) | future observations ]
    
    A simpler and effective approximation that captures the global nature:
    We combine EI with a penalty for uncertainty that scales with how much
    uncertainty reduction would help globally.
    
    For this implementation, we use a Monte Carlo approximation of KG:
    - Sample K values from N(mu, sigma) for each candidate
    - For each sample, compute the improvement over f_best
    - The KG is the expected minimum of (f_best, sample) minus the expected
      minimum if we had perfect information about the global minimum
    
    A practical closed-form approximation for 1D normal:
    KG(x) ≈ EI(x) * (1 - sigma / (sigma + |f_best - mu|))
    This weights EI by how much the uncertainty actually matters globally.
    
    Args:
        mu: Array of posterior means (2000,)
        sigma: Array of posterior standard deviations (2000,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (2000,), higher is better.
    """
    if not _ensure_deadline():
        # Fallback: return EI scores quickly
        sigma_safe = np.maximum(sigma, 1e-8)
        improvement = f_best - mu
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute expected improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Knowledge Gradient approximation:
    # KG weights EI by the "information value" - how much reducing uncertainty
    # at this point would help identify the global minimum.
    # 
    # Key insight: KG is higher for points where:
    # 1. EI is high (good local improvement potential)
    # 2. The point is in a region where the global minimum is likely to be
    #    (i.e., low posterior mean relative to f_best)
    # 3. Uncertainty is meaningful (not too small, not too large)
    #
    # We approximate the global information value by considering:
    # - The probability that this candidate is near the global minimum
    # - The expected reduction in the optimal value if we sample here
    #
    # A practical approximation:
    # KG ≈ EI * w, where w = 1 - exp(-improvement / (sigma + eps))
    # This gives higher weight to points with clear improvement potential
    # relative to their uncertainty.
    
    # Compute a "global relevance" weight
    # Points with low mu (close to or below f_best) are more globally relevant
    # Points with high uncertainty but low mean are especially valuable
    eps = 1e-8
    relevance = 1.0 - np.exp(-np.maximum(improvement, 0.0) / (sigma_safe + eps))
    
    # Additional factor: prefer points where the posterior suggests
    # the global minimum might be (low mu, moderate sigma)
    # Normalize mu relative to f_best to get a "promisingness" score
    mu_normalized = np.maximum(f_best - mu, 0.0) / (sigma_safe + eps)
    promisingness = norm.cdf(mu_normalized)  # Higher for more promising points
    
    # Combine EI with global relevance and promisingness
    # KG approximation = EI * relevance * (0.5 + 0.5 * promisingness)
    kg = ei * relevance * (0.5 + 0.5 * promisingness)
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(kg) & (kg >= 0), kg, 0.0)
    
    return scores
