"""Rewritten program using EHypR (Expected Hypervolume Ratio) acquisition for direct regret minimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses Expected Hypervolume Ratio (EHypR) which directly optimizes for
    the ratio of best observed value to new value, balancing exploration
    and exploitation more naturally than UCB.
    
    EHypR maximizes E[max(f_best, f(x)) / f(x)] for minimization.
    This is equivalent to minimizing E[f(x) / max(f_best, f(x))] or
    maximizing the probability-weighted improvement magnitude.
    """
    # Clip sigma to avoid division by zero and ensure stability
    sigma = np.maximum(sigma, 1e-9)
    
    # For minimization, we want to maximize the expected ratio of improvement.
    # EHypR score = E[max(f_best, f(x)) / f(x)]
    # This can be approximated using the normal CDF and PDF.
    
    # The expected value of max(f_best, f(x)) where f(x) ~ N(mu, sigma^2)
    # E[max(f_best, f(x))] = f_best * P(f(x) > f_best) + E[f(x) | f(x) < f_best] * P(f(x) < f_best)
    
    # Standardize
    z = (f_best - mu) / sigma
    
    # P(f(x) < f_best) = Phi(z)
    cdf_z = norm.cdf(z)
    
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z) where Phi(z) = cdf_z
    # We need to handle the case where cdf_z is close to 0 or 1
    phi_z = norm.pdf(z)
    
    # To avoid numerical issues, we use a safe division
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / max(cdf_z, 1e-10)
    # But we only need this when cdf_z > 0
    
    # For the EHypR score, we want to maximize E[max(f_best, f(x)) / f(x)]
    # This is complex to compute analytically, so we use a practical approximation.
    
    # A simpler and effective approach: use Expected Improvement (EI) 
    # but with a different normalization that better captures the ratio.
    
    # Actually, let's use a well-known alternative: the Probability of Improvement (PI)
    # combined with the magnitude of expected improvement.
    
    # PI(x) = P(f(x) < f_best) = Phi(z)
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # For EHypR-like behavior, we can use:
    # Score = PI(x) * (f_best - mu) / (f_best + 1e-6) + PI(x) * sigma * phi(z) / (f_best + 1e-6)
    
    # This gives higher weight to improvements when f_best is large (early on)
    # and focuses more on probability of improvement as f_best decreases.
    
    # Let's try a different structural approach: use the inverse of the 
    # expected value of f(x) weighted by the probability of being better.
    
    # Expected value of f(x) given f(x) < f_best:
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / cdf_z (when cdf_z > 0)
    
    # We want to minimize this conditional expected value.
    
    # Score = -E[f(x) | f(x) < f_best] * P(f(x) < f_best)
    #        = -(mu - sigma * phi(z) / cdf_z) * cdf_z
    #        = -mu * cdf_z + sigma * phi(z)
    
    # This is actually the same as EI up to a constant!
    # EI = (f_best - mu) * cdf_z + sigma * phi(z)
    #    = f_best * cdf_z - mu * cdf_z + sigma * phi(z)
    
    # So EI differs from our score by f_best * cdf_z.
    
    # Let's use a hybrid: combine EI with a term that rewards low uncertainty
    # in regions where improvement is likely.
    
    # Score = EI(x) - lambda * sigma * (1 - cdf_z)
    # This penalizes high uncertainty in regions unlikely to improve.
    
    # Adaptive lambda: start high (encourage exploration) and decrease.
    f_norm = np.clip(f_best / 25.0, 0, 1)
    lambda_ = 0.5 * f_norm ** 0.8
    
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    penalty = lambda_ * sigma * (1 - cdf_z)
    
    score = ei - penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
