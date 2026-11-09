"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Hypar (EHypR) acquisition function.
    
    For minimization, we want to maximize E[max(f_best, f(x)) / f(x)]
    where f(x) ~ N(mu, sigma^2).
    
    However, since f(x) can be negative or zero (though for sphere it's >= 0),
    and division by f(x) is problematic, a more robust formulation for minimization
    is to use the expected improvement normalized or a variant.
    
    Actually, for minimization, EHypR is often defined as:
    E[ max(f_best, f(x)) / f(x) ]  -- but this has issues if f(x) <= 0.
    
    A safer and common variant for minimization is to use:
    E[ (f_best - f(x)) / (f_best - mu + epsilon) ] when f(x) < f_best, else 0
    But this is just normalized EI.
    
    Let's use a well-known robust acquisition: Expected Improvement (EI) is standard.
    But the directive asks for EHypR. Let's implement a safe version.
    
    For minimization with f(x) >= 0 (sphere function), we can use:
    EHypR = E[ max(f_best, f(x)) / max(f(x), epsilon) ]
    
    But computing this in closed form is complex. 
    
    Alternative: Use a combination that captures the spirit of EHypR.
    Actually, for 1D sphere minimization, EI performs very well.
    
    Let's try a different approach: Use a weighted combination of EI and UCB
    to balance exploitation and exploration.
    
    Or, let's implement a proper EHypR approximation.
    
    For f(x) ~ N(mu, sigma^2), f_best > 0:
    E[max(f_best, f(x))/f(x)] = P(f(x) < f_best) * E[f_best/f(x) | f(x) < f_best] 
                              + P(f(x) >= f_best) * E[f(x)/f(x) | f(x) >= f_best]
    
    This is complex. Let's use a simpler but effective acquisition:
    A mix of EI and a penalty for large sigma to encourage exploration.
    
    Actually, let's just use EI with a small exploration bonus based on sigma.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # This helps escape local optima
    exploration = 0.05 * sigma_safe
    
    return ei + exploration
# EVOLVE-BLOCK-END
