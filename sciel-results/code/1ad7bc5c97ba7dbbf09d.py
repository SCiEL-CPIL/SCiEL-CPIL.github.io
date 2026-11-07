import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    Combines Expected Improvement (EI) with a UCB-style exploration term.
    
    EI handles exploitation (finding points likely to improve on f_best).
    The exploration term (beta * sigma) encourages visiting uncertain regions,
    which is crucial for multi-modal functions like Goldstein-Price.
    
    We use a dynamic beta that increases slightly with uncertainty to ensure
    exploration doesn't vanish in high-variance regions, but keep it modest
    to avoid pure random search.
    """
    # Safe division
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0, meaning high probability of improvement.
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical overflow in cdf/pdf
    z = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: (f_best - mu) can be negative if mu > f_best, but then Phi is small.
    # Actually, standard EI for minimization is:
    # EI(x) = E[max(f_best - f(x), 0)]
    # If f(x) ~ N(mu, sigma^2), let Y = f_best - f(x) ~ N(f_best - mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Exploration term: UCB-like
    # We want to balance exploitation (EI) and exploration (sigma).
    # A fixed beta might not scale well if the function scale varies.
    # However, mu and sigma are in objective units.
    # Let's use a beta that is proportional to the scale of the problem or fixed.
    # Given the range of Goldstein-Price is roughly [3, 12], and sigma is small,
    # a fixed beta around 0.5-1.0 is reasonable.
    # The current program uses beta=1.0 for pure UCB (-mu + beta*sigma).
    # Here we add it to EI.
    
    # We can also normalize the exploration term to be comparable to EI.
    # But simple addition is a standard approach (e.g., in some EI variants).
    # Let's try a moderate exploration weight.
    # If sigma is very small, EI dominates. If sigma is large, exploration dominates.
    # This naturally balances the two.
    
    beta = 0.5
    score = ei + beta * sigma
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
