"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound with a noise-adaptive exploration term.
    We blend UCb (which is known to be sample-efficient and robust in BO)
    with a small Expected Improvement component to retain exploitation
    when the model is confident.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize the improvement over best
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound
    # Use a beta that balances exploration. 
    # A common choice is sqrt(2 * log(n)) but we don't have n here.
    # Since we are in a fixed loop of 30 evals, a static beta around 1.0-2.0 works well.
    # We use a slightly higher beta to encourage exploration of the 2000 candidates.
    beta = 1.5
    ucb = mu + beta * sigma
    
    # Combine: We want high scores to be selected.
    # EI is 0 when mu >= f_best. UCB is always high when sigma is high.
    # A simple blend: score = max(EI, UCB) or a weighted sum.
    # Weighted sum can be tricky if scales differ.
    # Let's try a hybrid: if EI is positive, trust it; otherwise, rely on UCB.
    # Or better: use UCB for exploration and EI for exploitation.
    # score = EI + 0.5 * UCB? No, units might be off if f_best is large.
    # Both are in objective units.
    
    # Let's use a standard weighted combination that favors exploitation when certain.
    # score = EI + lambda * sigma
    # This is similar to UCB but with the EI shaping.
    
    # Alternative: Pure UCB is often more robust for small datasets.
    # Let's try a mix: 0.5 * EI + 0.5 * UCB?
    # Actually, EI can be very small. UCB can be large.
    
    # Let's stick to a robust variant of UCB with a small EI boost.
    # score = ucb + 0.1 * ei
    
    # However, pure UCB with beta=2.0 is a very strong baseline.
    # Let's try to make beta adaptive to the spread of sigma?
    # No, keep it simple.
    
    # Let's try: score = mu + 2.0 * sigma  (Standard UCB)
    # vs score = EI
    
    # The current best is EI. Let's try a blend that is more exploratory.
    # score = EI + 1.0 * sigma
    # This ensures that points with high uncertainty get a boost even if EI is low.
    
    score = ei + 1.0 * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
