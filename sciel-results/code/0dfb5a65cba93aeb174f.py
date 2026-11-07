"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a valid array of zeros to avoid crash, though evaluator will kill anyway
        return np.zeros_like(mu)

    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Strategy: Entropy Maximization Proxy + UCB Blend ---
    # The directive suggests using log(sigma^2) to target regions of highest information gain.
    # This forces the sampler to map the landscape efficiently, especially useful for
    # multimodal or complex functions like Beale, where pure exploitation (EI) or
    # simple confidence bounds (UCB) might get stuck in local minima or over-explore
    # already known noisy regions without gaining structural insight.
    
    # 1. Compute Entropy Proxy: log(sigma^2) = 2 * log(sigma)
    # This score is high where uncertainty is high.
    # We scale it to be comparable to the other terms.
    # The scale of log(sigma) depends on the function's noise level and length scales.
    # A simple approach is to normalize it relative to the current distribution of sigma.
    log_sigma_sq = 2.0 * np.log(sigma_safe)
    
    # To make the entropy term comparable to the exploitation term (-mu),
    # we can normalize log_sigma_sq to have a similar range or magnitude.
    # However, direct addition can be tricky if scales differ significantly.
    # Let's look at the UCB term: -mu + beta * sigma.
    # The sigma term in UCB is linear in sigma.
    # The entropy term is logarithmic.
    
    # Alternative interpretation:
    # "Entropy maximization" in BO often refers to minimizing the expected posterior variance
    # over the domain, which is equivalent to maximizing the log-determinant of the kernel matrix
    # or approximating it with sum of log variances.
    # Here we only have point-wise sigma.
    # Scoring by log(sigma^2) directly encourages visiting points with high variance.
    # This is a strong exploration signal.
    
    # Let's blend this with a standard exploitation signal.
    # Exploitation: -mu (we want low mean)
    # Exploration (UCB-style): beta * sigma
    # Exploration (Entropy-style): gamma * log(sigma^2)
    
    # The inspiration 1 (score 6.79) used pure UCB with beta=5.0.
    # The current program (score 1.87) used EI + 0.5*sigma.
    # The directive asks for a structural change towards entropy/log-variance.
    
    # Let's try a score that combines:
    # 1. A strong exploitation term: -mu
    # 2. An entropy-based exploration term: gamma * log(sigma^2)
    # 
    # Why log(sigma^2) instead of sigma?
    # log(sigma^2) grows slower than sigma for very large sigma, preventing the algorithm
    # from being dominated by a single region of extreme noise (if any exists).
    # It also emphasizes the *relative* increase in information.
    
    # We need to tune gamma.
    # If we use pure log(sigma^2), we might ignore the mean entirely, which is bad.
    # So we combine: Score = -mu + gamma * log(sigma^2)
    
    # To make the scales comparable, we can normalize the log term.
    # Let's compute the mean and std of the log term across the candidates?
    # That might be overkill and introduce noise.
    # Let's try a fixed gamma first.
    # Typical sigma values in standardized GP outputs are O(1).
    # log(1^2) = 0.
    # If sigma = 2, log(4) ~ 1.38.
    # If sigma = 0.5, log(0.25) ~ -1.38.
    # So the range is roughly +/- 1.4 for sigma in [0.5, 2].
    # The range of -mu depends on f_best. If f_best is 10, mu might be 0..10.
    # So -mu is much larger in magnitude.
    
    # Therefore, we need a significant gamma or we need to scale mu.
    # However, mu is in objective units.
    
    # Let's try a different structural approach inspired by "Information Gain".
    # The mutual information between the observed function value at a point x and the
    # global minimum is proportional to the variance at x.
    # So maximizing sigma^2 (or log sigma^2) is a valid proxy for information gain.
    
    # Let's blend UCB and Entropy.
    # UCB: -mu + beta * sigma
    # Entropy: gamma * log(sigma^2)
    
    # Let's try: Score = -mu + beta * sigma + gamma * log(sigma^2)
    # This adds an extra exploration boost in high-variance regions beyond the linear UCB term.
    
    beta = 2.0  # Moderate UCB exploration
    gamma = 1.0 # Entropy exploration weight
    
    ucb_term = -mu + beta * sigma_safe
    entropy_term = gamma * np.log(sigma_safe ** 2)
    
    scores = ucb_term + entropy_term
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
