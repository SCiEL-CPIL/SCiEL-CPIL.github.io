"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function for Bayesian optimization.
    
    Strategy:
    - Early phase (high uncertainty / few observations implied by large sigma relative to f_best):
      Use UCB (Upper Confidence Bound) with a high exploration constant to encourage 
      exploration of distant basins in the multimodal Branin function.
    - Late phase (low uncertainty / many observations implied by small sigma):
      Switch to EI (Expected Improvement) to exploit local minima.
    
    Since we don't have the iteration count directly, we use a heuristic based on the 
    magnitude of sigma relative to the current best value or a fixed threshold. 
    However, a more robust proxy for "early vs late" in a fixed 30-step loop without 
    state is difficult. 
    
    Alternative Strategy: 
    Use a weighted combination of UCB and EI. 
    UCB encourages exploration of areas with high sigma. 
    EI encourages exploitation of areas with low mu.
    
    Let's use a dynamic beta for UCB that decays? No, stateless.
    
    Let's look at the inspiration: "UCB with fixed beta=2.5" was mentioned but 
    the lesson says "UCB (beta=1) provides stronger exploration... Measured: 0.6571 -> 1.421".
    The current program is EI (score 1.0).
    
    A materially different approach is using Probability of Improvement (PI) or 
    a modified UCB.
    
    Let's try a UCB variant with a beta that is scaled by the current uncertainty 
    or just a well-tuned fixed beta that favors exploration more than standard EI.
    
    Actually, for Branin, which has multiple global minima, exploration is key.
    Standard EI often gets stuck.
    
    Let's implement UCB: mu - beta * sigma.
    The lesson suggests beta=1 improved over EI (1.0 -> 1.421).
    Let's try a slightly higher beta to ensure we don't miss the other basins, 
    e.g., beta = 1.5 or 2.0.
    
    However, pure UCB can be too exploratory late in the run.
    
    Let's try a hybrid: 
    Score = EI + alpha * (f_best - mu - sigma)  -- this is basically UCB.
    
    Let's stick to the lesson's success: UCB with beta=1 was good. 
    Let's try beta=2.0 to be safer for multimodality, or perhaps a 
    "Exploration-Exploitation" balance via a weighted sum of UCB and EI.
    
    Let's try: 0.5 * UCB(beta=2) + 0.5 * EI
    This ensures we have both the drive to explore (via UCB) and the 
    precise targeting of improvements (via EI).
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to minimize, so we look for low mu and high sigma.
    # UCB_min = mu - beta * sigma
    # To make it an "acquisition score" (higher is better), we can use:
    # Score = -mu + beta * sigma  (or similar)
    # But EI is already a score. Let's use a standard UCB formulation that 
    # can be combined.
    # A common form for minimization acquisition is:
    # a(x) = mu(x) - beta * sigma(x)  (we want to minimize this value)
    # But our interface requires "higher meaning evaluate next".
    # So we should return -a(x) = -mu + beta * sigma.
    
    # Let's use beta = 1.5 for strong exploration.
    beta = 1.5
    ucb_score = -mu + beta * sigma_safe
    
    # Hybrid: Average of EI and scaled UCB?
    # EI values are usually smaller in magnitude than raw UCB scores if f_best is close to mu.
    # It is better to normalize or just pick one.
    
    # Given the lesson "UCB (beta=1) ... 1.421", let's try pure UCB with beta=2.0 
    # to be more aggressive in exploration, as Branin has 3 distinct basins.
    
    # Actually, let's look at the "materially different" requirement.
    # Current: EI.
    # New: UCB.
    
    # Let's use UCB with beta=2.0.
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
