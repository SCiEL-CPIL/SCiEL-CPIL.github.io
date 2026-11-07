"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state to track elapsed time for safety
_start_time = time.time()
_DEADLINE = 300.0  # Seconds, keeping a safety margin below the 330s kill switch

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Combines the structural idea of UCB (linear balance of exploration/exploitation)
    with the refinement machinery of EI (non-linear, risk-averse scoring) via a hybrid approach.
    
    In high dimensions (6D), pure EI can be too greedy, while pure UCB with fixed beta
    might explore too much or too little depending on the scale of mu/sigma.
    
    We use a hybrid score:
    Score = EI + lambda * (UCB_component)
    
    However, a more robust and common "refinement" is to use an adaptive UCB parameter
    or a combination that emphasizes exploitation when we are close to the optimum and
    exploration when we are far.
    
    Let's look at the inspirations:
    - Inspiration 1 (UCB beta=1.0) scored 1.0441. This suggests that shifting from EI to UCB
      with beta=1.0 is an improvement.
    - The current program (EI) scored 1.0000.
    
    Strategy:
    1. Start with the UCB formulation which performed better: -mu + beta * sigma.
    2. Refine it by potentially adjusting beta or adding a small EI component to capture
       the non-linear improvement potential, or by normalizing the components.
    
    Actually, a very strong baseline is often just a well-tuned UCB or EI.
    Since UCB(1.0) beat EI, let's stick with UCB but perhaps tune the beta or ensure
    numerical stability.
    
    Another refinement: The "search" directive in the prompt's context might imply
    trying to find a better acquisition function. Since we can't search at runtime
    effectively without extra evaluations, we rely on a robust heuristic.
    
    Let's try a "Gaussian Process UCB" with a slightly higher beta to encourage more
    exploration in the 6D space, or stick to the proven 1.0.
    
    Wait, Inspiration 1 used beta=1.0 and got 1.0441.
    The current code uses EI.
    
    Let's combine:
    Use UCB as the base, but add a small EI term to help focus on improvements?
    Or, more simply, just use the UCB that worked.
    
    However, the directive says "COMBINE".
    Let's use a weighted sum of normalized EI and UCB?
    Normalization is tricky without knowing the scale.
    
    Alternative: Use EI but with a "tuning" parameter that effectively makes it more exploratory?
    EI is already balanced.
    
    Let's look at the "Gaussian" mention in Inspiration 1's strategy description.
    It says "replaces it with a Gaussian...". It likely means the GP prior.
    
    Let's try a hybrid:
    score = EI + alpha * (sigma)
    This is essentially EI with an added exploration bonus.
    
    Or, let's use the UCB formula but with a beta that is slightly larger, e.g., 1.5,
    to see if more exploration helps in 6D. But we don't know if that's better.
    
    The safest "combination" is to use the structure of the UCB (linear, fast, stable)
    and the refinement of ensuring we are capturing the "improvement" aspect.
    
    Actually, a very effective acquisition function is the "Expected Improvement" itself,
    but the fact that UCB(1.0) beat it suggests the EI implementation or the nature of EI
    was slightly suboptimal for this specific GP setup (Matern 5/2, ls=0.2).
    
    Let's implement UCB with beta=1.0 as the primary score, but include a time guard.
    To "combine" with the current EI program's robustness, we will ensure high numerical
    stability and handle edge cases (sigma near 0) carefully, similar to the current code's
    `np.maximum(sigma, 1e-8)`.
    
    Let's also consider that `f_best` is not used in UCB. This is a known downside: UCB
    doesn't directly target the current best. However, for minimization, minimizing the
    lower bound is a strong proxy.
    
    To incorporate `f_best` (and thus the "EI" aspect), we can use a "Probability of Improvement"
    or a modified EI.
    
    Let's try a score that is:
    Score = EI + beta * sigma
    This adds a linear exploration bonus to EI.
    If beta=0, it's EI.
    If we make EI small relative to sigma, it becomes UCB-like.
    
    Let's try beta=1.0 in this additive form?
    EI is roughly (f_best - mu) * Phi(z) + sigma * phi(z).
    For points with high sigma, EI is small (because f_best - mu is likely large positive, but Phi(z) is small? No, if mu is high, f_best-mu is negative? No, f_best is min.
    If mu is high (bad), f_best - mu is negative. z is negative. Phi(z) is small. EI is small.
    If mu is low (good), f_best - mu is positive. z is positive. Phi(z) is large. EI is large.
    So EI focuses on points likely to be better than f_best.
    
    UCB (-mu + beta*sigma) focuses on points that are low (low mu) or uncertain (high sigma).
    
    A combination:
    score = EI + 0.5 * sigma  # Add a small exploration bonus to EI
    
    Let's try this. It keeps the exploitation focus of EI (which uses f_best) but adds
    the linear exploration push of UCB.
    
    Parameters:
    mu: (2000,)
    sigma: (2000,)
    f_best: float
    
    """
    # Check time budget
    current_time = time.time()
    if current_time - _start_time > _DEADLINE:
        # Return a neutral score to avoid timeout kill, though this should rarely happen
        # for 2000 points.
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    
    # Safety: sigma should be non-negative.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    # Use scipy for stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Calculate UCB component (exploration bonus)
    # We add a fraction of sigma to encourage exploration.
    # The inspiration UCB(1.0) scored 1.0441. EI scored 1.0.
    # If we just add 1.0 * sigma, we might over-explore.
    # Let's try a smaller weight, e.g., 0.5, to blend the two.
    # Or, we can use the UCB score directly if we think EI is too weak.
    # But the directive is to combine.
    
    # Let's try: score = EI + 0.5 * sigma
    # This is a "boosted EI".
    
    alpha = 0.5
    scores = ei + alpha * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores

# EVOLVE-BLOCK-END
