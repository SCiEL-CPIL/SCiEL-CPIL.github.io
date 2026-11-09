"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive Upper Confidence Bound (UCB).
    
    mu: (N,) posterior mean
    sigma: (N,) posterior std
    f_best: best observed value so far
    
    Returns: (N,) scores, higher is better.
    """
    # Deadline guard: return zeros if we are running out of time (safety margin)
    # This is a static function, so we assume the caller manages the loop time,
    # but we can add a trivial check just in case, though usually not needed for a single call.
    # The prompt asks for an explicit deadline guard. Since we don't have the start time,
    # we assume this function is fast enough. However, to strictly follow the directive,
    # we would need to pass time or have a global. 
    # Given the signature is fixed, we cannot pass time. 
    # We will ensure the computation is O(N) and very fast.
    
    # Clip sigma to avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Adaptive UCB strategy:
    # In low-budget BO (30 evals), exploration is critical.
    # Standard UCB: score = mu + beta * sigma
    # For minimization, we want to find low values.
    # However, the acquisition function returns scores where HIGHER means "evaluate next".
    # So we want to select points that are likely to be low (good) or uncertain.
    
    # Let's think about the objective. We are minimizing f.
    # f_best is the lowest observed value.
    # A point x is attractive if:
    # 1. Its mean is low (exploitation)
    # 2. Its variance is high (exploration)
    
    # UCB for minimization usually looks like:
    # UCB(x) = mu(x) - beta * sigma(x)
    # But the return value must be such that higher is better.
    # So we can return: score = - (mu - beta * sigma) = -mu + beta * sigma
    
    # Let's determine beta.
    # A common adaptive beta is based on the number of iterations, but we don't have that.
    # Another approach: scale beta with the scale of the problem or the uncertainty of the best point.
    # Let's use a fixed beta that is aggressive enough for 30 steps.
    # With 2000 candidates, a beta of ~1.0 to 2.0 is often effective for 2D problems in ~30 steps.
    # Let's try an adaptive beta based on the current sigma at the best predicted point?
    # Or simply a well-tuned constant.
    
    # Inspiration 1 (score 1.09) used Pure UCB.
    # Let's use a UCB with a beta that balances exploration.
    # If we use beta = 1.0, it's a standard trade-off.
    # If we use beta = 2.0, it's more exploratory.
    
    # Let's look at the scale. mu and sigma are in objective units.
    # Ackley function values are roughly in [0, 10] initially, dropping to 0.
    # If f_best is around 5, and sigma is around 1, beta=1 gives a 1 unit exploration bonus.
    
    # To make it adaptive without iteration count:
    # We can estimate the "quality" of the current best.
    # But a simple, robust choice for low-budget 2D is a moderately high beta.
    # Let's try beta = 1.5.
    
    beta = 1.5
    
    # Score = -mu + beta * sigma
    # Higher score means lower mu (good) or higher sigma (good for exploration)
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
