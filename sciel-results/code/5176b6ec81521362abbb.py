"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Robust UCB with adaptive exploration weight based on the spread of the posterior.
    # Weierstrass 1D is highly multimodal, so we want a balance that favors exploration
    # when uncertainty is high relative to the current best, but exploits when it isn't.
    
    # Ensure sigma is strictly positive to avoid division by zero.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute the z-score for standard normal CDF/PDF.
    # This is the standard EI formulation, but we use UCB instead.
    # UCB: mu + kappa * sigma
    # We want to minimize, so higher score = better (lower expected value).
    # So score = -mu + kappa * sigma? No, higher score means "evaluate next".
    # If we want to minimize f, we want to pick points where f is likely low.
    # So a good acquisition function for minimization is something like -mu + kappa * sigma.
    # Or, we can define the score as the expected improvement in the *negative* direction?
    # Let's stick to the convention: higher score = more desirable to evaluate.
    # If we are minimizing, a point with low mu and high sigma is good.
    # So score = -mu + kappa * sigma is a reasonable UCB-like score for minimization.
    
    # Let's try a dynamic kappa.
    # A common heuristic is kappa = 2.5 or so.
    # But we can also use the "expected improvement" idea but adapted for minimization.
    # EI_min = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # This is exactly what the original code does. The original code scored 1.0.
    # The observation says UCB replaced EI and improved the score to 2.155.
    # So let's implement a robust UCB.
    
    # Let's use a fixed kappa first, maybe 2.0.
    # score = -mu + kappa * sigma
    
    # To make it more robust, we can normalize the scores or use a more sophisticated kappa.
    # Let's try a kappa that depends on the number of evaluations or the spread of mu.
    # However, we don't have the number of evaluations directly, only f_best.
    # We can estimate the "progress" by looking at how low f_best is.
    # But for a 1D function, a fixed kappa often works well.
    
    # Let's try a slightly higher kappa to encourage exploration in the early stages.
    # Since we don't know the stage, let's use a value that balances well.
    # kappa = 2.0 is a good start.
    
    kappa = 2.0
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
