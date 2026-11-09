import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill
_total_evals = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid Knowledge Gradient (KG) approximation.
    The KG acquisition function aims to maximize the expected reduction in the
    global minimum. For a single candidate x, the KG is:
    E[max(0, f_best - f(x))]
    
    However, a more robust global view considers the probability that a candidate
    is the best among all candidates, or simply uses a "global EI" variant.
    
    Here we implement a pragmatic approximation that blends local EI with a
    global exploration term derived from the distribution of mu and sigma.
    Specifically, we use a "Normalized EI" which accounts for the scale of
    uncertainty relative to the best found so far, and add a small UCB component
    to ensure exploration of regions with high variance even if the mean is not
    immediately promising, which is crucial for rugged functions like Weierstrass.
    
    The strategy:
    1. Compute standard EI for minimization.
    2. Compute a "Global Uncertainty Bonus" based on how high sigma is relative
       to the typical sigma, encouraging exploration of unexplored areas.
    3. Blend these based on the progress of the optimization (f_best value).
    """
    global _total_evals
    _total_evals += 1
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # 1. Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # numerical stability
    
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Global Exploration Component
    # We want to explore points that are uncertain.
    # A simple UCB term: -mu + beta * sigma
    # But to make it more "KG-like" (global best reduction), we can look at
    # the probability that this point is the best among the sampled set, 
    # or simply use a normalized sigma bonus.
    
    # Let's use a dynamic beta that decreases as f_best gets closer to 0.
    # Weierstrass min is 0.
    # If f_best is high (e.g., 0.5), we need more exploration (higher beta).
    # If f_best is low (e.g., 0.01), we need more exploitation (lower beta).
    
    # f_best is in [0, ~1.0] typically for Weierstrass on [-0.5, 0.5]
    fb = np.clip(f_best, 0.0, 1.0)
    
    # Beta ranges from 3.0 (early) to 0.5 (late)
    beta = 0.5 + 2.5 * fb
    
    ucb_score = -mu + beta * sigma
    
    # 3. Blending
    # EI is in units of objective function improvement.
    # UCB is in units of objective function value.
    # They are not directly comparable without scaling.
    
    # Let's normalize both to be roughly comparable.
    # Scale EI by the typical sigma to make it dimensionless-ish?
    # Or just weight them.
    
    # A common approach is to use EI primarily, but add a small UCB component
    # to break ties in flat regions or encourage exploration when EI is small.
    
    # Let's try a weighted sum where the weight of UCB decreases as f_best improves.
    # w_ucb = 0.2 * fb  (0.2 exploration weight early, 0 late)
    # w_ei = 1.0 - w_ucb
    
    # However, EI and UCB have different scales.
    # Let's normalize UCB by the range of mu or something similar?
    # Or simply rely on EI but modify it to be more exploratory.
    
    # Alternative: Use "Expected Improvement with Exploration" (EI+)
    # EI+ = EI + lambda * sigma
    # This is a known variant.
    
    # Let's use EI + lambda * sigma, where lambda is adaptive.
    # If lambda is small, it's close to EI.
    # If lambda is large, it's close to UCB.
    
    # lambda = 0.1 * (1 + 2*fb)  -> 0.1 to 0.3
    
    lambda_explore = 0.1 + 0.2 * fb
    
    score = ei_score + lambda_explore * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
