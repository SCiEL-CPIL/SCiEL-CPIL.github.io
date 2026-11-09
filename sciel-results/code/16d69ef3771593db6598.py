"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    The parameter beta is chosen to balance exploration and exploitation.
    A common heuristic is beta = 1 + step_count, but since we don't have step_count,
    we use the simple regret (f_best - min_possible) as a proxy for how much 
    improvement is left. If f_best is far from the known minimum (-3.32237 for Hartmann 6d),
    we explore more (higher beta). If f_best is close, we exploit more (lower beta).
    
    However, the problem states the minimum is -3.32237. We can use this to scale beta.
    Let's define a target minimum m_min = -3.32237.
    Regret R = f_best - m_min.
    If R is large, we want to explore. If R is small, we want to exploit.
    
    A simple adaptive beta:
    beta = 2.0 + 4.0 * np.exp(-0.5 * R)  # Just a guess, maybe too complex.
    
    Let's stick to a simpler, robust heuristic often used in practice:
    beta = 1.0 + 2.0 * (f_best - np.min(mu)) / (np.max(mu) - np.min(mu) + 1e-9)
    This is relative to the current batch's uncertainty.
    
    Actually, a standard robust choice for LCB in BO is beta = 1.0 or beta = 2.0.
    Let's try a slightly adaptive approach based on the gap to the theoretical minimum.
    Theoretical min for Hartmann 6d is approx -3.32237.
    
    Let's use a fixed but slightly higher beta than standard EI implies, or just a good LCB.
    EI is equivalent to LCB with a specific beta that varies by point.
    LCB with constant beta is often more stable.
    
    Let's try beta = 2.0. This encourages exploration more than standard EI (which is like beta~1 in some regimes).
    To be safer and potentially improve over the seed (which is EI), let's use a moderate exploration parameter.
    
    Formula: score = mu - beta * sigma
    """
    # Ensure sigma is non-negative and finite
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clean up any potential numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta:
    # If f_best is very high (far from optimum), explore more.
    # If f_best is low (close to optimum), exploit more.
    # Hartmann 6d min is ~ -3.32.
    # Initial random points might be around 0 or positive.
    # Let's say if f_best > -2.0, we are early -> beta = 3.0
    # If f_best < -3.0, we are late -> beta = 1.5
    # Linear interpolation between these ranges?
    
    # Simpler: Just use a fixed beta that is slightly more exploratory than EI's implicit average.
    # EI maximizes E[f(x) - f_best | f(x) < f_best].
    # LCB minimizes mu - beta*sigma.
    # For standard normal, EI ~ (f_best-mu)Phi(z) + sigma phi(z).
    # If we pick beta = 1.0, it's very exploitative.
    # If we pick beta = 2.0, it's balanced.
    
    # Let's try a dynamic beta based on the regret relative to the known global min.
    global_min = -3.32237
    regret = f_best - global_min
    
    # If regret is large (e.g., > 1.0), use higher beta.
    # If regret is small (e.g., < 0.1), use lower beta.
    # beta = 1.0 + 2.0 * np.exp(-regret)
    # If regret=0, beta=3.0. If regret=5, beta~1.0.
    
    beta = 1.0 + 2.0 * np.exp(-regret)
    
    # LCB score: higher is better for selection (since we want to minimize f, 
    # but the acquisition returns scores where "higher meaning evaluate next").
    # Wait, the prompt says: "Return 2000 finite scores, higher meaning evaluate next".
    # And "Scoring (lower is better): mean over the seeds of the simple regret".
    # The function `acquisition` returns scores. The caller picks the argmax.
    # So if we want to minimize the objective f, we should return scores that are high 
    # for points with low predicted mean and/or high uncertainty.
    # LCB for minimization: minimize (mu - beta * sigma).
    # To make "higher score = better", we should return -(mu - beta * sigma) = -mu + beta * sigma.
    # OR, simply return -mu + beta * sigma.
    
    # Let's double check EI.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z).
    # If mu is very low (good), f_best - mu is large positive, EI is large.
    # So EI returns high scores for good points.
    # LCB (for minimization) value is mu - beta * sigma.
    # We want to pick the point with the *lowest* LCB value.
    # But the interface requires *highest* score to be picked.
    # So we must return -LCB = -mu + beta * sigma.
    
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
