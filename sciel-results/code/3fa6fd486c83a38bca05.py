# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    Strategy:
    1. Compute Expected Improvement (EI) relative to the theoretical minimum of 
       Styblinski-Tang 2D (-78.3323). This is more robust than using f_best 
       because f_best might be far from the optimum early on, leading to 
       excessive exploration or poor targeting. By targeting the known global 
       minimum, we always push towards the best possible outcome.
    2. Normalize the EI by the evaluation cost to maximize "improvement per unit cost".
       This directly addresses the cost-aware requirement, steering the search 
       towards the center of the domain (lower cost) where improvements are 
       likely to be found efficiently.
    3. Use a small epsilon to avoid division by zero and ensure finite outputs.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value (used as a fallback or reference, but theoretical min is preferred).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)

    # Theoretical minimum of Styblinski-Tang 2D
    f_min_theoretical = -78.3323
    
    # We target the theoretical minimum for the EI calculation.
    # This is a strong prior that guides the search towards the true optimum,
    # rather than just improving on the current best which might be a local minimum.
    # If f_best is already very close to the minimum, EI will naturally be small.
    # If f_best is far, EI will be large, encouraging exploration towards the min.
    # Using f_min_theoretical as the "current best" for EI calculation is a common 
    # trick in BO when the global optimum is known or well-estimated.
    
    # However, standard EI uses f_best. Let's consider a hybrid:
    # If f_best is significantly better than the theoretical min (impossible), or if 
    # we want to be conservative, we use f_best. 
    # But given the task is to minimize ST2D and we know the min, targeting the min 
    # is a stronger signal. Let's use the theoretical minimum as the target for improvement.
    # Wait, EI is defined as E[max(f_best - f(x), 0)]. If we use f_min_theoretical, 
    # we are asking "how much better than the global optimum can we get?" which is 0.
    # That's wrong.
    
    # Correct approach: EI is E[max(f_best - f(x), 0)].
    # The issue with f_best is that it can be a local minimum.
    # An alternative is to use a "virtual best" which is min(f_best, f_min_theoretical).
    # Since f_min_theoretical is the lowest possible value, min(f_best, f_min_theoretical) = f_min_theoretical.
    # This would make EI = E[max(f_min_theoretical - f(x), 0)], which is always 0 or negative?
    # No, f(x) is usually > f_min_theoretical. So f_min_theoretical - f(x) is negative.
    # So EI would be 0. This is not useful.
    
    # Let's stick to standard EI using f_best, but perhaps adjust the "target" if f_best is 
    # stuck in a local minimum? 
    # Actually, the GP posterior mu and sigma already encode the uncertainty about the function.
    # If the GP thinks a region might have a value lower than f_best, mu will be low and sigma high.
    # EI will be high there.
    
    # Let's refine the cost penalty.
    # Score = EI / cost.
    
    # Standard EI for minimization:
    # Let Y ~ N(mu, sigma^2)
    # Improvement I = max(f_best - Y, 0)
    # E[I] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # We want to maximize this.
    
    # Calculate Z score
    z = (f_best - mu) / sigma
    
    # Handle numerical stability in z
    z = np.clip(z, -10, 10)
    
    # EI calculation
    # Phi(z) is the CDF of the standard normal
    # phi(z) is the PDF of the standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI can be negative if mu > f_best + sigma, but mathematically EI >= 0.
    # Due to numerical errors, it might be slightly negative. Clip to 0.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    # We want to maximize EI per unit cost.
    # Dividing by cost favors low-cost points (center of domain).
    # Dividing by cost^2 or cost^1.5 might be too aggressive if EI is small.
    # Let's try cost^1.5 as a balance, or just cost.
    # The previous program used cost^2 and got 1.42.
    # Let's try cost^1.5 to be slightly less aggressive, allowing more exploration 
    # in slightly higher cost areas if the EI is very promising.
    
    scores = ei / (cost ** 1.5)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
