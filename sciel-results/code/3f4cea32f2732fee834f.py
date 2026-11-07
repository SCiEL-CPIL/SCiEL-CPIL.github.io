import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a time-aware
    exploration component. The exploration weight adapts based on the
    uncertainty relative to the current best and the predicted mean.
    
    This approach uses a simpler, more robust exploration strategy that
    focuses on regions where the GP is uncertain but the predicted mean
    suggests potential for improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Exploration term: focus on high-uncertainty regions that are also
    # promising (predicted mean better than f_best)
    # Use a simpler weighting: sigma * (1 - sigmoid(mu - f_best))
    # The sigmoid term gives weight 1 when mu << f_best and 0 when mu >> f_best
    
    # Normalize the difference between mu and f_best for the sigmoid
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-6 else 1.0
    sig_arg = 5.0 * (mu - f_best) / mu_range
    sig_arg = np.clip(sig_arg, -50, 50)
    promise_weight = 1.0 / (1.0 + np.exp(sig_arg))
    
    # Exploration: scale sigma by the promise weight
    # Use a beta that's moderate to balance exploration and exploitation
    beta = 0.2
    exploration = beta * sigma_safe * promise_weight
    
    # Combine EI and exploration
    scores = ei + exploration
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
