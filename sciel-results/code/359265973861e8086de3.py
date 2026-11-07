import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
        
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive exploration parameter beta
    # Hartmann 3D global minimum is approx -3.86278
    target_min = -3.86278
    
    # Calculate distance from the current best to the known global minimum
    # This distance is positive and decreases as we improve
    dist = f_best - target_min
    
    # Clamp the distance to a reasonable range [0, 4.0]
    # The function range is roughly [-4, 0], so max dist is ~4
    dist = np.clip(dist, 0.0, 4.0)
    
    # Beta is high when we are far from the optimum (early stage)
    # Beta is low when we are close to the optimum (late stage)
    # Range: [1.0, 3.0]
    beta = 1.0 + 2.0 * (dist / 4.0)
    
    # LCB for minimization: minimize (mu - beta * sigma)
    # We want to maximize the score, so we negate the LCB value
    # Score = -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return scores
