import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Hybrid Thompson Sampling with Expected Improvement.
    
    Combines the stochastic exploration of Thompson Sampling (which proved
    effective for escaping local minima) with the deterministic exploitation
    signal of Expected Improvement. A small exploitation bias is added to
    stabilize selection against unlucky samples.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Component 1: Thompson Sampling (Stochastic Exploration) ---
    # Sample from posterior predictive distribution: y ~ N(mu, sigma^2)
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # Base score from TS: higher is better (we minimize y, so -y is the score)
    ts_score = -y_samples
    
    # --- Component 2: Expected Improvement (Deterministic Exploitation) ---
    # EI measures the expected amount by which a candidate improves upon f_best.
    # Standardized improvement
    z_ei = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # --- Component 3: Exploitation Bias ---
    # A deterministic bias that nudges the score based on the mean prediction
    # relative to f_best. This helps counteract unlucky high TS samples.
    bias_strength = 0.1
    exploitation_bias = bias_strength * (f_best - mu)
    
    # --- Combining the Components ---
    # The TS score dominates in magnitude (order of function values), while EI
    # is smaller (order of improvement potential). To create a meaningful hybrid,
    # we weight the EI term to be significant but not overwhelming.
    # 
    # Strategy: Use TS as the primary driver for exploration, but add a weighted
    # EI term to ensure that regions with high expected improvement are favored
    # even if a particular TS sample was unlucky. The exploitation bias further
    # stabilizes the selection.
    
    # Weight for EI: Since TS score is roughly -mu (order 1-2) and EI is order 0.1-1,
    # a weight of 1.0 makes EI comparable to the stochastic variation in TS.
    ei_weight = 1.0
    
    scores = ts_score + ei_weight * ei + exploitation_bias
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
