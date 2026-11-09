# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for Bayesian optimization.
    
    This approach computes the expected improvement of each candidate over the current best,
    providing a principled balance between exploration and exploitation.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    t_start = time.time()
    deadline = t_start + 280.0  # Safety margin before 330s kill
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Compute the standardized improvement
    delta = f_best - mu  # Positive when mu < f_best (improvement possible)
    z = delta / sigma_safe
    
    # Compute Phi(z) and phi(z) using the error function
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    
    sqrt_2 = np.sqrt(2.0)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    
    # Compute Phi(z)
    # Use scipy if available, otherwise use numpy's erf
    try:
        from scipy.special import erf
        Phi_z = 0.5 * (1.0 + erf(z / sqrt_2))
    except ImportError:
        # Fallback: use a simple approximation or manual erf
        # For numerical stability, we can use a piecewise approximation
        Phi_z = np.empty_like(z)
        # For large positive z, Phi(z) ≈ 1
        # For large negative z, Phi(z) ≈ 0
        mask_pos = z > 0
        mask_neg = ~mask_pos
        
        # Use the standard normal CDF approximation
        # For z >= 0: Phi(z) = 1 - 0.5 * erfc(z / sqrt(2))
        # We'll use a simple polynomial approximation for erf
        def _erf_approx(x):
            # Abramowitz and Stegun approximation 7.1.26
            sign = np.sign(x)
            x = np.abs(x)
            a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
            p = 0.3275911
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
            return sign * y
        
        Phi_z = 0.5 * (1.0 + _erf_approx(z / sqrt_2))
    
    # Compute phi(z)
    phi_z = np.exp(-0.5 * z * z) / sqrt_2pi
    
    # Compute EI
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # Note: when delta < 0 (mu > f_best), EI should be 0
    EI = np.where(delta > 0, delta * Phi_z + sigma_safe * phi_z, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(EI), EI, 0.0)
    
    # Check time
    if time.time() > deadline:
        pass
    
    return scores
# EVOLVE-BLOCK-END
