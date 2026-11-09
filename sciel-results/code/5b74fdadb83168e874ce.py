import numpy as np
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Entropy-Regularized Expected Improvement (ER-EI) acquisition function.
    
    Strategy:
    Standard EI focuses purely on exploitation (improving upon f_best) and can get
    stuck in local optima if the initial samples are poor or if the GP uncertainty
    is not correctly calibrated. ER-EI adds a term proportional to the predictive
    entropy (uncertainty) of the GP. This encourages the algorithm to visit regions
    with high uncertainty, effectively performing a deterministic form of exploration
    that is more stable than random Thompson Sampling and more principled than
    fixed-beta UCB.
    
    The score is: EI(x) + kappa * H(x)
    where H(x) = 0.5 * (1 + log(2 * pi * sigma^2)) is the differential entropy
    of the Gaussian posterior at x.
    
    We use a fixed kappa (0.1) as a reasonable default for standardization.
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Deadline guard: If we are close to the 330s limit, return a fast, deterministic
    # exploitation score to ensure the loop can finish or at least not crash in this call.
    if elapsed > 310:
        mu = np.asarray(mu, dtype=np.float64)
        # For minimization, we want to pick the point with the lowest mu.
        # The interface requires higher score = evaluate next.
        # So we return -mu.
        return -mu

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive for log calculation
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement I(x) = f_best - f(x)
    # EI(x) = E[max(0, f_best - f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # EI(x) = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    # To avoid numerical issues with very small sigma, we use a stable formulation
    # or just rely on scipy if available, but let's stick to numpy for speed if possible.
    # However, scipy.stats.norm is available. Let's use it for robustness.
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Entropy H(x)
    # H = 0.5 * (1 + log(2 * pi * sigma^2))
    # Note: The constant 0.5 * (1 + log(2*pi)) is the same for all points if sigma was constant,
    # but sigma varies. The varying part is log(sigma).
    # H(x) = 0.5 * (1 + log(2 * pi)) + log(sigma)
    # We can ignore the constant part for ranking purposes, so we just use log(sigma).
    # However, to be faithful to the "Entropy" concept, let's compute the full term or just log(sigma).
    # Using log(sigma) is sufficient for the relative ranking and avoids large constant offsets.
    entropy_term = np.log(sigma)
    
    # Combine EI and Entropy
    # kappa controls the balance. 
    # A typical value for standardized functions might be small.
    # Let's try kappa = 0.1.
    kappa = 0.1
    
    scores = ei + kappa * entropy_term
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
