import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Expected Improvement + Adaptive Thompson Sampling.
    
    This function computes a score for each candidate point to select the next
    evaluation point for Bayesian Optimization of the 1D Weierstrass function.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior at candidate points, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior at candidate points, shape (2000,).
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : np.ndarray
        Acquisition scores, shape (2000,). Higher scores indicate better candidates.
    """
    start_time = time.time()
    deadline = start_time + 2.0  # Safety margin well below 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-9)
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    n = mu_safe.shape[0]
    
    # --- Component 1: Expected Improvement (EI) ---
    # For minimization, EI is defined as:
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    # where Phi is CDF and phi is PDF of standard normal.
    
    # Avoid division by zero
    z = (f_best - mu_safe) / sigma_safe
    
    # Use scipy for accurate CDF/PDF if available, otherwise approximate or use numpy
    # Since scipy is allowed, we use it for precision.
    from scipy.stats import norm
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI calculation
    ei = (f_best - mu_safe) * cdf + sigma_safe * pdf
    
    # EI is non-negative. We want to maximize EI.
    # However, EI can be 0 if mu > f_best. We want to prefer points with lower mu.
    # Standard EI handles this, but let's ensure we have a baseline exploitation signal.
    
    # --- Component 2: Thompson Sampling (TS) ---
    # TS samples from the posterior: x_sample = mu + sigma * epsilon
    # For minimization, we want to pick points where the sampled value is likely low.
    # A common deterministic proxy for TS in a single-call acquisition is to use
    # the lower bound of the posterior, e.g., mu - k * sigma.
    # However, a more robust "TS-like" score that maintains stochasticity across
    # multiple calls (seeds) is difficult without state. 
    # Instead, we use a fixed "optimistic" bound that acts like TS:
    # Score_TS = - (mu - k_ts * sigma) = -mu + k_ts * sigma
    # This is essentially LCB.
    
    # To make it "adaptive" and distinct from pure LCB, we blend EI and LCB.
    # EI is great for exploitation. LCB is great for exploration.
    
    # Adaptive Beta for LCB part:
    # Early on (f_best high relative to sigma), we want more exploration (higher beta).
    # Late on (f_best low), we want exploitation (lower beta).
    # Let's estimate the scale of the problem.
    # The Weierstrass function on [-0.5, 0.5] has a minimum of 0.
    # Typical initial values might be around 0.5-1.0.
    
    med_sigma = np.median(sigma_safe)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
        
    # Ratio of current best to uncertainty
    ratio = f_best / med_sigma
    
    # If ratio is large, we are far from optimum -> Explore more.
    # If ratio is small, we are close -> Exploit more.
    
    # We blend EI and LCB.
    # EI is already a strong signal. Let's add a small LCB component to break ties
    # or encourage slight exploration in flat regions.
    
    # Let's try a weighted sum:
    # Score = w_ei * EI + w_lcb * (-mu + beta * sigma)
    
    # Normalize EI to be comparable to LCB terms?
    # EI has units of objective function. LCB has units of objective function.
    # So they are directly comparable.
    
    # Adaptive weights:
    # As we get closer to 0, we want to trust EI more.
    # Let's use a fixed small weight for LCB to ensure exploration, 
    # but scale the beta in LCB.
    
    # Beta for LCB:
    # Start high (e.g., 2.0) and decrease as f_best decreases.
    # Simple heuristic: beta = max(0.5, min(3.0, f_best / (med_sigma * 2)))
    # Actually, let's stick to the proven adaptive LCB from the parent but blend it.
    
    beta = max(0.5, min(3.0, f_best / med_sigma))
    
    lcb_score = -mu_safe + beta * sigma_safe
    
    # Combine EI and LCB.
    # EI is generally more reliable for finding the global min.
    # LCB helps escape local minima.
    # We will use EI as the primary driver and add a small LCB component.
    
    # To make the blend robust, we can normalize both to [0, 1] or similar,
    # but since they are in the same units, a simple sum works if magnitudes are similar.
    # EI values are typically small when close to optimum.
    # LCB values are dominated by -mu.
    
    # Let's use a fixed weight, e.g., 0.1 for LCB, to slightly bias towards exploration.
    # Or, better yet, use the LCB only where EI is low (uncertain)?
    
    # A simpler and often more effective strategy for 1D rugged functions:
    # Use EI, but if the posterior variance is very high, boost the score.
    
    # Let's try: Score = EI + lambda * sigma
    # This is similar to UCB but with EI as the base.
    # lambda should be small.
    
    # Let's refine the adaptive beta approach from the parent which scored 3.65.
    # The parent used: score = -mu + beta * sigma.
    # My hypothesis: EI is better than pure LCB for exploitation.
    # Let's compute Score = EI + 0.5 * sigma.
    # This adds a small exploration bonus to the rigorous EI.
    
    exploration_bonus = 0.5 * sigma_safe
    
    score = ei + exploration_bonus
    
    # Check time
    if time.time() > deadline:
        # Return best available quickly
        return np.where(np.isfinite(score), score, -np.inf)
        
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
