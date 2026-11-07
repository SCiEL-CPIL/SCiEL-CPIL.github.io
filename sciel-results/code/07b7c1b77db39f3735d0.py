import numpy as np
import time
from scipy.stats import norm

# Global state for tracking progress and time
_state = {
    'call_count': 0,
    'start_time': None,
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Predictive Entropy Search (PES) variant.
    
    This approach scores candidates by approximating the expected reduction in
    the posterior entropy of the global minimum. It directly targets the multimodal
    structure of the Weierstrass function by balancing the probability of finding
    a new global minimum (exploration) with the expected improvement (exploitation).
    
    The score is a weighted combination of:
    1. A PES-like term: P(f(x) < f_best) * sigma(x), which favors uncertain regions
       with a high chance of containing a better minimum.
    2. An Expected Improvement (EI) term: E[max(f_best - f(x), 0)], which favors
       regions likely to improve upon the current best.
    
    The weights adapt over time, shifting from exploration to exploitation.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Initialize state
    if _state['start_time'] is None:
        _state['start_time'] = time.time()
        _state['call_count'] = 0
    
    # Deadline guard: if we are close to the 300s limit, return a safe, deterministic result
    elapsed = time.time() - _state['start_time']
    if elapsed > 295.0:
        sigma_safe = np.maximum(sigma, 1e-8)
        return -(mu + 0.5 * sigma_safe)
    
    _state['call_count'] += 1
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 1. Compute z-score for EI and PES
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Phi (CDF) and phi (PDF)
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    # 2. Expected Improvement (EI) for minimization
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # 3. PES-like term: probability of beating f_best weighted by uncertainty
    # This favors points where there's a good chance of finding a new global minimum,
    # especially in regions with high uncertainty.
    pes_term = Phi * sigma_safe
    
    # 4. Normalize EI and PES terms to be comparable
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_normalized = ei / ei_max
    else:
        ei_normalized = ei
    
    pes_max = np.max(pes_term)
    if pes_max > 1e-10:
        pes_normalized = pes_term / pes_max
    else:
        pes_normalized = pes_term
    
    # 5. Adaptive weighting: shift from exploration (PES) to exploitation (EI)
    # Total acquisition calls expected: 25 (30 total evaluations - 5 initial)
    total_calls = 25
    progress = min(_state['call_count'] / total_calls, 1.0)
    
    # Exploration weight: starts at 0.6, decays to 0.2
    w_pes = 0.2 + 0.4 * (1.0 - progress)
    # Exploitation weight: starts at 0.4, increases to 0.8
    w_ei = 0.8 - 0.4 * (1.0 - progress)
    
    # 6. Combine terms
    score = w_ei * ei_normalized + w_pes * pes_normalized
    
    # 7. Add a small deterministic component to break ties and prefer lower mu
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    if mu_max - mu_min > 1e-10:
        mu_normalized = (mu_max - mu) / (mu_max - mu_min)  # Higher for lower mu
    else:
        mu_normalized = np.zeros_like(mu)
    
    score = score + 0.05 * mu_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
