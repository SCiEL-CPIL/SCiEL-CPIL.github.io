"""
Strategy: Pure Lower Confidence Bound (LCB) acquisition function.
Unlike the previous EI/UCB hybrid, this approach uses a single, direct
formula for minimization: score = mu - kappa * sigma.
This contrasts with EI (which estimates expected improvement) and UCB (which
is typically used for maximization or requires sign flipping).
LCB is structurally simpler and often more robust in multi-modal landscapes
like Styblinski-Tang because it aggressively explores high-uncertainty regions
without the non-linear saturation effects of the CDF in EI.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (n,), posterior mean
        sigma: array of shape (n,), posterior std
        f_best: float, best observed value so far
    
    Returns:
        scores: array of shape (n,), higher is better
    """
    # Safety check for time
    if time.time() > _DEADLINE - 5.0:
        # Fallback: return pure uncertainty to break ties or just zero
        # But LCB is very fast, so this is unlikely.
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB for minimization: we want low values.
    # The "Lower Confidence Bound" of the true function value is mu - kappa * sigma.
    # We want to MINIMIZE this bound (find the lowest possible value).
    # However, the acquisition function returns scores where HIGHER is better.
    # So, if we define the bound B = mu - kappa * sigma, we want to maximize -B
    # OR we can just define the score as -B.
    # Let's verify:
    # If mu is low and sigma is low, B is low. -B is high. Good.
    # If mu is high and sigma is high, B might be low (if kappa*sigma > mu-f_best? No, mu is absolute).
    # Actually, standard LCB for minimization is:
    # argmin_x (mu(x) - kappa * sigma(x))
    # To convert to a maximization problem (higher score is better):
    # score(x) = - (mu(x) - kappa * sigma(x)) = -mu(x) + kappa * sigma(x)
    
    # Let's use a fixed kappa. 
    # In many BO implementations, kappa is tuned. 
    # A common choice is around 1.0 to 3.0.
    # Given the multi-modality, we might want to explore more, so kappa=2.0 is a reasonable start.
    # Note: The previous program used a blend. This is pure LCB.
    
    kappa = 2.0
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
# EVOLVE-BLOCK-END
