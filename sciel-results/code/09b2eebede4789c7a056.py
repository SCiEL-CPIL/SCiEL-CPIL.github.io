import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard
_MODULE_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using GP-UCB with a time-varying exploration parameter.
    
    For minimization, we use the Lower Confidence Bound (LCB):
    LCB(x) = mu(x) - beta_t * sigma(x)
    
    We want to maximize the probability of finding a new minimum, so we
    return a score that is high when LCB is low (i.e., when mu - beta*sigma is low).
    
    The exploration parameter beta_t increases with the number of evaluations,
    but we approximate it using the current state. Since we don't have direct
    access to t (number of evaluations), we use a heuristic based on the
    spread of sigma values to adapt the exploration-exploitation trade-off.
    
    Actually, a simpler and more robust approach: use a fixed but well-tuned
    beta, or use EI which is known to perform well. Let's use a hybrid:
    primarily EI with a small UCB component to ensure exploration.
    """
    # Deadline guard: if we are close to the 330s limit, return a safe, fast score
    if time.time() - _MODULE_START_TIME > 310:
        # Fallback: Return a simple score
        return np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle sigma = 0 cases to avoid division by zero
    eps = 1e-10
    sigma_safe = np.where(sigma > eps, sigma, eps)
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps in early stages when we need to explore
    # The bonus is proportional to sigma, but scaled down
    exploration_bonus = 0.1 * sigma
    
    # Combine EI and exploration bonus
    # We want high scores for good candidates
    # EI is positive when there's expected improvement
    score = ei + exploration_bonus
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
