import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0  # 5s margin before 330s kill

# Track state across calls for density-aware acquisition
_call_count = 0
# We'll use a simple approach: maintain a running estimate of where we've been
# Since we don't receive coordinates, we'll use the mu/sigma patterns to infer
# exploration vs exploitation needs

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Check deadline
    if time.time() > _DEADLINE:
        return -mu
    
    # Ensure inputs are valid
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-10)  # avoid log(0)
    
    n = len(mu)
    
    # Adaptive kappa based on number of evaluations
    # Early: more exploration, later: more exploitation
    # 30 total evaluations, 5 initial, so 25 acquisition-driven
    if _call_count <= 5:
        kappa = 3.0  # high exploration early
    elif _call_count <= 10:
        kappa = 2.5
    elif _call_count <= 15:
        kappa = 2.0
    elif _call_count <= 20:
        kappa = 1.5
    elif _call_count <= 25:
        kappa = 1.0
    else:
        kappa = 0.5
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal
    diff = f_best - mu
    z = diff / sigma
    
    # Clip z to avoid numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    # Compute EI
    ei = diff * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # UCB component
    ucb = -mu + kappa * sigma
    
    # Combine EI and UCB with adaptive weighting
    # Early: weight more on UCB (exploration), later: weight more on EI (exploitation)
    if _call_count <= 8:
        w_ei = 0.3
        w_ucb = 0.7
    elif _call_count <= 15:
        w_ei = 0.5
        w_ucb = 0.5
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    scores = w_ei * ei + w_ucb * ucb
    
    # Density-aware penalty: penalize candidates where mu is very low 
    # (indicating we've been exploring there) by using a diversity bonus
    # based on the spread of mu values
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-10
    
    # Normalize mu to [0, 1] for diversity calculation
    mu_norm = (mu - mu_min) / mu_range
    
    # Candidates with mu far from the current best (f_best) get a diversity bonus
    # This encourages exploring regions we haven't refined yet
    diversity_bonus = 0.1 * np.abs(mu_norm - 0.5)  # mild bonus for extreme regions
    
    scores = scores + diversity_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
