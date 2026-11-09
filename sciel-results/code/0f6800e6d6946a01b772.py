# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


# --- Tunable parameters (will be calibrated) ---
# We expose a small set of parameters and optimize them.

def _ei_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0, noise_floor=1e-9):
    """Generalized expected improvement per unit cost.
    
    gamma: exploration/exploitation trade-off (higher = more exploration)
    beta: cost discounting exponent (higher = more penalizes expensive points)
    """
    sigma_safe = np.maximum(sigma, noise_floor)
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    # Expected improvement (EI) with gamma scaling
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Score is EI / cost^beta
    score = ei / (cost ** beta)
    return score


def _ucb_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    score = (-mu + gamma * sigma) / (cost ** beta)
    return score

def _lcb_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """LCB-like acquisition for minimization: mu - gamma * sigma, cost-discounted.
    We maximize this score to find points with low potential minimum.
    """
    score = (mu - gamma * sigma) / (cost ** beta)
    return score

def _pi_score(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """Probability of improvement per unit cost."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    score = pi / (cost ** beta)
    return score


def _calibrate_params(mu, sigma, f_best, cost, n_candidates=10, time_budget=5.0):
    """Lightweight calibration of (gamma, beta) by maximizing a surrogate
    objective: we assume that lower scores on points with low mu and high sigma
    (relative to f_best) are desirable for a minimizer. We use a proxy:
    the acquisition should rank points with low mu highly, and points with
    high sigma moderately (exploration). We evaluate candidate parameters
    on the current data and pick the one that best separates low-mu points.
    
    In practice, we'll just try a few reasonable values and pick based on
    a simple heuristic: the chosen parameters should produce scores that
    are higher for the best-known region (low mu) and lower for expensive
    points.
    """
    t0 = time.time()
    
    # A small grid of (gamma, beta)
    # Expanded grid for better resolution
    gamma_vals = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    beta_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    best_score = -np.inf
    best_params = (1.0, 1.0)  # default to standard EI
    best_method = 'ei'
    
    # For each candidate, compute a proxy quality:
    # We want the top-k scores to correspond to points with low mu.
    # Proxy: correlation between score and (-mu), or the mean mu of top-k
    # points. Lower mean mu of top-k is better.
    
    top_k = min(20, len(mu) // 10)
    if top_k < 1:
        top_k = 1
    
    for method in ['ei', 'ucb', 'lcb']:
        for gamma in gamma_vals:
            for beta in beta_vals:
                if time.time() - t0 > time_budget:
                    break
                if method == 'ei':
                    scores = _ei_score(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
                elif method == 'ucb':
                    scores = _ucb_score(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
                elif method == 'lcb':
                    scores = _lcb_score(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
                else:
                    continue
                
                if not np.all(np.isfinite(scores)):
                    continue
                
                # Get indices of top-k scores
                top_idx = np.argsort(scores)[-top_k:]
                mean_mu_top = np.mean(mu[top_idx])
                
                # We want low mean_mu_top. Use negative as "score" to maximize.
                proxy = -mean_mu_top
                
                if proxy > best_score:
                    best_score = proxy
                    best_params = (gamma, beta)
                    best_method = method
    
    return best_method, best_params[0], best_params[1]


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle any non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters (quick, ~5s max)
    method, gamma, beta = _calibrate_params(mu_safe, sigma_safe, f_best, cost_safe, 
                                             n_candidates=10, time_budget=5.0)
    
    if method == 'ei':
        scores = _ei_score(mu_safe, sigma_safe, f_best, cost_safe, gamma=gamma, beta=beta)
    elif method == 'ucb':
        scores = _ucb_score(mu_safe, sigma_safe, f_best, cost_safe, gamma=gamma, beta=beta)
    elif method == 'lcb':
        scores = _lcb_score(mu_safe, sigma_safe, f_best, cost_safe, gamma=gamma, beta=beta)
    else:
        scores = _ei_score(mu_safe, sigma_safe, f_best, cost_safe, gamma=1.0, beta=1.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
