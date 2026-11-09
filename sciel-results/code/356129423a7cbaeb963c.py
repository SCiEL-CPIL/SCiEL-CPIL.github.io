# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _pi_score(mu, sigma, f_best, cost, beta=1.0, noise_floor=1e-9):
    """Probability of improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # PI is in [0, 1]; higher is better
    score = pi / (cost_safe ** beta)
    return score


def _ei_score(mu, sigma, f_best, cost, beta=1.0, noise_floor=1e-9):
    """Expected improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative; higher is better
    score = ei / (cost_safe ** beta)
    return score


def _ucb_score(mu, sigma, f_best, cost, gamma=2.0, beta=1.0, noise_floor=1e-9):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, noise_floor)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _hybrid_score(mu, sigma, f_best, cost, w_pi, w_ei, w_ucb,
                  beta_pi, beta_ei, beta_ucb, gamma_ucb):
    """Weighted hybrid of PI, EI, and UCB, all cost-discounted."""
    s_pi = _pi_score(mu, sigma, f_best, cost, beta=beta_pi)
    s_ei = _ei_score(mu, sigma, f_best, cost, beta=beta_ei)
    s_ucb = _ucb_score(mu, sigma, f_best, cost, gamma=gamma_ucb, beta=beta_ucb)
    
    # Normalize each to [0, 1] for fair blending
    def _norm(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    s_pi_n = _norm(s_pi)
    s_ei_n = _norm(s_ei)
    s_ucb_n = _norm(s_ucb)
    
    score = w_pi * s_pi_n + w_ei * s_ei_n + w_ucb * s_ucb_n
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=8.0):
    """Grid search over acquisition parameters using a proxy objective.
    
    The proxy evaluates how well the top-k scored candidates balance
    exploitation (low mu) and exploration (high sigma) while respecting
    cost efficiency.
    """
    t0 = time.time()
    
    # Parameter grids - structured to explore the space efficiently
    w_pi_vals = [0.2, 0.3, 0.4, 0.5]
    w_ei_vals = [0.2, 0.3, 0.4]
    # w_ucb = 1 - w_pi - w_ei, must be >= 0.1
    beta_pi_vals = [0.8, 1.0, 1.2]
    beta_ei_vals = [0.8, 1.0, 1.2]
    beta_ucb_vals = [0.8, 1.0, 1.2]
    gamma_ucb_vals = [1.5, 2.0, 2.5, 3.0]
    
    top_k = min(30, max(5, len(mu) // 100))
    
    best_proxy = -np.inf
    best_params = (0.3, 0.4, 0.3, 1.0, 1.0, 1.0, 2.0)  # defaults
    
    for w_pi in w_pi_vals:
        for w_ei in w_ei_vals:
            w_ucb = 1.0 - w_pi - w_ei
            if w_ucb < 0.1:
                continue
            
            for beta_pi in beta_pi_vals:
                for beta_ei in beta_ei_vals:
                    for beta_ucb in beta_ucb_vals:
                        for gamma_ucb in gamma_ucb_vals:
                            if time.time() - t0 > time_budget:
                                return best_params
                            
                            scores = _hybrid_score(
                                mu, sigma, f_best, cost,
                                w_pi, w_ei, w_ucb,
                                beta_pi, beta_ei, beta_ucb, gamma_ucb
                            )
                            
                            if not np.all(np.isfinite(scores)):
                                continue
                            
                            top_idx = np.argsort(scores)[-top_k:]
                            mean_mu = np.mean(mu[top_idx])
                            mean_sigma = np.mean(sigma[top_idx])
                            mean_cost = np.mean(cost[top_idx])
                            
                            # Proxy: want low mu, reasonable sigma (not too high), low cost
                            # Scale factors to balance terms
                            proxy = -mean_mu + 0.1 * np.log1p(mean_sigma) - 0.3 * mean_cost
                            
                            if proxy > best_proxy:
                                best_proxy = proxy
                                best_params = (w_pi, w_ei, w_ucb,
                                              beta_pi, beta_ei, beta_ucb, gamma_ucb)
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Hartmann 3D minimization."""
    t_start = time.time()
    deadline = t_start + 310.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters
    w_pi, w_ei, w_ucb, beta_pi, beta_ei, beta_ucb, gamma_ucb = _calibrate(
        mu_safe, sigma_safe, f_best, cost_safe, time_budget=8.0
    )
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple EI with default cost discounting
        scores = _ei_score(mu_safe, sigma_safe, f_best, cost_safe, beta=1.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute hybrid score
    scores = _hybrid_score(
        mu_safe, sigma_safe, f_best, cost_safe,
        w_pi, w_ei, w_ucb,
        beta_pi, beta_ei, beta_ucb, gamma_ucb
    )
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
