# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _ei_score(mu, sigma, f_best, cost, beta=1.0, noise_floor=1e-9):
    """Expected improvement per unit cost^beta."""
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative; higher is better
    score = ei / (cost_safe ** beta)
    return score


def _ucb_score(mu, sigma, f_best, cost, gamma=1.5, beta=1.0, noise_floor=1e-9):
    """UCB-like acquisition: -mu + gamma * sigma, cost-discounted."""
    cost_safe = np.maximum(cost, noise_floor)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _adaptive_lcb_score(mu, sigma, f_best, cost, beta_base=1.5, beta_exp=1.0, gamma=0.5, noise_floor=1e-9):
    """Adaptive LCB with cost-awareness.
    
    score = -(mu - beta * sigma) / cost^beta_exp
    where beta = beta_base * (1 + gamma * log(1 + sigma / max(f_best - mu, eps)))
    """
    sigma_safe = np.maximum(sigma, noise_floor)
    cost_safe = np.maximum(cost, noise_floor)
    
    # Relative uncertainty
    potential = np.maximum(f_best - mu, 1e-9)
    rel_uncertainty = sigma_safe / potential
    
    # Adaptive beta
    beta = beta_base * (1.0 + gamma * np.log1p(rel_uncertainty))
    
    lcb = mu - beta * sigma_safe
    score = -lcb / (cost_safe ** beta_exp)
    return score


def _normalize(s):
    """Min-max normalize to [0, 1]."""
    s_min = np.min(s)
    s_max = np.max(s)
    if s_max - s_min < 1e-12:
        return np.zeros_like(s)
    return (s - s_min) / (s_max - s_min)


def _score_candidates(mu, sigma, f_best, cost, w_ei, w_ucb, w_lcb, 
                      gamma_ucb, beta_ucb, beta_cost, gamma_lcb, beta_lcb_base):
    """Compute a weighted hybrid score from EI, UCB, and adaptive LCB."""
    s_ei = _ei_score(mu, sigma, f_best, cost, beta=beta_cost)
    s_ucb = _ucb_score(mu, sigma, f_best, cost, gamma=gamma_ucb, beta=beta_cost)
    s_lcb = _adaptive_lcb_score(mu, sigma, f_best, cost, 
                                 beta_base=beta_lcb_base, beta_exp=beta_cost, gamma=gamma_lcb)
    
    # Normalize each to [0, 1] for fair blending
    s_ei_n = _normalize(s_ei)
    s_ucb_n = _normalize(s_ucb)
    s_lcb_n = _normalize(s_lcb)
    
    # Weighted sum
    score = w_ei * s_ei_n + w_ucb * s_ucb_n + w_lcb * s_lcb_n
    return score


def _calibrate(mu, sigma, f_best, cost, time_budget=5.0):
    """Lightweight grid search to find the best combination of weights and parameters.
    
    Proxy objective: the top-k scored points should have low mean mu (exploitation)
    while maintaining reasonable cost spread (not all clustered on cheap points).
    """
    t0 = time.time()
    
    # Parameter grids
    w_ei_vals = [0.2, 0.3, 0.4, 0.5]
    w_ucb_vals = [0.2, 0.3, 0.4]
    gamma_ucb_vals = [1.0, 1.5, 2.0, 2.5]
    beta_cost_vals = [0.8, 1.0, 1.2]
    gamma_lcb_vals = [0.3, 0.5, 0.8]
    beta_lcb_base_vals = [1.0, 1.5, 2.0]
    
    top_k = min(20, len(mu) // 10)
    if top_k < 1:
        top_k = 1
    
    best_proxy = -np.inf
    best_params = (0.4, 0.3, 0.3, 1.5, 1.0, 1.0, 0.5, 1.5)  # defaults
    
    n_evals = 0
    for w_ei in w_ei_vals:
        for w_ucb in w_ucb_vals:
            w_lcb = 1.0 - w_ei - w_ucb
            if w_lcb < 0.05:
                continue
            
            for gamma_ucb in gamma_ucb_vals:
                for beta_cost in beta_cost_vals:
                    for gamma_lcb in gamma_lcb_vals:
                        for beta_lcb_base in beta_lcb_base_vals:
                            if time.time() - t0 > time_budget:
                                return best_params
                            
                            n_evals += 1
                            scores = _score_candidates(
                                mu, sigma, f_best, cost,
                                w_ei, w_ucb, w_lcb,
                                gamma_ucb, beta_cost, beta_cost,
                                gamma_lcb, beta_lcb_base
                            )
                            
                            if not np.all(np.isfinite(scores)):
                                continue
                            
                            top_idx = np.argsort(scores)[-top_k:]
                            mean_mu_top = np.mean(mu[top_idx])
                            cost_spread = np.std(cost[top_idx]) + 1e-9
                            
                            # Proxy: low mean_mu is good, some cost diversity is good
                            proxy = -mean_mu_top + 0.05 * np.log(cost_spread)
                            
                            if proxy > best_proxy:
                                best_proxy = proxy
                                best_params = (w_ei, w_ucb, w_lcb, gamma_ucb, beta_cost, 
                                               beta_cost, gamma_lcb, beta_lcb_base)
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Hartmann 3D minimization."""
    t_start = time.time()
    deadline = t_start + 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters (quick, ~5s max)
    w_ei, w_ucb, w_lcb, gamma_ucb, beta_cost, beta_cost2, gamma_lcb, beta_lcb_base = _calibrate(
        mu_safe, sigma_safe, f_best, cost_safe, time_budget=5.0
    )
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple EI with default cost discounting
        scores = _ei_score(mu_safe, sigma_safe, f_best, cost_safe, beta=1.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute hybrid score
    scores = _score_candidates(
        mu_safe, sigma_safe, f_best, cost_safe,
        w_ei, w_ucb, w_lcb,
        gamma_ucb, beta_cost, beta_cost,
        gamma_lcb, beta_lcb_base
    )
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
