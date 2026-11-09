# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 330.0  # wall-clock limit


def _safe_norm_cdf(x):
    """Numerically stable norm.cdf for large negative x."""
    x = np.asarray(x, dtype=np.float64)
    # For very negative x, cdf ~ 0; for very positive, ~1
    # scipy's norm.cdf is already stable, but let's ensure finiteness
    out = norm.cdf(x)
    return np.where(np.isfinite(out), out, 0.0)


def _safe_norm_pdf(x):
    """Numerically stable norm.pdf."""
    x = np.asarray(x, dtype=np.float64)
    out = norm.pdf(x)
    return np.where(np.isfinite(out), out, 0.0)


def _ei_cost_aware(mu, sigma, f_best, cost, gamma=1.0, beta=1.0):
    """Cost-aware Expected Improvement: EI / cost^beta.
    
    For minimization, EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    cdf_z = _safe_norm_cdf(z)
    pdf_z = _safe_norm_pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    # Ensure EI is non-negative (it should be, but numerical issues can arise)
    ei = np.maximum(ei, 0.0)
    cost_safe = np.maximum(cost, 1e-8)
    score = ei / (cost_safe ** beta)
    return score


def _ucb_cost_aware(mu, sigma, f_best, cost, gamma=2.0, beta=1.0):
    """Cost-aware UCB: (-mu + gamma * sigma) / cost^beta."""
    cost_safe = np.maximum(cost, 1e-8)
    score = (-mu + gamma * sigma) / (cost_safe ** beta)
    return score


def _pi_cost_aware(mu, sigma, f_best, cost, beta=1.0):
    """Cost-aware Probability of Improvement: PI / cost^beta."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    pi = _safe_norm_cdf(z)
    cost_safe = np.maximum(cost, 1e-8)
    score = pi / (cost_safe ** beta)
    return score


def _hybrid_score(mu, sigma, f_best, cost, w_ei=0.7, w_ucb=0.3, gamma=2.0, beta=1.0):
    """Hybrid of cost-aware EI and cost-aware UCB, normalized to comparable scales."""
    ei = _ei_cost_aware(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
    ucb = _ucb_cost_aware(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
    
    # Normalize each to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range > 1e-12:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_range > 1e-12:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    return score


def _calibrate(mu, sigma, f_best, cost):
    """Lightweight calibration: pick the best (method, params) by evaluating
    a proxy quality metric on the current posterior.
    
    Proxy: we want the top-k scored points to have low mu (exploit) and
    reasonable sigma (explore). We use the mean mu of the top-k points as
    a quality metric (lower is better). We also consider the spread of
    sigma among top-k points to ensure diversity.
    """
    t0 = time.time()
    remaining = _DEADLINE - t0
    if remaining < 2.0:
        # Not enough time, use defaults
        return 'hybrid', 0.7, 0.3, 2.0, 1.0
    
    top_k = min(30, max(5, len(mu) // 50))
    
    # Candidate configurations: (method, w_ei, w_ucb, gamma, beta)
    candidates = [
        ('ei', 1.0, 0.0, 1.0, 0.5),
        ('ei', 1.0, 0.0, 1.0, 1.0),
        ('ei', 1.0, 0.0, 1.0, 1.5),
        ('ucb', 0.0, 1.0, 1.5, 0.5),
        ('ucb', 0.0, 1.0, 2.0, 0.5),
        ('ucb', 0.0, 1.0, 2.5, 0.5),
        ('ucb', 0.0, 1.0, 2.0, 1.0),
        ('ucb', 0.0, 1.0, 2.5, 1.0),
        ('ucb', 0.0, 1.0, 3.0, 1.0),
        ('hybrid', 0.5, 0.5, 2.0, 0.5),
        ('hybrid', 0.6, 0.4, 2.0, 0.5),
        ('hybrid', 0.7, 0.3, 2.0, 0.5),
        ('hybrid', 0.7, 0.3, 2.5, 0.5),
        ('hybrid', 0.8, 0.2, 2.0, 1.0),
        ('hybrid', 0.8, 0.2, 2.5, 1.0),
        ('pi', 1.0, 0.0, 1.0, 1.0),
    ]
    
    best_quality = -np.inf
    best_params = ('hybrid', 0.7, 0.3, 2.0, 1.0)
    
    for method, w_ei, w_ucb, gamma, beta in candidates:
        if time.time() - t0 > remaining - 1.0:
            break
        
        try:
            if method == 'ei':
                scores = _ei_cost_aware(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
            elif method == 'ucb':
                scores = _ucb_cost_aware(mu, sigma, f_best, cost, gamma=gamma, beta=beta)
            elif method == 'pi':
                scores = _pi_cost_aware(mu, sigma, f_best, cost, beta=beta)
            else:  # hybrid
                scores = _hybrid_score(mu, sigma, f_best, cost, w_ei=w_ei, w_ucb=w_ucb,
                                       gamma=gamma, beta=beta)
        except Exception:
            continue
        
        if not np.all(np.isfinite(scores)):
            continue
        
        # Get top-k indices
        top_idx = np.argsort(scores)[-top_k:]
        mean_mu_top = np.mean(mu[top_idx])
        mean_sigma_top = np.mean(sigma[top_idx])
        
        # Quality: prefer low mean_mu_top, and some diversity (mean_sigma_top not too small)
        # Use a weighted combination: -mean_mu_top + 0.1 * mean_sigma_top
        quality = -mean_mu_top + 0.1 * mean_sigma_top
        
        if quality > best_quality:
            best_quality = quality
            best_params = (method, w_ei, w_ucb, gamma, beta)
    
    return best_params


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    global _T0
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: handle any non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Calibrate parameters
    method, w_ei, w_ucb, gamma, beta = _calibrate(mu_safe, sigma_safe, f_best, cost_safe)
    
    # Compute scores
    if method == 'ei':
        scores = _ei_cost_aware(mu_safe, sigma_safe, f_best, cost_safe, gamma=gamma, beta=beta)
    elif method == 'ucb':
        scores = _ucb_cost_aware(mu_safe, sigma_safe, f_best, cost_safe, gamma=gamma, beta=beta)
    elif method == 'pi':
        scores = _pi_cost_aware(mu_safe, sigma_safe, f_best, cost_safe, beta=beta)
    else:  # hybrid
        scores = _hybrid_score(mu_safe, sigma_safe, f_best, cost_safe, w_ei=w_ei, w_ucb=w_ucb,
                               gamma=gamma, beta=beta)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE - 5.0:
        # Return immediately with whatever we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
