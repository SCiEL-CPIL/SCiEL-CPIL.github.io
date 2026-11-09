"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 320s safety margin
_DEADLINE = 320.0


def _norm_cdf_vec(z):
    """Vectorized normal CDF with clipping for numerical stability."""
    z = np.clip(z, -10, 10)
    return 0.5 * (1.0 + np.vectorize(norm.cdf)(z))


def _norm_pdf_vec(z):
    """Vectorized normal PDF with clipping for numerical stability."""
    z = np.clip(z, -10, 10)
    return np.vectorize(norm.pdf)(z)


def _compute_ei(mu, sigma, f_best):
    """Compute Expected Improvement values."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    cdf_z = _norm_cdf_vec(z)
    pdf_z = _norm_pdf_vec(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    return np.maximum(ei, 0.0)


def _knowledge_gradient_score(mu, sigma, f_best, cost):
    """
    Compute a knowledge-gradient-inspired score that balances
    expected improvement with cost awareness.
    
    The score is: EI * (1 / (1 + lambda * (cost - min_cost)))
    where lambda adapts based on the spread of costs and EIs.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    cdf_z = _norm_cdf_vec(z)
    pdf_z = _norm_pdf_vec(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: cost is in [1, 2]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = max(cost_max - cost_min, 1e-9)
    cost_norm = (cost - cost_min) / cost_range  # [0, 1]
    
    # Adaptive lambda: stronger cost penalty when EI spread is low
    ei_max = np.max(ei)
    if ei_max < 1e-9:
        # No improvement expected: prefer cheapest
        scores = 1.0 / (1.0 + 10.0 * cost_norm)
        return scores
    
    ei_norm = ei / ei_max
    
    # When EI is high, we're less sensitive to cost; when low, more sensitive
    # lambda increases as mean EI decreases
    mean_ei_norm = np.mean(ei_norm)
    lambda_cost = 5.0 * (1.0 - mean_ei_norm) + 1.0  # range [1, 6]
    
    # Score: EI normalized, divided by cost factor
    cost_factor = 1.0 + lambda_cost * cost_norm
    scores = ei_norm / cost_factor
    
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using knowledge-gradient-inspired scoring
    with adaptive cost sensitivity.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - scores (higher = evaluate next)
    """
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Compute scores
    scores = _knowledge_gradient_score(mu, sigma, f_best, cost)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > _DEADLINE:
        return scores
    
    return scores


def cost_safe(cost):
    """Ensure cost values are safe for division."""
    return np.maximum(cost, 1e-6)
# EVOLVE-BLOCK-END
