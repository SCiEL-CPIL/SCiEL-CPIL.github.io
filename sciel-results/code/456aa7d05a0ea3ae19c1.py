# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).
    
    STRUCTURAL CHANGE: Uses an additive cost penalty model instead of pure division.
    Score(x) = EI(x) - lambda_cost * cost(x) + lambda_exp * sigma(x)
    
    This allows high-EI points in expensive regions to be selected if the expected 
    improvement outweighs the cost, while still preferring cheap points when EI is 
    comparable. The lambda parameters are auto-calibrated from the surrogate 
    statistics to ensure the terms are commensurate.
    """
    t_start = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    n = mu.size

    # Numerical guards
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # 1. Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -40.0, 40.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # 2. Auto-calibrate weights based on current surrogate statistics
    # We want the cost penalty and exploration bonus to be small perturbations 
    # relative to the typical EI, so that EI dominates but cost/uncertainty 
    # break ties and guide the search.
    
    # Estimate typical scales using medians for robustness against outliers
    finite_ei = ei[np.isfinite(ei) & (ei > 0)]
    if finite_ei.size > 0:
        scale_ei = float(np.median(finite_ei))
    else:
        # If no improvement is expected, fall back to sigma scale
        finite_sig = sigma_safe[np.isfinite(sigma_safe)]
        scale_ei = float(np.median(finite_sig)) if finite_sig.size else 1.0
    if not np.isfinite(scale_ei) or scale_ei < 1e-8:
        scale_ei = 1.0

    finite_cost = cost_safe[np.isfinite(cost_safe)]
    scale_cost = float(np.median(finite_cost)) if finite_cost.size else 1.0
    if not np.isfinite(scale_cost) or scale_cost < 1e-8:
        scale_cost = 1.0

    finite_sig = sigma_safe[np.isfinite(sigma_safe)]
    scale_sigma = float(np.median(finite_sig)) if finite_sig.size else 1.0
    if not np.isfinite(scale_sigma) or scale_sigma < 1e-8:
        scale_sigma = 1.0

    # Set lambda_cost such that lambda_cost * scale_cost = 0.1 * scale_ei
    # This makes cost a 10% tie-breaker factor
    lambda_cost = 0.1 * scale_ei / scale_cost
    
    # Set lambda_exp such that lambda_exp * scale_sigma = 0.1 * scale_ei
    # This makes exploration a 10% tie-breaker factor
    lambda_exp = 0.1 * scale_ei / scale_sigma

    # Ensure lambdas are finite and non-negative
    lambda_cost = float(np.clip(lambda_cost, 0.0, 1e6))
    lambda_exp = float(np.clip(lambda_exp, 0.0, 1e6))

    # 3. Construct the final score
    # Higher score is better
    # EI is the primary driver
    # Cost is penalized
    # Uncertainty is rewarded
    scores = ei - lambda_cost * cost_safe + lambda_exp * sigma_safe

    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if (time.time() - t_start) > 300.0:
        return scores

    return scores
# EVOLVE-BLOCK-END
