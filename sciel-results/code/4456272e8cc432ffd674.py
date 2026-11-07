"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm, t as student_t
import time

_start_time = time.time()
_TIME_LIMIT = 295.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Student-t Expected Improvement (EI)
    # This accounts for model uncertainty more robustly than Gaussian EI.
    # We use a degrees of freedom parameter (nu) that reflects our confidence.
    # Lower nu (more uncertainty) -> heavier tails -> more exploration.
    # Here we set nu=4, a common choice for robust Bayesian optimization.
    nu = 4.0
    
    delta = f_best - mu
    z = delta / sigma_safe
    
    # The Student-t EI formula:
    # E[max(0, f_best - X)] where X ~ t_nu(mu, sigma)
    # This can be computed as:
    # EI_t = delta * (1 - T_cdf(z)) + sigma * t_pdf(z) * (nu + z^2) / (nu - 1)
    # Note: T_cdf is the CDF of the Student-t distribution.
    # Since z is in "standard t" units (mean 0, variance 1 approx), we use standard t.
    
    t_cdf = student_t.cdf(z, nu)
    t_pdf = student_t.pdf(z, nu)
    
    # Standard t-EI formula
    # The term (nu + z^2)/(nu - 1) adjusts the pdf for the heavier tails.
    # For nu=4, this is (4+z^2)/3.
    correction_factor = (nu + z**2) / (nu - 1)
    ei_t = delta * (1 - t_cdf) + sigma_safe * t_pdf * correction_factor
    
    # 2. Cost-aware scaling
    # Instead of simply dividing by cost, we use a power law with alpha < 1.
    # This softens the penalty for high-cost points, allowing the algorithm
    # to occasionally explore expensive regions if the expected improvement
    # is very high.
    cost_safe = np.maximum(cost, 1e-9)
    alpha = 0.8
    cost_weight = cost_safe ** alpha
    
    # 3. Confidence-adaptive exploitation
    # We modulate the exploitation term (EI) based on the "confidence" of the model.
    # If the relative uncertainty (sigma / |f_best|) is high, we rely more on exploration.
    # If it is low, we rely more on exploitation.
    
    # Calculate a relative uncertainty metric
    # We use a small epsilon to avoid division by zero if f_best is 0
    f_best_abs = np.maximum(np.abs(f_best), 1e-9)
    rel_uncertainty = sigma_safe / f_best_abs
    
    # Normalize rel_uncertainty to a weight between 0 and 1
    # If rel_uncertainty is high, weight_exploit is low (more exploration).
    # If rel_uncertainty is low, weight_exploit is high (more exploitation).
    # We use a sigmoid-like transformation or simple clipping.
    # Let's use a simple inverse relationship capped.
    # weight_exploit = 1 / (1 + rel_uncertainty)
    # This gives 0.5 when uncertainty equals the best value.
    
    # To make it more aggressive in exploration when uncertainty is high:
    # weight_exploit = np.exp(-rel_uncertainty)
    
    # Let's try a linear decay that saturates.
    # If rel_uncertainty > 1, we are in high uncertainty mode.
    weight_exploit = 1.0 / (1.0 + rel_uncertainty)
    
    # The exploration component is driven by the uncertainty itself.
    # We want to encourage points with high sigma, especially if they are cheap.
    exploration_score = sigma_safe / cost_weight
    
    # Combine:
    # Score = (weight_exploit * EI_t + (1 - weight_exploit) * Exploration) / cost_weight
    # Actually, a better formulation is to blend the *signal* before dividing by cost,
    # or divide both terms.
    
    # Let's define the core acquisition value as a blend of Exploitation (EI) and Exploration (Sigma)
    # Blend ratio:
    # exploit_weight = weight_exploit
    # explore_weight = 1 - weight_exploit
    
    # However, EI already contains some exploration (via sigma in the formula).
    # To explicitly boost exploration in high-uncertainty regimes, we add a term.
    
    # Let's refine:
    # Primary signal: EI_t
    # Secondary signal: Uncertainty (sigma)
    
    # If we are confident (low rel_uncertainty), we trust EI_t.
    # If we are uncertain (high rel_uncertainty), we add a bonus for high sigma.
    
    bonus_weight = np.clip(rel_uncertainty, 0.0, 2.0) * 0.5 # Max bonus weight 1.0 if rel_unc > 2
    
    score = (ei_t + bonus_weight * sigma_safe) / cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Optional: Normalize score to avoid numerical issues in argmax if values are extreme
    # But the evaluator just picks the max, so scaling doesn't matter for selection, only for stability.
    # We'll leave it as is, but ensure no inf/nan.
    
    return score
# EVOLVE-BLOCK-END
