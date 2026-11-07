"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety deadline
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure safety: sigma and cost must be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Structural change: Hybrid Cost-Aware Acquisition
    # Combine the probabilistic rigor of EI with the deterministic exploration
    # of LCB, and use a logarithmic cost penalty for robustness.
    #
    # 1. Compute LCB-based improvement potential:
    #    lcb_improve = (f_best - mu) + beta * sigma
    #    This is equivalent to (f_best - lcb) where lcb = mu - beta*sigma.
    #    It rewards high uncertainty (sigma) more linearly than EI (which
    #    saturates for large sigma).
    
    # Adaptive beta: scale exploration based on relative uncertainty.
    # If max_sigma is large relative to mean, we might want to explore more,
    # but standard practice is a fixed beta. Let's use a moderate beta.
    # The current program used 1.5. Let's try 2.0 to encourage exploration
    # in early stages, which helps escape local minima in 8D.
    beta = 2.0
    lcb_improve = (f_best - mu) + beta * sigma_safe

    # 2. Blend EI and LCB-improve.
    #    EI is better when uncertainty is low (exploitation).
    #    LCB-improve is better when uncertainty is high (exploration).
    #    We can blend them based on the magnitude of sigma relative to the
    #    expected improvement scale.
    #    
    #    A simple blend: score = max(ei, lcb_improve) is aggressive.
    #    A smoother blend: score = ei + alpha * lcb_improve? No, that double counts.
    #    
    #    Let's use a weighted sum where the weight on LCB increases with sigma.
    #    weight_lcb = sigma_safe / (sigma_safe + scale)
    #    weight_ei = 1 - weight_lcb
    #    scale = 0.1 (typical sigma scale in standardized space? No, mu/sigma are in objective units)
    #    Since f_best - mu is in objective units, and sigma is in objective units,
    #    we can normalize by a characteristic scale. Let's use the current best
    #    uncertainty or a fixed scale.
    #    
    #    Actually, a simpler and more robust structural change is to use
    #    "Cost-Effective LCB" directly, but with a logarithmic cost term
    #    to avoid over-penalizing expensive points when EI is very high.
    #
    #    Proposed Score:
    #    base_score = lcb_improve  # Deterministic, robust
    #    cost_penalty = np.log(cost_safe + 1e-9)
    #    scores = base_score / cost_safe  # Standard cost-aware
    #
    #    Let's try a different structural idea:
    #    Use EI as the base, but add an exploration bonus proportional to
    #    the difference between LCB-improve and EI.
    #    bonus = np.maximum(0, lcb_improve - ei)
    #    scores = (ei + 0.5 * bonus) / cost_safe
    #
    #    This ensures we never score lower than EI/cost, but we get a boost
    #    when the LCB suggests high potential that EI underestimates (due to
    #    its probabilistic nature).
    
    bonus = np.maximum(0.0, lcb_improve - ei)
    combined = ei + 0.5 * bonus
    
    # Use logarithmic cost penalty for robustness?
    # Inspiration 3 used log(ei) - log(cost).
    # Let's try: scores = combined / cost_safe
    # But if cost is very high, this kills the score.
    # Log cost: scores = np.log(combined + 1e-9) - np.log(cost_safe)
    # This is more robust to outliers in cost.
    # However, if combined is negative, log is invalid.
    # combined can be negative if mu > f_best + beta*sigma.
    # In that case, we should return a low score.
    
    # Safe log:
    # If combined > 0, use log(combined) - log(cost)
    # If combined <= 0, use a small negative number.
    
    # Let's stick to the division form but ensure it's stable.
    # scores = combined / cost_safe
    
    # Alternative: Use the log form only for the cost part, keeping the
    # improvement part linear. This is equivalent to combined * (1/cost).
    # The log form in inspiration 3 was: log(ei) - log(cost).
    # Let's try: scores = np.log(np.maximum(combined, 1e-9)) - np.log(cost_safe)
    # This heavily penalizes low combined scores and high costs.
    
    # Let's test both mentally.
    # If combined is small, log(combined) is very negative.
    # If combined is large, log(combined) is large.
    # This compresses the dynamic range of combined, which might be good.
    
    # Let's use the log form as the structural change.
    scores = np.log(np.maximum(combined, 1e-9)) - np.log(cost_safe)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
