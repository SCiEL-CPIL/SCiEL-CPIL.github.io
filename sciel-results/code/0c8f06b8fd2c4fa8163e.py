import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: simple EI / cost
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-12)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    
    # Use vectorized normal CDF and PDF
    # For large |z|, use asymptotic approximations for numerical stability
    # Phi(z) for z > 8 is ~1, for z < -8 is ~0
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    
    # Clip z to avoid overflow in exp
    z_clipped = np.clip(z, -50, 50)
    
    # Compute phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    phi_z = np.exp(-0.5 * z_clipped ** 2) / np.sqrt(2 * np.pi)
    
    # Compute Phi(z) using the error function
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    from scipy.special import erf
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    
    # Expected Improvement
    ei = (f_best - mu) * cdf_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        cost_safe = np.maximum(cost, 1e-12)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Cost-aware scoring
    # Cost is 1 + mean(x_scaled), where x_scaled is in [0,1]
    # So cost ranges from 1 to 2
    # We want to penalize higher cost evaluations
    
    cost_safe = np.maximum(cost, 1e-12)
    
    # Strategy: Use a power-law cost penalty
    # The exponent should balance exploration and exploitation
    # 
    # Key insight: In the early stages, we should explore more (lower cost penalty)
    # In the later stages, we should exploit (higher cost penalty)
    #
    # We can estimate the stage by looking at the spread of mu relative to f_best
    # If mu values are close to f_best, we're in exploitation mode
    
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    
    # If mu is tightly clustered, we're exploiting
    # If mu is spread out, we're exploring
    if mu_range > 1e-9:
        # Normalized position of f_best in the mu range
        # f_best should be >= mu_min (since f_best is the minimum observed)
        f_best_rel = (f_best - mu_min) / mu_range
        f_best_rel = np.clip(f_best_rel, 0, 1)
    else:
        f_best_rel = 0.5
    
    # Cost exponent: lower when exploring, higher when exploiting
    # Range: [0.5, 2.0]
    cost_exp = 0.5 + 1.5 * f_best_rel
    
    # Also consider the mean sigma to adjust
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    if sigma_max > 1e-9:
        sigma_rel = sigma / sigma_max
    else:
        sigma_rel = np.zeros_like(sigma)
    
    # Per-point cost exponent adjustment:
    # High uncertainty -> lower cost penalty (more exploration)
    # Low uncertainty -> higher cost penalty (more exploitation)
    cost_exp_adjusted = cost_exp * (0.6 + 0.4 * (1.0 - sigma_rel))
    
    # Base score: EI normalized by cost
    scores = ei / (cost_safe ** cost_exp_adjusted)
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Add an exploration bonus for high-uncertainty points
    # This helps escape local optima
    exploration_bonus = 0.15 * sigma_rel * ei
    
    scores += exploration_bonus
    
    # Alternative: Use a hybrid with UCB
    # UCB = -mu + kappa * sigma (for minimization, higher is better)
    # Adaptive kappa: higher when less converged
    
    # Estimate convergence
    improvement_spread = np.std(f_best - mu)
    if improvement_spread > 1e-9:
        convergence_ratio = sigma_mean / improvement_spread
    else:
        convergence_ratio = 1.0
    
    # kappa: [1.0, 4.0] - higher when less converged
    kappa = 1.0 + 3.0 * np.clip(1.0 - convergence_ratio, 0, 1)
    
    # UCB score (for minimization, higher is better):
    ucb = -mu + kappa * sigma
    
    # Normalize UCB to be comparable to EI scale
    # EI is in objective units, UCB is also in objective units
    # But UCB can be negative, so we need to shift it
    ucb_shifted = np.maximum(ucb, 0)
    
    # Scale UCB by a factor to make it comparable to EI
    ei_max = np.max(ei)
    ucb_max = np.max(ucb_shifted)
    
    if ucb_max > 1e-9 and ei_max > 1e-9:
        ucb_scale = ei_max / ucb_max
    else:
        ucb_scale = 1.0
    
    ucb_score = ucb_shifted * ucb_scale / (cost_safe ** cost_exp_adjusted)
    
    # Final score: max of EI-based and UCB-based
    scores = np.maximum(scores, ucb_score)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
