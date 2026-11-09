"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Enforce deadline with safety margin
    start_time = time.time()
    deadline = start_time + 295.0  # 300s limit, 5s safety margin

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle potential zeros in sigma and cost
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Strategy: Cost-Aware Upper Confidence Bound (UCB)
    # UCB is often more robust than EI in early stages or when the GP is poorly calibrated.
    # It balances exploitation (mu) and exploration (sigma) with a parameter kappa.
    # We use a dynamic kappa that decreases as we get more confident (higher f_best improvement)
    # or simply a fixed kappa that works well for 2D problems.
    
    # For minimization, we want low mu and high sigma.
    # Standard UCB for minimization: score = mu - kappa * sigma
    # But we want to maximize the acquisition score.
    # So we maximize: -mu + kappa * sigma
    # Or equivalently, minimize: mu - kappa * sigma
    
    # Let's define the raw UCB score for minimization (lower is better)
    # ucb_raw = mu - kappa * sigma
    
    # We need to normalize or scale this to be comparable to the objective values.
    # The objective values for Three-Hump Camel are typically in the range [0, 10].
    # mu and sigma are in objective units.
    
    # A common choice for kappa in 2D is around 2.0 to 4.0.
    # Let's try a slightly aggressive exploration initially.
    kappa = 3.0
    
    ucb_score = -mu + kappa * sigma_safe
    
    # Cost-aware adjustment:
    # We divide by a function of cost.
    # The previous program used cost^0.7.
    # Let's try cost^0.5 (sqrt) which is less punitive to high-cost regions than cost^1.0
    # but more punitive than cost^0.1.
    # Alternatively, we can use a linear penalty: ucb_score - lambda * cost
    # But division is standard for "efficiency".
    
    # Let's use cost^0.5
    cost_penalty = np.sqrt(cost_safe)
    
    score = ucb_score / cost_penalty
    
    # Safety: ensure all scores are finite
    if not np.all(np.isfinite(score)):
        # Fallback to simple cost-aware EI if UCB fails (unlikely)
        z = (f_best - mu) / sigma_safe
        from scipy.stats import norm
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        score = EI / cost_penalty
        if not np.all(np.isfinite(score)):
            score = np.zeros_like(mu)
    
    # Check time budget
    if time.time() > deadline:
        pass
    
    return score
# EVOLVE-BLOCK-END
