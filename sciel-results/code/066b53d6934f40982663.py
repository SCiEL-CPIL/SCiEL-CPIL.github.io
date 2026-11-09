"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (griewank 2d).
    
    Uses a hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB)
    with adaptive exploration weight, scaled by evaluation cost.
    
    The approach:
    1. Compute EI for exploitation (high-confidence improvements)
    2. Compute LCB for exploration (uncertain regions)
    3. Blend them with an adaptive weight that favors exploitation as data grows
    4. Scale by 1/cost to account for evaluation expense
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Small epsilon to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate Z-score for Expected Improvement
    # For minimization: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Exploitation component: Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Exploration component: Lower Confidence Bound for minimization
    # LCB = mu - beta * sigma (lower is better, so we want low mu and high sigma)
    # We use a moderate beta that balances exploration
    beta = 1.5
    lcb = mu - beta * sigma
    
    # Normalize both components to [0, 1] range for blending
    # EI is always non-negative, so we can use it directly
    # LCB can be negative, so we need to handle that
    
    # Adaptive blending: as f_best gets closer to 0 (the minimum), 
    # we should exploit more. Use a smooth transition.
    # When f_best is large (early), explore more; when small (late), exploit more.
    # Scale f_best to get an exploration weight
    # f_best for griewank is in [0, ~1], so we can use it directly
    # exploration_weight = 1 / (1 + f_best) would work, but let's be more nuanced
    
    # Use a sigmoid-like function: weight for exploitation increases as f_best decreases
    # When f_best is near 0, weight_exploit -> 1 (pure exploitation)
    # When f_best is large, weight_exploit -> 0.5 (balanced)
    # This is a heuristic based on the known range of griewank
    
    # Actually, let's use a simpler approach: blend with a fixed weight
    # but make the exploration term less aggressive
    
    # Hybrid: primarily EI with a small LCB component for exploration
    # The LCB term helps escape local minima in griewank
    
    # Normalize LCB to be comparable to EI
    # EI is in objective units, LCB is also in objective units (but inverted)
    # We want high scores to be better, so we negate LCB
    lcb_score = -lcb  # Higher is better (lower mu, higher sigma)
    
    # Combine: weight * EI + (1 - weight) * normalized_LCB
    # Use a fixed weight that favors exploitation but allows some exploration
    weight_exploit = 0.7
    
    # Scale LCB to be comparable to EI
    # EI is typically in [0, f_best], LCB score can be large
    # Let's normalize LCB by f_best to make it comparable
    if f_best > eps:
        lcb_normalized = lcb_score / f_best
    else:
        lcb_normalized = lcb_score
    
    # Clip to reasonable range
    lcb_normalized = np.clip(lcb_normalized, 0, 10)
    
    # Combined score
    combined = weight_exploit * ei + (1 - weight_exploit) * lcb_normalized
    
    # Cost-aware: divide by cost to account for evaluation expense
    # Cost is >= 1.0, so this is a mild normalization
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
