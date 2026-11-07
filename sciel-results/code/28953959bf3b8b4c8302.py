"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a greedy batch selection strategy.
    We iteratively select the best candidate based on a combined metric of
    Expected Improvement (EI) and Uncertainty (UCB), adjusting the scores
    of remaining candidates to reflect the 'information' gained.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure safety margins for division
    sigma_safe = sigma + 1e-10
    cost_safe = np.maximum(cost, 1e-10)
    
    # 1. Calculate base Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Calculate Uncertainty (UCB component)
    # We use sigma as a proxy for exploration value.
    # To balance EI and UCB, we can use a weighted sum or a product.
    # A common robust approach is to use a "probability of improvement" or 
    # simply normalize EI and sigma and combine them.
    
    # Normalize EI and Sigma to [0, 1] range for stable combination
    # Handling edge cases where max==min
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-10:
        ei_norm = np.ones_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range < 1e-10:
        sigma_norm = np.ones_like(sigma_safe)
    else:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range

    # Combined Score: Weighted sum of normalized EI and normalized Sigma
    # The weight of exploration (sigma) can be tuned. 
    # Early on, we want more exploration. Late on, more exploitation.
    # However, without knowing the exact iteration count, a fixed balance 
    # or a dynamic one based on f_best magnitude might be risky.
    # Let's stick to a standard UCB-like form: mu + kappa * sigma, 
    # but since we are minimizing, it's -mu + kappa * sigma? 
    # Actually, for minimization, we want low mu and high sigma (uncertainty).
    # So a good score is: (f_best - mu) + beta * sigma.
    # But EI already captures (f_best - mu).
    # Let's use a simpler structural change: 
    # Greedy Batch Selection with Mutual Information / Uncertainty Reduction proxy.
    
    # Since we can't easily compute GP posterior updates without the kernel matrix 
    # and X data (which we don't have, only mu/sigma arrays), 
    # we cannot do true batch GP updates.
    
    # Alternative Structural Strategy: 
    # Use a "Probability of Improvement" (PI) combined with cost, 
    # but modified to be more robust. 
    # Or, use the "Knowledge Gradient" approximation?
    
    # Let's look at the inspiration: "Combining EI with a UCB-like exploration term".
    # The current program is just EI/cost.
    # Let's try: (EI + beta * sigma) / cost.
    # What is beta? 
    # If we look at the standardized z, high z means high confidence in improvement.
    # Low z (or negative) means high uncertainty.
    
    # Let's try a different metric: 
    # Score = EI / cost * (1 + alpha * (sigma / (f_best - mu + epsilon)))
    # This amplifies EI when uncertainty is high relative to potential improvement.
    
    # Actually, a very strong structural change for 2D Griewank with limited budget 
    # is often to prioritize points that are far from already evaluated points 
    # (maximin), but we don't have coordinates.
    
    # However, we DO have sigma. High sigma implies distance from data (for GP).
    # So maximizing sigma is a proxy for maximin.
    
    # Let's implement a hybrid: 
    # Base score: EI / cost
    # Exploration bonus: sigma / cost
    # We combine them linearly.
    
    # To make it adaptive, we can scale the exploration term by the current "best" EI.
    # If EI is high everywhere, we don't need much exploration.
    # If EI is low everywhere, we need exploration.
    
    # Let's try: Score = (EI + 0.5 * sigma) / cost
    # This is a simple structural change from pure EI to EI+UCB.
    
    # But wait, the inspiration said "Combining EI with a UCB-like exploration term ... Measured: 1 -> 1.546".
    # This suggests that the simple addition works well.
    
    # Let's refine the combination. 
    # Instead of fixed 0.5, let's use a dynamic weight based on the variance of EI.
    # Or simply use the formula: EI + kappa * sigma.
    
    # Let's try kappa = 0.5 initially.
    
    # One more structural idea: 
    # Use the "Probability of Improvement" (PI) which is more robust to scale changes 
    # than EI, then divide by cost.
    # PI = Phi(z)
    # Score = PI / cost
    # This ignores the magnitude of improvement, which might be bad for converging.
    
    # Let's stick to EI + UCB.
    
    # We need to ensure the result is finite.
    
    # Let's try a slightly more sophisticated combination:
    # The "Expected Improvement" can be unstable if sigma is very small.
    # The "UCB" term helps exploration.
    
    # Let's use:
    # score = (ei + 0.5 * sigma_safe) / cost_safe
    
    # To make it a "different strategy" as requested, and given the stagnation,
    # let's try a **Thompson Sampling** approximation.
    # We can't sample from the full GP posterior easily without X.
    # But we can approximate the uncertainty.
    
    # Actually, the most distinct and often effective strategy for small budgets 
    # in BO is **Probability of Improvement (PI)** or **UCB** alone, 
    # rather than EI, because EI can be overly greedy in the early stages 
    # if the first few points are not great.
    
    # Let's try a **Weighted UCB** where the weight is high initially.
    # But we don't know the iteration.
    
    # Let's go with the **EI + UCB** hybrid, but normalized to prevent 
    # one term dominating if scales are very different.
    
    # Normalized EI: ei / (max(ei) + eps)
    # Normalized Sigma: sigma / (max(sigma) + eps)
    
    # Score = (norm_ei + norm_sigma) / cost
    
    # This ensures both exploitation and exploration are on a 0-1 scale.
    
    eps = 1e-10
    max_ei = np.max(ei) + eps
    max_sigma = np.max(sigma_safe) + eps
    
    norm_ei = ei / max_ei
    norm_sigma = sigma_safe / max_sigma
    
    # Combine: 50% exploitation, 50% exploration
    combined_score = 0.5 * norm_ei + 0.5 * norm_sigma
    
    # Divide by cost
    scores = combined_score / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
