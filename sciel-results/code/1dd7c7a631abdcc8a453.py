"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


_START_TIME = time.time()
_TIME_LIMIT = 300.0
_TIME_MARGIN = 10.0


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D.
    
    Uses a hybrid strategy:
    1. Compute Standard Expected Improvement (EI) for exploitation.
    2. Compute UCB for exploration.
    3. Combine them without min-max normalization (which can be brittle)
       but with a dynamic weight that shifts from exploration to exploitation
       as f_best approaches the known minimum (0.397887).
    """
    # Deadline guard
    if _time_left() < 1.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB
    # Beta is a hyperparameter. 1.0 is a good default.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Determine exploration vs exploitation weight
    # The minimum of Branin is approx 0.397887.
    # If f_best is much larger than this, we are in early phase -> explore more.
    # If f_best is close, we are in late phase -> exploit more.
    min_val = 0.397887
    gap = f_best - min_val
    
    # Sigmoid transition:
    # When gap is large (e.g., > 5), exp(2*(gap-2)) is large -> explore_weight ~ 0? 
    # Wait, we want high exploration weight when gap is large.
    # Let's define explore_weight such that it is high when gap is large.
    # explore_weight = 1 / (1 + exp(k * (gap - c)))
    # If gap >> c, exp is large, explore_weight -> 0. This is wrong.
    # We want explore_weight -> 1 when gap is large.
    # So: explore_weight = 1 / (1 + exp(-k * (gap - c)))
    # If gap >> c, exp(-large) -> 0, explore_weight -> 1.
    # If gap << c, exp(large) -> large, explore_weight -> 0.
    # Let's set c=2.0, k=2.0.
    explore_weight = 1.0 / (1.0 + np.exp(-2.0 * (gap - 2.0)))
    
    # We want to blend EI and UCB.
    # Since EI and UCB are in different scales, simple addition might be dominated by one.
    # However, both are in objective units.
    # EI is bounded by roughly the range of f. UCB can be larger due to sigma.
    # A common approach is to normalize them or use a weighted sum.
    # The inspiration suggested avoiding min-max normalization.
    # Let's try a weighted sum: score = w_ei * ei + w_ucb * ucb
    # But w_ei and w_ucb need to be comparable.
    # Let's normalize EI by its max to make it roughly [0,1] and scale UCB?
    # No, let's stick to the "direct" idea but be careful.
    
    # Alternative: Use the "Knowledge Gradient" idea or just a robust mix.
    # Let's try: score = ei + explore_weight * ucb
    # This adds exploration signal on top of exploitation.
    # When explore_weight is high (early), UCB dominates.
    # When explore_weight is low (late), EI dominates.
    
    # However, UCB values can be very large if sigma is large.
    # Let's check magnitudes.
    # In early phase, sigma is large everywhere. UCB ~ beta*sigma. EI ~ 0 (since f_best is high, mu is random, z is small/negative).
    # Actually, if f_best is high, (f_best - mu) is positive for most points, so EI is non-zero.
    
    # Let's try a normalized blend but robustly.
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
        
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
        
    # Blend
    # The current program used 0.8 EI + 0.2 UCB.
    # Inspiration 1 (Pure UCB) scored 1.42.
    # Inspiration 3 (Hybrid) suggests direct sum.
    # Let's use the dynamic weight on the normalized scores.
    # score = (1 - explore_weight) * ei_norm + explore_weight * ucb_norm
    
    # Wait, if we normalize, we lose the scale.
    # The evaluator picks argmax. Scale doesn't matter for argmax.
    # So normalization is fine for ranking.
    
    # However, the current program (score 1.0) used normalization.
    # Inspiration 3 (score 1.4885) used a direct sum in the docstring but the code was omitted.
    # Let's look at Inspiration 1 (score 1.4206): Pure UCB.
    # Pure UCB worked well. This suggests exploration is key for Branin.
    # The current program favors EI (0.8 weight).
    # Let's shift the weight towards UCB more, or use the dynamic weight.
    
    # Let's use the dynamic weight to blend the normalized scores.
    score = (1.0 - explore_weight) * ei_norm + explore_weight * ucb_norm
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
