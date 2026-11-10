"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item in a multiple-constraint knapsack.
    
    Args:
        prizes: array of shape (5, 100) - prize values for each constraint/item
        weights: array of shape (5, 100) - weight costs for each constraint/item
    
    Returns:
        array of shape (100,) - attractiveness score per item (larger = more attractive)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are correct
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        raise ValueError("Inputs must have shape (5, 100)")
        
    # 1. Calculate efficiency ratios: prize per unit weight for each constraint
    eps = 1e-9
    safe_weights = weights + eps
    ratios = prizes / safe_weights  # shape (5, 100)
    
    # 2. Use Harmonic Mean of ratios as the core efficiency score.
    # The harmonic mean is more sensitive to low values (bad constraints) than the geometric mean,
    # which helps in avoiding items that are terrible on just one constraint.
    # H = N / sum(1/x_i)
    inv_ratios = 1.0 / ratios
    sum_inv = inv_ratios.sum(axis=0)
    
    # Avoid division by zero if a ratio was extremely small (though eps prevents exact 0)
    harmonic_scores = 5.0 / sum_inv
    
    # 3. Incorporate Total Prize.
    # Items with higher total prize are generally better, even if efficiency is similar.
    total_prize = prizes.sum(axis=0)
    
    # 4. Combine Efficiency and Prize.
    # We use a multiplicative combination to ensure both are important.
    # harmonic_scores can be small or large. total_prize is in [0, 5].
    # To balance scales, we can normalize or just multiply.
    # Let's normalize total_prize to [0, 1] to prevent it from dominating if prizes are large?
    # Prizes are uniform [0, 1], so total_prize is [0, 5].
    # Harmonic mean of ratios [0, 1]/[0, 1] can be very large if weights are tiny.
    
    # Let's look at the distribution.
    # If weights are ~0.5, ratios ~2. H ~ 2.
    # If one weight is 0.9, ratio ~1. H drops.
    
    # A simple product:
    score = harmonic_scores * total_prize
    
    # 5. Sanitize
    score = np.where(np.isfinite(score) & (score > 0), score, 1e-9)
    
    return score
# EVOLVE-BLOCK-END
