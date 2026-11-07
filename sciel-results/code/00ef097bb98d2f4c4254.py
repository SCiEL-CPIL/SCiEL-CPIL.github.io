"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combines GP-UCB (exploration) with a small EI component.
    In 6D, pure EI can be overly conservative. GP-UCB with beta~1 encourages
    exploration of high-uncertainty regions, which helps escape local minima.
    
    We use a weighted combination: score = beta * sigma - mu + alpha * EI
    where beta controls exploration and alpha is a small exploitation term.
    
    Actually, let's stick with a well-tuned GP-UCB since it showed improvement (1.044).
    But we can also try a slightly different approach: use the expected improvement
    but with a different scaling, or use a combination.
    
    Let's use GP-UCB with beta=1.0 which was measured to improve to 1.044.
    To potentially do better, we can try:
    1. A slightly different beta value
    2. A hybrid that blends UCB with EI
    
    Let's try a hybrid: 0.7 * UCB + 0.3 * EI_normalized
    But normalizing EI is tricky. Let's just use UCB with a tuned beta.
    
    Actually, the measured result showed GP-UCB (beta=1) improved from 1.0 to 1.044.
    Let's try beta=1.5 or beta=2 to see if more exploration helps, or if less helps.
    
    Alternatively, we can use a different acquisition: Probability of Improvement (PI)
    or Expected Hypervolume Improvement (EHVI), but those are more complex.
    
    Let's go with a refined GP-UCB: score = -mu + beta * sigma
    with beta=1.0 as the baseline that worked. To potentially improve further,
    we can add a small penalty for very high sigma (to avoid pure noise) or
    use a slightly different formulation.
    
    Actually, let's try: score = -mu + beta * sigma, but clip sigma to avoid
    extreme values dominating. Or use log(sigma) instead of sigma.
    
    Let's try: score = -mu + beta * log1p(sigma)
    This dampens the effect of very high uncertainty.
    
    But wait, the previous measurement showed beta=1 with linear sigma worked.
    Let's try a slightly different approach: use the expected improvement
    but with a "temperature" parameter that makes it less conservative.
    
    Actually, let's just use GP-UCB with beta=1.0 as it was measured to work.
    To potentially do better, we can try adaptive beta or a hybrid.
    
    Let's try: 0.5 * (-mu + sigma) + 0.5 * EI
    This blends exploration and exploitation.
    """
    # Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for blending
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: 60% UCB, 40% EI
    score = 0.6 * ucb_norm + 0.4 * ei_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
