"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Confidence-Weighted UCB for cost-aware minimization of Griewank 2D.
    
    Structural change: Instead of blending EI and UCB, we use a UCB where
    the exploration coefficient is modulated by a confidence factor.
    
    The confidence factor = Phi((f_best - mu) / sigma) measures how likely
    the point is to actually improve. This gates the exploration bonus:
    - High confidence (likely improvement): full exploration bonus
    - Low confidence (unlikely improvement): reduced exploration bonus
    
    This prevents the acquisition from being dominated by high-noise regions
    where sigma is large but the point is unlikely to actually improve.
    
    Key properties:
    1. Preserves UCB's linear structure (better for multimodal landscapes)
    2. Strictly penalizes uncertainty in low-confidence regions
    3. Adaptive: as f_best improves, the confidence gate becomes stricter
    4. Cost-aware: divides by evaluation cost
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu and cost are finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Confidence factor: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is the probability of improvement
    z = (f_best - mu) / sigma
    confidence = norm.cdf(z)
    
    # Clip confidence to avoid extreme values
    confidence = np.clip(confidence, 1e-6, 1.0 - 1e-6)
    
    # Adaptive exploration coefficient
    # When f_best is high (early), we want more exploration
    # When f_best is low (late), we want less exploration
    # Use a smooth transition based on f_best
    f_best_norm = np.clip(f_best, 0.0, 1.0)
    
    # Base beta: ranges from 1.5 (early) to 0.5 (late)
    beta_base = 0.5 + 1.0 * f_best_norm
    
    # The confidence-weighted beta
    # When confidence is high, use full beta
    # When confidence is low, reduce beta proportionally
    # Use a power to make the gating sharper
    beta = beta_base * confidence
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher score = more desirable to evaluate
    scores = (-mu + beta * sigma) / cost
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
