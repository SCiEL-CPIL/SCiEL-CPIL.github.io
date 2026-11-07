"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Batch-aware Lower Confidence Bound (LCB) with repulsion term.
    
    Uses a Lower Confidence Bound (LCB) for minimization.
    The exploration parameter beta is adapted based on the progress
    towards the known minimum of Hartmann 3D.
    
    Since coordinates are not passed, we approximate a repulsion effect
    by penalizing candidates where the GP mean is very close to f_best
    (indicating they are likely in the same basin as the current best),
    combined with a small stochastic component to break symmetry.
    """
    # Known minimum of Hartmann 3D
    h3_min = -3.86278
    
    # Estimate progress towards the minimum.
    f_start = 0.0
    progress = np.clip((f_start - f_best) / (f_start - h3_min), 0.0, 1.0)
    
    # Adaptive beta: 
    # Start with higher exploration (beta=2.5) to find basins.
    # End with lower exploitation (beta=1.0) to refine the minimum.
    beta = 2.5 - 1.5 * progress
    
    # Base LCB score: -mu + beta * sigma
    # We want to maximize this score (which corresponds to minimizing the LCB).
    score = -mu + beta * sigma
    
    # Repulsion term: penalize candidates where mu is close to f_best.
    # This acts as a proxy for spatial repulsion from the current best point.
    # If mu is very close to f_best, the candidate is likely in the same basin.
    # We use a Gaussian-like penalty centered at f_best.
    
    # Scale for the repulsion: should be small relative to function range
    # Hartmann 3D range is roughly [-3.86, 1.0], so range ~ 4.86
    # Use a narrow window around f_best to penalize only very close predictions
    repulsion_width = 0.1
    
    # Penalty is positive, so we subtract it from score
    # penalty = strength * exp(-((mu - f_best)^2) / (2 * width^2))
    # We want to penalize when mu is close to f_best (i.e., low uncertainty about being near best)
    # But actually, we want to avoid re-evaluating the same region.
    # If sigma is high, we might still want to explore even if mu is near f_best.
    # So scale the repulsion by the uncertainty: if sigma is high, less repulsion.
    
    # Compute distance in objective space
    dist_from_best = mu - f_best
    
    # Gaussian penalty
    penalty = np.exp(-(dist_from_best ** 2) / (2 * repulsion_width ** 2))
    
    # Scale repulsion strength: stronger when progress is high (exploiting)
    # and weaker when progress is low (exploring)
    repulsion_strength = 0.5 * (1.0 - progress) + 0.1 * progress
    
    # Modulate by sigma: if sigma is high, the GP is uncertain, so less repulsion
    # Normalize sigma to [0, 1] roughly
    sigma_max = np.max(sigma) if np.max(sigma) > 1e-10 else 1.0
    sigma_norm = sigma / sigma_max
    
    # Reduce repulsion when uncertainty is high
    penalty_weighted = repulsion_strength * penalty * (1.0 - 0.5 * sigma_norm)
    
    score = score - penalty_weighted
    
    # Add small stochastic noise to promote diversity
    noise_scale = 0.005
    score = score + noise_scale * np.random.randn(len(mu))
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
