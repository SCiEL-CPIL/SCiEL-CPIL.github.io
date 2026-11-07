"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Hypervolume-based acquisition function
# For minimization, we consider the 2D space (objective, -f_best) or similar
# Actually, let's use a different approach: Expected Hypervolume Improvement (EHVI)
# But since we only have a single objective and f_best, let's think differently.
# 
# The directive says: "maximizes the expected increase in the hypervolume of the Pareto front 
# defined by the objective and the current best value"
# 
# In single-objective optimization with a reference point, the hypervolume contribution 
# of a point x with objective f(x) is max(0, f_ref - f(x)) where f_ref is a reference point.
# The "Pareto front" in this context is just the set of evaluated points.
# 
# A simpler interpretation: Use an acquisition function that balances exploration and 
# exploitation more effectively. Let's try a combination of EI and a distance-based term,
# or use Upper Confidence Bound with a different formulation.
#
# Actually, let's implement a proper hypervolume-based approach:
# The hypervolume of a point in 1D (just objective) relative to reference f_ref is:
# HV = max(0, f_ref - f)
# 
# For the expected hypervolume improvement, we need E[max(0, f_ref - Y)] where Y ~ N(mu, sigma^2)
# This is the same as EI but with f_ref instead of f_best!
#
# So EHVI = (f_ref - mu) * Phi((f_ref - mu)/sigma) + sigma * phi((f_ref - mu)/sigma)
# where f_ref is a reference point.
#
# But what should f_ref be? A common choice is the worst observed value or a fixed value.
# Let's use f_ref = f_best + k * sigma_max or similar.
#
# Actually, let me reconsider. The key insight is that hypervolume-based methods naturally
# balance exploration and exploitation. For single-objective, this reduces to a modified EI.
#
# Let's try: score = EI + beta * sigma (UCB-like) but with a different beta schedule,
# or use the actual EHVI formulation.
#
# I'll implement a hybrid: use EI but with an adaptive reference point that encourages
# exploration when uncertainty is high.

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    
    # Hypervolume-based: use a reference point that's worse than f_best
    # to encourage exploration of potentially better regions
    # f_ref = f_best + alpha * max_sigma
    # This makes the "improvement" relative to a worse reference, 
    # which gives more weight to high-uncertainty regions
    
    # Adaptive alpha based on current best and uncertainty
    max_sigma = np.max(sigma)
    if max_sigma > 1e-6:
        # Scale alpha so that the reference point is meaningfully worse
        alpha = 0.5  # Tune this
        f_ref = f_best + alpha * max_sigma
    else:
        f_ref = f_best
    
    # EHVI with adaptive reference
    z_hv = (f_ref - mu) / sigma
    ehvi = (f_ref - mu) * norm.cdf(z_hv) + sigma * norm.pdf(z_hv)
    
    # Combine: use EHVI as it naturally balances exploration/exploitation
    # But ensure it's finite
    scores = ehvi
    
    # Clip to ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
