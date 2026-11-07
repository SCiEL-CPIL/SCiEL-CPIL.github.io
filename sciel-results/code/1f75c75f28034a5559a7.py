# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-entropy heuristic for Bayesian optimization.
    
    We approximate the expected information gain about the global minimum
    per unit cost. The key insight is that evaluating a point provides
    information proportional to its uncertainty (sigma) and its potential
    to improve upon f_best. We combine Expected Improvement (EI) with an
    information gain term that rewards high uncertainty in promising regions.
    
    The score is: (EI + alpha * sigma * promising_weight) / (cost + eps)
    
    where promising_weight encourages exploration in regions where mu is low
    (close to the expected minimum) but uncertainty is high.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Information gain term: reward high sigma in regions where improvement is likely
    # Use a soft weight that is high when mu is much lower than f_best (promising)
    # and sigma is high (uncertain)
    # The weight peaks when mu is well below f_best but not too far (to avoid
    # chasing unrealistic low values)
    
    # A good proxy: sigma * exp(-(f_best - mu)^2 / (2 * scale^2))
    # where scale controls how broad the "promising" region is
    # As f_best decreases, we want to focus more tightly
    
    # Adaptive scale: start broad, narrow as we approach the optimum
    if f_best > 0:
        scale = max(0.5, f_best)
    else:
        scale = 0.5
    
    # Promising weight: Gaussian centered at mu = f_best - scale (i.e., where improvement is likely)
    # Actually, let's use a simpler approach: weight by how much lower mu is than f_best,
    # but with a saturation to avoid over-weighting extreme values
    
    # Use: weight = tanh((f_best - mu) / (f_best + eps)) * sigma
    # This gives high weight when mu << f_best, saturating at 1
    if f_best > 0:
        promising_weight = np.tanh((f_best - mu) / (f_best + eps))
    else:
        promising_weight = np.zeros_like(mu)
    
    # Information gain: sigma weighted by how promising the region is
    info_gain = sigma_safe * np.maximum(promising_weight, 0.0)
    
    # Adaptive alpha: start with more exploration, shift to exploitation
    # When f_best is high (early), explore more; when low (late), exploit more
    if f_best > 1.0:
        alpha = 0.3
    elif f_best > 0.3:
        alpha = 0.2
    else:
        alpha = 0.1
    
    # Combine: (EI + alpha * info_gain) / cost
    score = (ei + alpha * info_gain) / (cost + eps)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
