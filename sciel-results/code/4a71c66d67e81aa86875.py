# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition for minimizing Branin 2D.
    
    This function approximates the expected improvement in the hypervolume
    of the Pareto front defined by the objective value and the uncertainty.
    It effectively balances exploration (high sigma) and exploitation (low mu)
    by identifying points that would expand the non-dominated region in the
    (mean, -std) space.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Hypervolume contribution scores; higher is better.
    """
    # Ensure sigma is positive to avoid division by zero or log issues
    sigma = np.maximum(sigma, 1e-12)
    
    # 1. Define a reference point for the Hypervolume calculation.
    # The reference point is chosen such that points with mu < f_best and 
    # significant sigma contribute positively.
    # We use f_best as the threshold for the objective axis.
    # For the uncertainty axis, we consider the contribution relative to 0.
    
    # 2. Compute a proxy for the "expected hypervolume improvement".
    # In a 2D minimization problem (minimize mu, maximize -sigma? No, we minimize risk too? 
    # Actually, in multi-objective BO, we usually minimize f and maximize confidence? 
    # Here, let's stick to the single-objective context but use HV logic.
    # 
    # A simpler and robust structural change: 
    # Use the "Expected Hypervolume Improvement" (EHVI) concept adapted for 
    # single-objective by considering the distribution.
    # However, true EHVI is expensive. 
    # 
    # Let's use a "Pareto-front inspired" score:
    # Score = (f_best - mu) * (1 + alpha * sigma / (f_best - mu + epsilon)) ? No.
    #
    # Let's try a different structural approach: 
    # "Probability of Improvement" (PI) combined with "Uncertainty" in a multiplicative way,
    # or simply use the "Upper Confidence Bound" (UCB) which is structurally different 
    # from EI (additive vs multiplicative-like integral). 
    # 
    # Wait, the directive specifically asked for: "Hypervolume-based acquisition function 
    # that maximizes the expected increase in the hypervolume of the Pareto front 
    # defined by the objective and the current best value".
    #
    # Let's implement a simplified EHVI approximation.
    # For a single objective, the "Pareto front" of the candidates can be viewed 
    # in the space of (mu, sigma). We want to find points that are non-dominated 
    # in terms of having low mu and high potential (low sigma? or high sigma for exploration?).
    # 
    # Actually, a common heuristic that mimics HV contribution in 2D (x, y) where 
    # we minimize x and maximize y (or minimize -y) is:
    # Contribution_i = (x_ref - x_i) * (y_i - y_ref)
    # 
    # Let x = mu (minimize)
    # Let y = -sigma (maximize? No, sigma is uncertainty. High sigma is good for exploration).
    # So we maximize sigma?
    # If we treat the problem as minimizing mu and maximizing sigma (exploration),
    # the Pareto front consists of points that are best in mu and best in sigma.
    # 
    # Reference point: 
    # x_ref = f_best (we only care about improvements below f_best)
    # y_ref = 0 (sigma is always positive)
    #
    # Hypervolume contribution of point i relative to ref:
    # width = max(0, f_best - mu_i)
    # height = max(0, sigma_i)
    # Score_i = width * height
    #
    # This is structurally different from EI. EI is an integral. This is a product.
    # It strongly favors points with both low mean AND high uncertainty.
    # This might be too exploratory. 
    # 
    # Let's refine: 
    # Standard EI: E[max(0, f_best - f)]
    # This new score: (f_best - mu) * sigma
    # 
    # Let's test this logic.
    # If mu is very low, width is large. If sigma is high, height is large.
    # This is essentially a "risk-seeking" acquisition.
    #
    # To make it more robust and closer to a standard acquisition while keeping the 
    # "Hypervolume" structural flavor, we can normalize or use a different reference.
    # 
    # Let's try: 
    # scores = np.maximum(0, f_best - mu) * sigma
    #
    # This is computationally cheap and structurally distinct from EI.
    
    width = np.maximum(0, f_best - mu)
    height = sigma
    scores = width * height
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
