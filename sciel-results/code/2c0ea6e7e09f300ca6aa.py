"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Strategy: Pure Probability of Improvement (PI).
    
    The current program uses an adaptive LCB (Lower Confidence Bound) with a cost-dependent
    exploration parameter (kappa). While LCB is a standard choice, it balances exploration
    and exploitation in a way that might be suboptimal for this specific cost structure 
    (where cost is roughly proportional to distance from origin, and the minimum is near origin).
    
    PI(x) = P(f(x) < f_best) = norm.cdf((f_best - mu(x)) / sigma(x))
    
    This metric is purely exploitative: it scores a point based solely on the probability
    that it will improve upon the current best. It ignores the magnitude of the potential
    improvement (unlike EI) and ignores the uncertainty in a multiplicative way (unlike LCB/UCB).
    
    Why PI might be better here:
    1. **Cost Awareness via Correlation**: Since the cost is correlated with location (higher cost = further from origin), and the Rosenbrock minimum is at the origin (low cost), points with low cost are inherently more likely to be near the minimum. The GP posterior `mu` and `sigma` will reflect this. By using PI, we strongly favor points where the GP *predicts* a low value. If the GP has learned that low-cost regions have lower `mu`, PI will naturally concentrate search there without needing explicit cost weighting in the formula.
    
    2. **Simplicity and Robustness**: PI is less sensitive to the scale of `sigma` than LCB. In regions of high uncertainty (early on), LCB with high kappa might explore widely. PI will focus on where the *probability* of improvement is highest. For a 2D problem with a smooth function like Rosenbrock, once the GP has a few points, the probability of improvement becomes a very sharp signal for the basin of attraction.
    
    3. **Avoiding Over-exploration**: The previous LCB approach with adaptive kappa might have been over-exploring expensive regions if `sigma` was high there. PI dampens the effect of high `sigma` because the z-score `(f_best - mu)/sigma` becomes smaller (less positive) if `sigma` is large, unless `mu` is also very low. This effectively creates a "soft" penalty for uncertainty that is not as aggressive as a linear penalty in LCB.
    
    This is a structural change from a *bound-based* (LCB) acquisition to a *probability-based* (PI) acquisition.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Probability of Improvement (PI)
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    z = (f_best - mu) / sigma
    score = norm.cdf(z)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
         # Fallback to simple exploitation: -mu
         score = -mu
    
    # Ensure finite and in [0, 1] range (CDF is already)
    score = np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
