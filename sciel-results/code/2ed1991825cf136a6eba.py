"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using kernel-weighted denoising.
    
    Strategy:
    1. Use Gaussian kernel weighting to estimate E[Y|X] at a set of evaluation points.
    2. Fit a degree-5 polynomial to these denoised pseudo-labels.
    3. Return coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin: finish by 100s, limit is 120s
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n == 0:
        return np.zeros(6)
    
    # Step 1: Create evaluation grid for denoising
    # Use quantiles to ensure good coverage
    n_eval = min(2000, n)
    eval_x = np.quantile(x, np.linspace(0.01, 0.99, n_eval))
    
    # Step 2: Kernel-weighted denoising
    # Choose bandwidth based on data spread
    x_range = x.max() - x.min()
    if x_range < 1e-10:
        bandwidth = 1.0
    else:
        bandwidth = x_range * 0.05  # 5% of range
    
    # Compute kernel weights and weighted means for each eval point
    pseudo_y = np.zeros(n_eval)
    for i in range(n_eval):
        if time.time() > deadline - 10:
            # If running out of time, use remaining eval points with simple mean
            pseudo_y[i:] = y.mean()
            break
        # Gaussian kernel weights
        diff = (x - eval_x[i]) / bandwidth
        weights = np.exp(-0.5 * diff**2)
        
        # Weighted mean
        w_sum = weights.sum()
        if w_sum > 1e-10:
            pseudo_y[i] = (weights * y).sum() / w_sum
        else:
            pseudo_y[i] = y.mean()
    
    # Step 3: Fit degree-5 polynomial to denoised pseudo-labels
    try:
        coeff = np.polynomial.polynomial.polyfit(eval_x, pseudo_y, 5)
    except Exception:
        # Fallback to linear fit
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
        except Exception:
            coeff[0] = y.mean()
    
    return coeff
# EVOLVE-BLOCK-END
