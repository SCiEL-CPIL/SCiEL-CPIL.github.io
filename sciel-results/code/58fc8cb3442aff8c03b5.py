"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: combines Chebyshev polynomials (global smooth structure)
    with localized Gaussian bumps (local detail capture).
    
    The first 5 parameters scale the Chebyshev basis T_0..T_4 (standardized).
    The remaining 5 parameters scale Gaussian bumps centered at 5 evenly-spaced
    points in [0.1, 6], each with a fixed width chosen to cover ~1/5 of the domain.
    
    This hybrid gives BFGS both global polynomial structure and local flexibility,
    potentially capturing features that a pure polynomial of degree 9 misses.
    """
    x_min = 0.1
    x_max = 6.0
    n_points = x.shape[0]
    
    # --- Chebyshev part (first 5 coefficients) ---
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    cheb = np.empty((n_points, 5), dtype=np.float64)
    cheb[:, 0] = 1.0
    cheb[:, 1] = t
    for k in range(2, 5):
        cheb[:, k] = 2.0 * t * cheb[:, k-1] - cheb[:, k-2]
    
    # Standardize Chebyshev columns: var(T_0)=1, var(T_k)=0.5 for k>=1
    cheb_std = np.ones(5, dtype=np.float64)
    cheb_std[1:] = np.sqrt(0.5)
    cheb = cheb / cheb_std[np.newaxis, :]
    
    # --- Gaussian bump part (remaining 5 coefficients) ---
    # Centers evenly spaced in [0.1, 6]
    centers = np.linspace(x_min + 0.5, x_max - 0.5, 5)
    # Width: each bump covers roughly 1/5 of the domain
    width = (x_max - x_min) / 5.0
    
    # Compute Gaussian bumps: exp(-((x - c)/w)^2)
    # Standardize: subtract mean and divide by std for each bump
    gauss = np.empty((n_points, 5), dtype=np.float64)
    for j in range(5):
        gauss[:, j] = np.exp(-((x - centers[j]) / width) ** 2)
    
    # Standardize each Gaussian column
    for j in range(5):
        col = gauss[:, j]
        mean_j = np.mean(col)
        std_j = np.std(col)
        if std_j > 1e-12:
            gauss[:, j] = (col - mean_j) / std_j
        else:
            gauss[:, j] = 0.0
    
    # --- Combine: prediction = cheb @ params[:5] + gauss @ params[5:] ---
    prediction = cheb @ params[:5] + gauss @ params[5:]
    
    return prediction
# EVOLVE-BLOCK-END
