"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 5 Radial Basis Functions + 5 Polynomial terms.
    Centered inputs for better conditioning.
    
    RBF centers (in centered space, centered around 1.25):
    c0: (0, 0, 0)
    c1: (-0.6, -0.6, -0.6)
    c2: (0.6, 0.6, 0.6)
    c3: (-0.6, 0.6, 0.0)
    c4: (0.6, -0.6, 0.0)
    
    Polynomial terms (in centered space):
    p5: 1
    p6: xc
    p7: yc
    p8: zc
    p9: xc^2 + yc^2 + zc^2
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Center inputs around midpoint of [0.5, 2]
    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25

    # RBF Centers in centered coordinates
    # Adjusted to be slightly more spread out towards corners
    cx = np.array([0.0, -0.6, 0.6, -0.6, 0.6])
    cy = np.array([0.0, -0.6, 0.6, 0.6, -0.6])
    cz = np.array([0.0, -0.6, 0.6, 0.0, 0.0])
    
    # Adjusted sigma for potentially better conditioning/coverage
    sigma2 = 0.09  # sigma = 0.3

    # Calculate RBF values efficiently
    # dist_sq[i, j] = ||v_i - c_j||^2
    # Expand: ||v_i||^2 - 2 v_i dot c_j + ||c_j||^2
    
    r2 = xc*xc + yc*yc + zc*zc  # (N,)
    
    # Cross term: (N, 5)
    cross = xc[:, np.newaxis] * cx[np.newaxis, :] + \
            yc[:, np.newaxis] * cy[np.newaxis, :] + \
            zc[:, np.newaxis] * cz[np.newaxis, :]
    
    # Center norms: (5,)
    c2 = cx*cx + cy*cy + cz*cz
    
    dist_sq = r2[:, np.newaxis] - 2.0 * cross + c2[np.newaxis, :]
    
    # Clip to avoid negative values due to floating point
    dist_sq = np.maximum(dist_sq, 0.0)
    
    rbf_vals = np.exp(-dist_sq / sigma2)  # (N, 5)
    
    # Polynomial terms
    poly_1 = np.ones_like(x)
    poly_x = xc
    poly_y = yc
    poly_z = zc
    poly_r2 = r2  # xc^2 + yc^2 + zc^2
    
    # Combine: p[0:5] for RBFs, p[5:10] for Polys
    pred = (rbf_vals @ p[0:5] + 
            poly_1 * p[5] + 
            poly_x * p[6] + 
            poly_y * p[7] + 
            poly_z * p[8] + 
            poly_r2 * p[9])
            
    return pred
# EVOLVE-BLOCK-END
