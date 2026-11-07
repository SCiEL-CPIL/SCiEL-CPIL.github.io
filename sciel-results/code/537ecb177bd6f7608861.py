"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 145.0  # safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: 5 Radial Basis Functions + 5 Polynomial terms.
    RBF centers:
    c0: (1.25, 1.25, 1.25)
    c1: (0.75, 0.75, 0.75)
    c2: (1.75, 1.75, 1.75)
    c3: (0.75, 1.75, 1.25)
    c4: (1.75, 0.75, 1.25)
    Sigma = 0.5
    
    Polynomial terms:
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

    xc = x - 1.25
    yc = y - 1.25
    zc = z - 1.25

    # RBF Centers
    cx = np.array([1.25, 0.75, 1.75, 0.75, 1.75])
    cy = np.array([1.25, 0.75, 1.75, 1.75, 0.75])
    cz = np.array([1.25, 0.75, 1.75, 1.25, 1.25])
    
    sigma2 = 0.25 # sigma^2

    # Calculate RBF values
    xx = xc[:, np.newaxis]
    yy = yc[:, np.newaxis]
    zz = zc[:, np.newaxis]
    
    ccx = cx[np.newaxis, :]
    ccy = cy[np.newaxis, :]
    ccz = cz[np.newaxis, :]
    
    dist_sq = (xx - ccx)**2 + (yy - ccy)**2 + (zz - ccz)**2
    rbf_vals = np.exp(-dist_sq / sigma2) # (N, 5)
    
    # Polynomial terms
    poly_1 = np.ones_like(x)
    poly_x = xc
    poly_y = yc
    poly_z = zc
    poly_r2 = xc*xc + yc*yc + zc*zc
    
    # Combine
    # p[0:5] for RBFs, p[5:10] for Polys
    pred = (rbf_vals @ p[0:5] + 
            poly_1 * p[5] + 
            poly_x * p[6] + 
            poly_y * p[7] + 
            poly_z * p[8] + 
            poly_r2 * p[9])
            
    return pred
# EVOLVE-BLOCK-END
