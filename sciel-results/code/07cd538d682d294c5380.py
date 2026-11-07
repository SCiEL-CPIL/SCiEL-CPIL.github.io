import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Gaussian RBF network for nonlinear oscillator acceleration.
    
    10 RBFs with centers spread across (t, x, v) space.
    Params are the RBF weights.
    """
    # Input ranges: t in [30, 50], x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    # Centers: spread across the domain
    # t centers: 30, 34, 38, 42, 46 (5 values)
    # x centers: -0.15, 0.0, 0.15 (3 values)  
    # v centers: -0.25, 0.0, 0.25 (3 values)
    # That's 5*3*3 = 45 combos, too many. Use 10 distinct centers.
    
    # Define 10 centers in (t, x, v) space
    # Spread t evenly, and vary x and v
    t_centers = np.array([32.0, 35.0, 38.0, 41.0, 44.0, 47.0, 33.0, 36.0, 40.0, 43.0])
    x_centers = np.array([-0.15, 0.0, 0.15, -0.10, 0.10, -0.05, 0.05, 0.0, -0.12, 0.12])
    v_centers = np.array([-0.20, 0.0, 0.20, -0.15, 0.15, -0.10, 0.10, 0.0, -0.25, 0.25])
    
    # Widths: scaled to domain sizes
    # t range ~20, x range ~0.38, v range ~0.70
    # Use sigma_t = 5, sigma_x = 0.1, sigma_v = 0.15
    sigma_t = 5.0
    sigma_x = 0.10
    sigma_v = 0.15
    
    # Compute squared distances in normalized space
    dt = (t - t_centers) / sigma_t
    dx = (x - x_centers) / sigma_x
    dv = (v - v_centers) / sigma_v
    
    # Squared distance: dt^2 + dx^2 + dv^2
    # Shape: (n_samples, 10)
    sq_dist = dt**2 + dx**2 + dv**2
    
    # Gaussian RBFs: exp(-0.5 * sq_dist)
    rbf = np.exp(-0.5 * sq_dist)
    
    # Weighted sum
    return rbf @ params
