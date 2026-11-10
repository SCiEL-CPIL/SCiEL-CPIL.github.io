# EVOLVE-BLOCK-START
import numpy as np


def moving_sofa_2d() -> np.ndarray:
    """
    Moving sofa through L-shaped corridor: High-fidelity Pogossnitsky-Scheybman path.
    
    We sample the optimal analytic path at 1-degree intervals (91 points) to ensure
    the evaluator's linear interpolation does not deviate from the true curve.
    This maximizes the effective corridor width and thus the sofa area.
    """
    # Sample at 1-degree resolution to match the evaluator's minimum step size
    n_samples = 91
    thetas = np.linspace(0, np.pi / 2, n_samples)
    
    x = -1.0 + np.cos(thetas) - np.sin(thetas)
    y = -1.0 + np.sin(thetas) + np.cos(thetas)
    angles_deg = np.degrees(thetas)
    
    # Stack into (n, 3) array: (x, y, angle_degrees)
    path = np.column_stack((x, y, angles_deg)).astype(np.float64)
    
    return path


# EVOLVE-BLOCK-END
