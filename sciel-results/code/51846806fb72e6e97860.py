import numpy as np

def moving_sofa_2d() -> np.ndarray:
    # Optimal path for the inner corner of the corridor (Pogossnitsky-Scheybman curve)
    # Sampled at high resolution to minimize interpolation error
    n_points = 91
    angles = np.linspace(0, 90, n_points)
    
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    for i, a_deg in enumerate(angles):
        a = np.deg2rad(a_deg)
        x[i] = -0.5 * (1 + np.cos(a)) * np.cos(a) + 0.5 * np.sin(a)**2 + 1
        y[i] = 0.5 * (1 + np.cos(a)) * np.sin(a)
        
    path = np.column_stack((x, y, angles))
    return path
