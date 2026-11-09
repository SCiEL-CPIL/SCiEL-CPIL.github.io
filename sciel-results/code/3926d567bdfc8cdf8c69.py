import numpy as np

def moving_sofa_2d() -> np.ndarray:
    # New approach: Path that moves the inner corner down to keep the sofa
    # within the horizontal arm longer, then transitions to the vertical arm.
    # We use a higher resolution and a modified curve.
    n_points = 200
    angles = np.linspace(0, 90, n_points)
    t = angles / 90.0
    
    # x moves linearly
    x = -2.0 * t
    
    # y moves down (negative) to keep v positive in the local frame
    # during the first half, then adjusts.
    # Using a cosine-based bump that goes negative.
    y = -0.4 * np.sin(np.pi * t)
    
    path = np.column_stack([x, y, angles])
    return path
