import numpy as np

def moving_sofa_2d() -> np.ndarray:
    angles = np.linspace(0, 90, 100)
    t = angles / 90.0
    x = -2.0 * t
    y = 0.6 * np.sin(np.pi * t)
    path = np.column_stack([x, y, angles])
    return path
