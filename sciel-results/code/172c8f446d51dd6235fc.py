import numpy as np


def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    polygon = np.stack([x, y], axis=1)
    return polygon
