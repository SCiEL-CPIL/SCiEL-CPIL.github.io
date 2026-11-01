import numpy as np

N = 14

def construct():
    # Generate vertices of a regular N-gon inscribed in the unit circle.
    # This configuration is symmetric and maximizes the minimum distance between points,
    # which correlates with maximizing the minimum triangle area for points on the convex hull.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points
