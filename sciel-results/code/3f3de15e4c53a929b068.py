import numpy as np

N = 14

def construct():
    # Generate vertices of a regular 14-gon
    # Radius 1 is sufficient as score is scale-invariant
    angles = 2 * np.pi * np.arange(N) / N
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points
