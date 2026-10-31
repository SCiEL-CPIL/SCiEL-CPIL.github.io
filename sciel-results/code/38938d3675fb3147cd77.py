"""
Optimized program for Heilbronn convex regions task.
Uses the property that the ratio of minimum triangle area to convex hull area
is affine invariant. The regular polygon maximizes this ratio for a given number of points.
"""
import numpy as np

N = 13

def construct():
    # Generate vertices of a regular N-gon on the unit circle.
    # The score is invariant to scaling and rotation, so unit circle is sufficient.
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    return points
