import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    # Iterate over all triples
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # If points are collinear or fewer than 3, area is 0
        return 0.0

def _objective(x):
    """
    Objective function to minimize.
    We want to maximize (Min Triangle Area / Hull Area).
    So we minimize - (Min Triangle Area / Hull Area).
    """
    points = x.reshape(-1, 2)
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    
    if hull_area < 1e-9:
        return 1e9  # Large penalty for degenerate hull
    
    ratio = min_area / hull_area
    return -ratio

def construct():
    # Initialize with a Halton sequence inside the unit circle
    # Halton sequence in base 2 and 3
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        # Map [0,1]^2 to unit circle
        # Use polar coordinates for better distribution?
        # Or just scale and clip?
        # Let's use a simple mapping: x,y in [0,1] -> angle, radius
        angle = 2 * np.pi * x
        radius = np.sqrt(y) # Uniform area distribution
        px = radius * np.cos(angle)
        py = radius * np.sin(angle)
        points.append([px, py])
    
    x0 = np.array(points).flatten()
    
    # Optimize
    # Nelder-Mead is good for small dimensions and non-differentiable functions
    res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    return res.x.reshape(-1, 2)

# EVOLVE-BLOCK-END
