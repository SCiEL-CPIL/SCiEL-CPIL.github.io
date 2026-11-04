import numpy as np
from scipy.optimize import minimize

N = 12
EPS = 0.02  # margin for non-overlap

def inner_vertices(phi, spacing):
    """
    Generate vertices for a 3x4 grid of unit hexagons.
    Each hexagon is rotated by phi and the grid is also rotated by phi.
    """
    verts = []
    for i in (-1, 0, 1):
        for j in (-1.5, -0.5, 0.5, 1.5):
            cx, cy = 2.0 * i, (np.sqrt(3) + spacing) * j
            # Rotate center by phi
            c = np.array([cx, cy])
            c = np.array([np.cos(phi) * c[0] - np.sin(phi) * c[1],
                          np.sin(phi) * c[0] + np.cos(phi) * c[1]])
            for k in range(6):
                a = phi + np.pi / 3 * k
                verts.append([c[0] + np.cos(a), c[1] + np.sin(a)])
    return np.array(verts)

def s_min(params, verts):
    """
    Calculate the minimum side length of the outer hexagon needed to contain
    the given vertices (inner hexagons).
    params: [phi, theta, cx, cy]
    verts: array of vertices of the inner hexagons
    """
    phi, theta, cx, cy = params
    c = np.array([cx, cy])
    # The outer hexagon has 6 sides. The distance from center to side is s * cos(pi/6).
    # We need max_{v in verts, n in normals} (v - c) . n <= s * cos(pi/6)
    # So s >= max_{v in verts, n in normals} (v - c) . n / cos(pi/6)
    maxdot = -np.inf
    for i in range(6):
        a = theta + np.pi / 6 + np.pi / 3 * i
        n = np.array([np.cos(a), np.sin(a)])
        dots = verts @ n - (c @ n)
        maxdot = max(maxdot, dots.max())
    return maxdot / np.cos(np.pi / 6)

def construct():
    best = None
    best_s = np.inf
    
    # We optimize phi, theta, cx, cy, and spacing.
    # However, spacing is a bit tricky because it affects the vertices.
    # Let's fix spacing to a small value for now, or optimize it?
    # If we optimize spacing, the problem becomes non-convex in a different way.
    # The current approach fixes the grid structure (3x4) and just rotates/shifts it.
    # This is a strong heuristic.
    
    # Let's try a grid of initial values.
    for phi0 in np.linspace(0, np.pi / 3, 5):
        for theta0 in np.linspace(0, np.pi / 3, 5):
            verts = inner_vertices(phi0, EPS)
            x0 = [phi0, theta0, 0.0, 0.0]
            res = minimize(s_min, x0, args=(verts,), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            s = res.fun
            if s < best_s:
                best_s = s
                best = res.x

    phi, theta, cx, cy = best
    s = best_s
    
    # Build inner list
    inner = []
    for i in (-1, 0, 1):
        for j in (-1.5, -0.5, 0.5, 1.5):
            cxl, cyl = 2.0 * i, (np.sqrt(3) + EPS) * j
            c = np.array([cxl, cyl])
            c = np.array([np.cos(phi) * c[0] - np.sin(phi) * c[1],
                          np.sin(phi) * c[0] + np.cos(phi) * c[1]])
            inner.append([c[0], c[1], np.degrees(phi)])
            
    return inner, (cx, cy), s, np.degrees(theta)
