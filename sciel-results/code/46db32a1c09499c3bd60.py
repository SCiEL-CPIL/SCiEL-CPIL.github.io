import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def min_triangle_area(points):
    """Compute minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except:
        return 0.0

def score(points):
    """Score = min triangle area / hull area."""
    ma = min_triangle_area(points)
    ha = hull_area(points)
    if ha < 1e-10:
        return -1.0
    return ma / ha

def construct():
    best_score = -1.0
    best_points = None
    
    # Multiple initializations
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. 4x4 grid minus 3 corners
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    # Remove 3 points to get 13
    indices_to_remove = [0, 15, 12]  # (0,0), (3,3), (3,0)
    keep = [i for i in range(16) if i not in indices_to_remove]
    initializations.append(grid[keep])
    
    # 3. 3x5 grid minus 2
    grid35 = np.array([[i, j] for i in range(3) for j in range(5)], dtype=float)
    keep35 = [i for i in range(15) if i not in [0, 14]]
    initializations.append(grid35[keep35])
    
    # 4. Hexagonal-ish pattern
    hex_points = []
    # Center
    hex_points.append([0, 0])
    # Ring 1: 6 points
    for k in range(6):
        angle = k * np.pi / 3
        hex_points.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 6 more points (total 13)
    for k in range(6):
        angle = (k + 0.5) * np.pi / 3
        hex_points.append([2*np.cos(angle), 2*np.sin(angle)])
    initializations.append(np.array(hex_points))
    
    for init in initializations:
        # Normalize to unit area for consistent optimization
        ha = hull_area(init)
        if ha > 1e-10:
            init = init / np.sqrt(ha)  # Scale to unit area
        
        # Optimize
        def neg_score(x):
            pts = x.reshape(N, 2)
            s = score(pts)
            return -s
        
        res = minimize(neg_score, init.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < best_score:
            s = -res.fun
            if s > best_score:
                best_score = s
                best_points = res.x.reshape(N, 2)
    
    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
