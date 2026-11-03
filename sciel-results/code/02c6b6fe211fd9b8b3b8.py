# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 11

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _penalty_outside(points):
    """
    Calculate penalty for points outside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    Returns a large positive value if any point is outside.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += (0 - y) * 100
        val_left = np.sqrt(3) * x - y
        if val_left < 0:
            penalty += (-val_left) * 100
        val_right = np.sqrt(3) * (1 - x) - y
        if val_right < 0:
            penalty += (-val_right) * 100
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    Uses a row-based symmetric lattice structure (3-4-3-1 points).
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    
    # Topology:
    # Row 0 (y=0): 3 points. Symmetric: (0.5-d0, 0), (0.5, 0), (0.5+d0, 0)
    # Row 1 (y=h1): 4 points. Symmetric: (0.5-1.5*d1, h1), (0.5-0.5*d1, h1), (0.5+0.5*d1, h1), (0.5+1.5*d1, h1)
    # Row 2 (y=h2): 3 points. Symmetric: (0.5-d2, h2), (0.5, h2), (0.5+d2, h2)
    # Row 3 (y=h): 1 point. (0.5, h)
    
    # Variables to optimize:
    # d0, d1, d2, h1, h2
    # Constraints:
    # 0 <= d0 <= 0.5 (so x in [0,1])
    # 0 <= d1 <= (1 - 2*h1/sqrt3) / 3 (so outer points x in [0,1])
    # 0 <= d2 <= (1 - 2*h2/sqrt3) / 2 (so outer points x in [0,1])
    # 0 < h1 < h2 < h
    
    def params_to_points(params):
        d0, d1, d2, h1, h2 = params
        pts = []
        # Row 0
        pts.append((0.5 - d0, 0.0))
        pts.append((0.5, 0.0))
        pts.append((0.5 + d0, 0.0))
        
        # Row 1
        pts.append((0.5 - 1.5 * d1, h1))
        pts.append((0.5 - 0.5 * d1, h1))
        pts.append((0.5 + 0.5 * d1, h1))
        pts.append((0.5 + 1.5 * d1, h1))
        
        # Row 2
        pts.append((0.5 - d2, h2))
        pts.append((0.5, h2))
        pts.append((0.5 + d2, h2))
        
        # Row 3
        pts.append((0.5, h))
        
        return np.array(pts)

    def objective(params):
        pts = params_to_points(params)
        min_area = _min_triangle_area(pts)
        pen = _penalty_outside(pts)
        # Add penalty for violating row ordering or spacing if needed, 
        # but penalty_outside should catch most boundary violations.
        # We also want to ensure h1 < h2.
        if h1 >= h2:
            return 1e6
        if h2 >= h:
            return 1e6
        if h1 <= 0:
            return 1e6
        return -min_area + pen

    # Initial guesses
    # Ideal hexagonal packing suggests specific ratios.
    # Let's try a few starts.
    
    best_score = -float('inf')
    best_params = None
    
    starts = [
        # Start 1: Roughly even spacing
        [0.2, 0.15, 0.1, 0.2, 0.4],
        # Start 2: Compressed bottom
        [0.1, 0.1, 0.05, 0.1, 0.3],
        # Start 3: Expanded bottom
        [0.4, 0.2, 0.15, 0.3, 0.5],
        # Start 4: Randomized
        [0.1 + 0.3*np.random.rand(), 0.05 + 0.15*np.random.rand(), 0.05 + 0.15*np.random.rand(), 0.1 + 0.2*np.random.rand(), 0.3 + 0.2*np.random.rand()]
    ]
    
    np.random.seed(42)
    for _ in range(5):
        starts.append([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(0.05, 0.2),
            np.random.uniform(0.05, 0.2),
            np.random.uniform(0.1, 0.3),
            np.random.uniform(0.3, 0.6)
        ])

    for start in starts:
        # Clip start to valid ranges roughly to help optimizer
        s = start.copy()
        s[0] = np.clip(s[0], 0.01, 0.49)
        s[1] = np.clip(s[1], 0.01, 0.3)
        s[2] = np.clip(s[2], 0.01, 0.2)
        s[3] = np.clip(s[3], 0.01, 0.4)
        s[4] = np.clip(s[4], 0.1, 0.8)
        
        res = minimize(objective, s, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        if res.fun < best_score:
            # Validate
            pts = params_to_points(res.x)
            if _penalty_outside(pts) < 1e-8:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x
    
    if best_params is None:
        # Fallback to first start
        best_params = starts[0]
        
    return params_to_points(best_params)
# EVOLVE-BLOCK-END
