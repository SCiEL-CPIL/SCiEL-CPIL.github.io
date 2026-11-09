# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    timeout = 630.0  # Safety margin for 600s limit
    
    def compute_score(points):
        """
        Compute the ratio A_min / A_hull.
        Returns 0 if invalid (collinear, hull failure, etc.).
        """
        try:
            if np.any(np.isnan(points)) or np.any(np.isinf(points)):
                return 0.0
            
            # Compute convex hull area
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-12:
                return 0.0
            
            # Compute minimum triangle area
            min_area = float('inf')
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
                        area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                        if area < min_area:
                            min_area = area
                            if min_area < 1e-12:
                                return 0.0
            return min_area / A_hull
        except Exception:
            return 0.0

    def objective(points_flat):
        """
        Objective function for minimization.
        We minimize -score.
        """
        points = points_flat.reshape((n, 2))
        score = compute_score(points)
        return -score

    # Initial configurations
    initializations = []
    
    # 1. Halton sequence (low discrepancy)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    halton_points = []
    for i in range(n):
        x = halton(i+1, 2)
        y = halton(i+1, 3)
        halton_points.append([x, y])
    initializations.append(np.array(halton_points))

    # 2. Grid-based perturbation (3x3 grid + 4 extra)
    # 3x3 grid
    grid = []
    for i in range(3):
        for j in range(3):
            grid.append([i/2.0, j/2.0])
    # Add 4 more points in the gaps
    extra = [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75]
    ]
    # Combine and shuffle slightly to avoid perfect symmetry locks? 
    # Actually, symmetry is good for Heilbronn.
    grid_points = grid + extra
    initializations.append(np.array(grid_points))

    # 3. Hexagonal packing fragment
    # Center + 6 neighbors + 6 more
    hex_points = []
    r = 0.3
    hex_points.append([0.5, 0.5])
    for i in range(6):
        angle = i * np.pi / 3
        hex_points.append([0.5 + r*np.cos(angle), 0.5 + r*np.sin(angle)])
    # Add 6 more around
    r2 = 0.6
    for i in range(6):
        angle = i * np.pi / 3 + np.pi/6
        hex_points.append([0.5 + r2*np.cos(angle), 0.5 + r2*np.sin(angle)])
    initializations.append(np.array(hex_points))

    # 4. Random with fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    random_points = rng.random((n, 2))
    initializations.append(random_points)

    best_points = None
    best_score = -1.0

    for init in initializations:
        if time.time() - start_time > timeout - 30:
            break
            
        # Normalize to unit square to help optimizer
        init = (init - init.min(axis=0)) / (init.max(axis=0) - init.min(axis=0) + 1e-12)
        
        try:
            result = minimize(
                objective, 
                init.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.success or result.fun < 0:
                candidate_points = result.x.reshape((n, 2))
                score = compute_score(candidate_points)
                
                if score > best_score:
                    best_score = score
                    best_points = candidate_points
                    
        except Exception:
            continue

    # If no optimization succeeded, fallback to best initial
    if best_points is None:
        for init in initializations:
            score = compute_score(init)
            if score > best_score:
                best_score = score
                best_points = init

    # Final check
    if best_points is None:
        # Absolute fallback: random
        rng = np.random.default_rng(seed=123)
        best_points = rng.random((n, 2))

    # Ensure finite and shape correct
    if not np.all(np.isfinite(best_points)):
        best_points = np.array([[i/n, (i+1)/n] for i in range(n)])
        
    return best_points

# EVOLVE-BLOCK-END
