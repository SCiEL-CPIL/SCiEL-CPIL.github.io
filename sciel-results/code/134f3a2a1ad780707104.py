# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute all triangle areas for a set of points. Returns array of areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                areas.append(area)
    return np.array(areas)


def _compute_ratio(points):
    """Compute A_min / A_hull ratio."""
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 1e-12:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        return a_min / a_hull
    except Exception:
        return 0.0


def _objective(params, n=13):
    """Negative ratio for minimization. params are (r_0, r_1, theta_1, ..., r_m, theta_m)."""
    # Parse params: first element is base radius, then pairs (r_k, theta_k) for harmonics
    r0 = params[0]
    m = (len(params) - 1) // 2
    
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n) * r0
    for k in range(1, m + 1):
        rk = params[2 * k - 1]
        thk = params[2 * k]
        radii += rk * np.cos(k * angles + thk)
    
    # Ensure positive radii
    radii = np.maximum(radii, 0.01)
    
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    points = np.column_stack([x, y])
    
    ratio = _compute_ratio(points)
    return -ratio


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    
    Uses a regular 13-gon as starting point, then optimizes radial perturbations
    using low-order Fourier modes.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    n = 13
    best_ratio = 0.0
    best_points = None
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_ratio = _compute_ratio(regular_points)
    best_points = regular_points.copy()
    
    # Try optimization with different numbers of Fourier modes
    for m in range(1, 4):
        if time.time() > deadline - 60:
            break
        
        # Parameters: r0, (r1, theta1), (r2, theta2), ..., (rm, thetam)
        n_params = 1 + 2 * m
        
        # Try multiple initial conditions
        for trial in range(3):
            if time.time() > deadline - 60:
                break
            
            # Initial parameters: base radius 1.0, small random perturbations
            rng = np.random.default_rng(seed=trial + m * 100)
            x0 = np.ones(n_params)
            x0[0] = 1.0  # base radius
            for k in range(1, m + 1):
                x0[2 * k - 1] = 0.05 * rng.random()  # small amplitude
                x0[2 * k] = rng.random() * 2 * np.pi  # random phase
            
            # Optimize
            try:
                result = minimize(_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 500, 'fatol': 1e-8, 'xatol': 1e-8})
                
                if result.fun < -best_ratio:
                    # Reconstruct points
                    r0 = result.x[0]
                    radii = np.ones(n) * r0
                    for k in range(1, m + 1):
                        rk = result.x[2 * k - 1]
                        thk = result.x[2 * k]
                        radii += rk * np.cos(k * angles + thk)
                    radii = np.maximum(radii, 0.01)
                    
                    x = radii * np.cos(angles)
                    y = radii * np.sin(angles)
                    points = np.column_stack([x, y])
                    
                    ratio = _compute_ratio(points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
            except Exception:
                pass
    
    # If optimization didn't improve, return regular 13-gon
    if best_points is None:
        best_points = regular_points
    
    return best_points


# EVOLVE-BLOCK-END
