# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _elapsed():
    return time.time() - _start_time

def _point_in_base(u, v):
    """Check if point (u,v) is in the base L-shaped corridor."""
    # Part 1: 0 <= u <= 1, -3 <= v <= 1
    cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    # Part 2: -3 <= u <= 1, 0 <= v <= 1
    cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return cond1 | cond2

def _generate_placements(path):
    """Generate all placements including start, end, and interpolated steps."""
    # Start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    all_points = [start]
    for i in range(len(path)):
        all_points.append(path[i])
    all_points.append(end)
    
    # Interpolate between consecutive points
    placements = [all_points[0]]
    for i in range(len(all_points) - 1):
        p1 = all_points[i]
        p2 = all_points[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for j in range(1, m + 1):
            t = j / m
            interp = p1 + t * (p2 - p1)
            placements.append(interp)
    
    return np.array(placements)

def _evaluate(path):
    """Evaluate the objective: count of sofa points in the intersection."""
    placements = _generate_placements(path)
    n_placements = len(placements)
    
    # Sofa grid
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    X = X.ravel()
    Y = Y.ravel()
    
    # Check each placement
    inside_all = np.ones(len(X), dtype=bool)
    
    for (px, py, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # Transform: R(-a)(w - (x,y))
        dx = X - px
        dy = Y - py
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        inside = _point_in_base(u, v)
        inside_all &= inside
        
        # Early exit if no points remain
        if not inside_all.any():
            return 0.0
    
    return float(inside_all.sum())

_start_time = time.time()
DEADLINE = 580.0  # Safety margin before 630s limit

def moving_sofa_2d() -> np.ndarray:
    """Optimize the path to maximize the sofa area."""
    global _start_time
    _start_time = time.time()
    
    # Initial path (from baseline)
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_score = _evaluate(best_path)
    
    # Try differential evolution with a smaller number of points first
    n_points = 6
    
    def objective(x):
        if _elapsed() > DEADLINE:
            return 0.0
        path = x.reshape(n_points, 3)
        # Ensure angles are in [0, 90]
        path[:, 2] = np.clip(path[:, 2], 0, 90)
        # Ensure angles are roughly monotonic (sort by angle)
        idx = np.argsort(path[:, 2])
        path = path[idx]
        return -_evaluate(path)
    
    bounds = [(-2, 2)] * n_points * 2 + [(0, 90)] * n_points
    
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False,
            callback=lambda x, convergence: None
        )
        
        if _elapsed() < DEADLINE and result.fun < -best_score:
            best_path = result.x.reshape(n_points, 3)
            best_path[:, 2] = np.clip(best_path[:, 2], 0, 90)
            idx = np.argsort(best_path[:, 2])
            best_path = best_path[idx]
            best_score = -result.fun
    except Exception:
        pass
    
    # Try local refinement with Nelder-Mead
    if _elapsed() < DEADLINE - 60:
        from scipy.optimize import minimize
        
        def objective2(x):
            if _elapsed() > DEADLINE:
                return 0.0
            path = x.reshape(n_points, 3)
            path[:, 2] = np.clip(path[:, 2], 0, 90)
            idx = np.argsort(path[:, 2])
            path = path[idx]
            return -_evaluate(path)
        
        try:
            result2 = minimize(
                objective2, best_path.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if result2.fun < -best_score:
                best_path = result2.x.reshape(n_points, 3)
                best_path[:, 2] = np.clip(best_path[:, 2], 0, 90)
                idx = np.argsort(best_path[:, 2])
                best_path = best_path[idx]
                best_score = -result2.fun
        except Exception:
            pass
    
    return best_path

# EVOLVE-BLOCK-END
