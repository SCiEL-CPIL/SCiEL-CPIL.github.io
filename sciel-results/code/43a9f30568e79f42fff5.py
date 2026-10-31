"""Improved program for Heilbronn triangles task."""
import time
import numpy as np
from itertools import combinations

N = 11

# Pre-compute all triangle index combinations
_TRI_INDICES = np.array(list(combinations(range(N), 3)))
_N_TRI = len(_TRI_INDICES)

# Triangle vertices: (0,0), (1,0), (1/2, sqrt(3)/2)
_SQRT3 = np.sqrt(3.0)
_TRI_AREA = _SQRT3 / 4.0  # Area of the unit-base equilateral triangle

# Barycentric coordinate bounds: a+b+c=1, a,b,c >= 0
# We parameterize by (a, b) with c = 1-a-b, so a>=0, b>=0, a+b<=1


def _min_area(points):
    """Compute minimum triangle area for N points. Returns 0 if any point is outside."""
    # points shape (N, 2)
    # Check if all points are inside the triangle using barycentric coordinates
    # For point (x,y): a = (sqrt(3)/2 * x - y/2) ... let's use the standard formula
    # Actually, let's just compute the areas and also check validity
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)


def _is_inside(points):
    """Check if all points are inside or on the boundary of the triangle."""
    # Barycentric coordinates for triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # For point (x, y):
    # a = 1 - x - y/sqrt(3)  ... let me re-derive
    # Using: P = a*A + b*B + c*C where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # x = b + 0.5*c
    # y = sqrt(3)/2 * c
    # So c = 2*y/sqrt(3), b = x - 0.5*c = x - y/sqrt(3), a = 1 - b - c
    x = points[:, 0]
    y = points[:, 1]
    c = 2.0 * y / _SQRT3
    b = x - y / _SQRT3
    a = 1.0 - b - c
    return np.all(a >= -1e-10) and np.all(b >= -1e-10) and np.all(c >= -1e-10)


def _min_area_valid(points):
    """Return min area if all points are inside, else 0."""
    if not _is_inside(points):
        return 0.0
    return _min_area(points)


def _soft_min_area(points, beta=50.0):
    """Smooth surrogate for min area using log-sum-exp."""
    p1 = points[_TRI_INDICES[:, 0]]
    p2 = points[_TRI_INDICES[:, 1]]
    p3 = points[_TRI_INDICES[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Check validity with penalty
    x = points[:, 0]
    y = points[:, 1]
    c = 2.0 * y / _SQRT3
    b = x - y / _SQRT3
    a = 1.0 - b - c
    
    # Penalty for being outside
    penalty = np.sum(np.maximum(0, -a)**2 + np.maximum(0, -b)**2 + np.maximum(0, -c)**2)
    
    # Soft min
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0
    log_sum = np.log(np.sum(np.exp(areas / beta))) - np.log(len(areas))
    soft_min = beta * (np.log(np.sum(np.exp(areas / beta))) - np.log(len(areas)))
    # Actually: softmin = beta * log(sum(exp(areas/beta))) - beta*log(n)
    # Let me just use: soft_min = -beta * log(sum(exp(-areas/beta)))
    soft_min = -beta * np.log(np.sum(np.exp(-areas / beta))) + beta * np.log(len(areas))
    
    return soft_min - penalty * 10.0


def construct():
    """Construct 11 points to maximize the minimum triangle area."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Multi-start simulated annealing with vectorized evaluation
    rng = np.random.default_rng(42)
    
    # Initial points: use a good heuristic starting configuration
    # Place points roughly evenly in the triangle
    def _initial_points():
        """Generate a reasonable initial configuration."""
        pts = []
        # Corners and midpoints for a good start
        # Use a grid-like distribution in barycentric coords
        # 11 points: try a few patterns
        # Pattern: 3 on each edge (not at vertices) + 5 interior
        # Or: use the known good configuration approach
        
        # Simple approach: random but constrained
        for _ in range(50):
            candidates = []
            for _ in range(N):
                a = rng.random()
                b = rng.random()
                if a + b > 1:
                    a, b = 1 - a, 1 - b
                # Point = a*(0,0) + b*(1,0) + (1-a-b)*(0.5, sqrt(3)/2)
                x = b + 0.5 * (1 - a - b)
                y = _SQRT3 / 2 * (1 - a - b)
                # Add small margin from edges
                # Transform back: x = b + 0.5*(1-a-b), y = sqrt(3)/2*(1-a-b)
                candidates.append([x, y])
            pts = np.array(candidates)
            if _is_inside(pts):
                score = _min_area(pts)
                if best_points is None or score > best_score:
                    best_score = score
                    best_points = pts.copy()
                if best_score > 0.01:  # Good enough start
                    break
        return best_points, best_score
    
    best_points, best_score = _initial_points()
    
    if best_points is None:
        # Fallback: simple random
        pts = []
        for _ in range(N):
            a = rng.random()
            b = rng.random()
            if a + b > 1:
                a, b = 1 - a, 1 - b
            x = b + 0.5 * (1 - a - b)
            y = _SQRT3 / 2 * (1 - a - b)
            pts.append([x, y])
        best_points = np.array(pts)
        best_score = _min_area_valid(best_points)
    
    # Simulated annealing
    current_points = best_points.copy()
    current_score = best_score
    
    # SA parameters
    T_start = 0.001
    T_end = 1e-6
    n_iter_per_temp = 5000
    
    # We'll do multiple SA runs with different initial temperatures and restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Reset to best found so far for this restart, or perturb it
        if restart == 0:
            current_points = best_points.copy()
            current_score = best_score
        else:
            # Perturb best points
            current_points = best_points + rng.normal(0, 0.02, (N, 2))
            # Project back inside triangle
            for i in range(N):
                x, y = current_points[i]
                # Barycentric
                c = 2.0 * y / _SQRT3
                b = x - y / _SQRT3
                a = 1.0 - b - c
                # Clip to valid range
                if a < 0: a = 0
                if b < 0: b = 0
                if c < 0: c = 0
                s = a + b + c
                if s > 0:
                    a /= s; b /= s; c /= s
                current_points[i] = [b + 0.5 * c, _SQRT3 / 2 * c]
            current_score = _min_area_valid(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T = T_start * (0.5 ** restart)  # Lower temp for later restarts
        
        for _ in range(n_iter_per_temp):
            if time.time() > deadline:
                break
            
            # Generate a perturbation
            idx = rng.integers(N)
            step_size = T * 100
            delta = rng.normal(0, step_size, 2)
            new_points = current_points.copy()
            new_points[idx] += delta
            
            # Project back into triangle if needed
            x, y = new_points[idx]
            c = 2.0 * y / _SQRT3
            b = x - y / _SQRT3
            a = 1.0 - b - c
            if a < -1e-9 or b < -1e-9 or c < -1e-9:
                # Project to nearest point in triangle
                # Simple: clip barycentric coords
                a_c = max(0, a)
                b_c = max(0, b)
                c_c = max(0, c)
                s = a_c + b_c + c_c
                if s > 0:
                    a_c /= s; b_c /= s; c_c /= s
                new_points[idx] = [b_c + 0.5 * c_c, _SQRT3 / 2 * c_c]
            
            new_score = _min_area_valid(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
            else:
                # Accept with probability exp((new - cur)/T)
                if T > 1e-15:
                    delta_e = (new_score - current_score) / T
                    if delta_e > -50 and rng.random() < np.exp(delta_e):
                        current_points = new_points
                        current_score = new_score
            
            # Cool down
            T *= 0.999
    
    # Final polish: local optimization using Nelder-Mead on the best points
    if time.time() < deadline - 2.0:
        try:
            from scipy.optimize import minimize
            
            def _neg_min_area_flat(x_flat):
                pts = x_flat.reshape(N, 2)
                # Check validity
                x = pts[:, 0]
                y = pts[:, 1]
                c = 2.0 * y / _SQRT3
                b = x - y / _SQRT3
                a = 1.0 - b - c
                if np.any(a < -1e-8) or np.any(b < -1e-8) or np.any(c < -1e-8):
                    return 1.0  # Large penalty
                return -_min_area(pts)
            
            x0 = best_points.flatten()
            res = minimize(_neg_min_area_flat, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res.success or res.fun < _neg_min_area_flat(x0):
                res_pts = res.x.reshape(N, 2)
                if _is_inside(res_pts):
                    res_score = _min_area(res_pts)
                    if res_score > best_score:
                        best_score = res_score
                        best_points = res_pts
        except Exception:
            pass
    
    return best_points
