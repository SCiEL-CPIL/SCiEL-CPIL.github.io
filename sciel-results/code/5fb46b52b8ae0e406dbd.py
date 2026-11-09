"""Heilbronn triangles: maximize the smallest triangle area among 11 points in a unit-area equilateral triangle."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
SIDE = 1.0
H = np.sqrt(3) / 2
TRI_AREA = 0.5 * SIDE * H  # area of the equilateral triangle with side 1

# Boundaries for points inside the equilateral triangle
# Triangle vertices: (0,0), (1,0), (0.5, H)
# For a point (x, y) to be inside: y >= 0, y <= H, x >= y / H, x <= 1 - y / H
# But we want to stay a bit inside to avoid boundary issues
EPS = 1e-6

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """The objective to maximize: min triangle area divided by triangle area (normalization)."""
    area = _min_triangle_area(points)
    return area / TRI_AREA


def _feasible(points):
    """Check if all points are inside the triangle."""
    for p in points:
        x, y = p
        if y < -EPS or y > H + EPS:
            return False
        if x < y / H - EPS or x > 1 - y / H - EPS:
            return False
    return True


def _penalty(points):
    """Penalty for points outside the triangle."""
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0:
            penalty += (y) ** 2
        if y > H:
            penalty += (y - H) ** 2
        if x < y / H:
            penalty += (x - y / H) ** 2
        if x > 1 - y / H:
            penalty += (x - (1 - y / H)) ** 2
    return penalty


def construct():
    """Place 11 points in the unit equilateral triangle to maximize the smallest triangle area."""
    t0 = time.time()
    DEADLINE = t0 + 140  # safety margin before 150s limit
    
    # Define bounds for each coordinate
    # x ranges from 0 to 1, y ranges from 0 to H
    # But to ensure feasibility, we'll use bounds that are inside
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, H))    # y
    
    def objective(flat_points):
        """Minimize the negative of the min triangle area, with penalty for infeasibility."""
        points = flat_points.reshape(-1, 2)
        score = _min_triangle_area(points)
        pen = _penalty(points)
        return -score + 100.0 * pen
    
    # Initial guess: spread points roughly
    rng = np.random.RandomState(42)
    init = []
    for i in range(N):
        u = rng.uniform(0.05, 0.95)
        v = rng.uniform(0.05, H * 0.9)
        x = u + v / H  # this isn't quite right, let's use barycentric
        # Use barycentric coordinates
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 + 0.5 * r2
        y = H * r2
        init.append([x, y])
    init_flat = np.array(init).flatten()
    
    # Phase 1: Differential Evolution for global optimization
    best_score = -np.inf
    best_points = None
    
    if time.time() < DEADLINE - 60:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=42,
                polish=False,
                callback=None
            )
            de_points = de_result.x.reshape(-1, 2)
            de_score = _score(de_points)
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement using Nelder-Mead or BFGS on the best point found
    if best_points is not None and time.time() < DEADLINE - 20:
        try:
            # Try multiple local optimizations from different starting points
            starts = [best_points.flatten().copy()]
            # Add a few perturbed starts
            for _ in range(3):
                perturbed = best_points + 0.01 * np.random.randn(*best_points.shape)
                starts.append(perturbed.flatten())
            
            for start in starts:
                if time.time() > DEADLINE - 10:
                    break
                try:
                    local_result = minimize(
                        objective,
                        start,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    local_points = local_result.x.reshape(-1, 2)
                    local_score = _score(local_points)
                    if local_score > best_score:
                        best_score = local_score
                        best_points = local_points.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Final: if we have a valid result, return it; otherwise fall back to a heuristic
    if best_points is not None and _feasible(best_points):
        return np.array(best_points)
    
    # Fallback: random points inside the triangle
    rng = np.random.RandomState(0)
    points = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        x = r1 + 0.5 * r2
        y = H * r2
        points.append([x, y])
    return np.array(points)
