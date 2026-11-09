"""Heilbronn triangles: 11 points in unit-area equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of unit base equilateral triangle

# The triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_fast(points):
    """Vectorized-ish min triangle area computation."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        # Vectors from point i to all others
        diff = points - points[i]  # (n, 2)
        # Cross products with all pairs j,k > i
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs(diff[j, 0] * diff[k, 1] - diff[k, 0] * diff[j, 1])
                if area < min_area:
                    min_area = area
    return min_area


def _project_to_triangle(x):
    """Project a point onto the equilateral triangle with vertices (0,0), (1,0), (0.5, H)."""
    # Barycentric coordinates
    # P = a*V0 + b*V1 + c*V2, a+b+c=1
    # x = b*1 + c*0.5, y = c*H
    # c = y/H, b = x - 0.5*c
    c = x[1] / H
    b = x[0] - 0.5 * c
    a = 1.0 - b - c
    if a < 0:
        # Project: set a=0, renormalize b,c
        s = b + c
        if s <= 0:
            return V0.copy()
        b /= s
        c /= s
        return b * V1 + c * V2
    elif b < 0:
        s = a + c
        if s <= 0:
            return V0.copy()
        a /= s
        c /= s
        return a * V0 + c * V2
    elif c < 0:
        s = a + b
        if s <= 0:
            return V0.copy()
        a /= s
        b /= s
        return a * V0 + b * V1
    else:
        return x


def _inside_triangle(x):
    """Check if point is inside the triangle (with small margin)."""
    c = x[1] / H
    b = x[0] - 0.5 * c
    a = 1.0 - b - c
    margin = 1e-9
    return a >= -margin and b >= -margin and c >= -margin


def _min_pairwise_distance(points):
    """Minimum pairwise distance among points."""
    n = len(points)
    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_d:
                min_d = d
    return min_d


def _neg_min_area(x):
    """Negative of min triangle area for minimization."""
    pts = np.array([_project_to_triangle(x[i*2:i*2+2]) for i in range(N)])
    return -_min_triangle_area_fast(pts)


def _neg_min_dist(x):
    """Negative of min pairwise distance for minimization."""
    pts = np.array([_project_to_triangle(x[i*2:i*2+2]) for i in range(N)])
    return -_min_pairwise_distance(pts)


def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    # Initial guess: use a structured grid-like placement
    # Place points in a hexagonal-like pattern within the triangle
    def _initial_guess():
        """Create a reasonable initial configuration."""
        pts = []
        # Corner-ish points (slightly inside)
        eps = 0.02
        pts.append([eps, eps])
        pts.append([1 - eps, eps])
        pts.append([0.5, H - eps])
        # Edge midpoints (slightly inside)
        pts.append([0.5, eps])
        pts.append([0.5 + eps, H/2 + eps])
        pts.append([0.5 - eps, H/2 + eps])
        # Interior points
        pts.append([0.5, H/3])
        pts.append([0.3, H/4])
        pts.append([0.7, H/4])
        pts.append([0.5, 2*H/3])
        pts.append([0.35, H/2])
        return pts

    pts = _initial_guess()
    best_pts = np.array(pts)
    best_val = _min_triangle_area_fast(best_pts)

    # Stage 1: Optimize min pairwise distance (smoother objective)
    if time.time() < deadline - 30:
        x0 = np.array([coord for pt in pts for coord in pt])
        bounds = [(0, 1), (0, H)] * N

        def neg_min_dist_obj(x):
            pts_arr = np.array([_project_to_triangle(x[i*2:i*2+2]) for i in range(N)])
            return -_min_pairwise_distance(pts_arr)

        try:
            res_de = differential_evolution(
                neg_min_dist_obj, bounds, seed=42, maxiter=200, popsize=20,
                tol=1e-6, mutation=(0.3, 1.2), recombination=0.8,
                polish=False, init='latinhypercube'
            )
            if time.time() < deadline - 10:
                pts_de = np.array([_project_to_triangle(res_de.x[i*2:i*2+2]) for i in range(N)])
                val = _min_triangle_area_fast(pts_de)
                if val > best_val:
                    best_val = val
                    best_pts = pts_de
        except Exception:
            pass

    # Stage 2: Polish with direct min-area optimization using L-BFGS
    remaining = deadline - time.time()
    if remaining > 5:
        # Multiple restarts from different initial points
        restarts = []
        # Use best found so far
        restarts.append(best_pts.copy())
        # Add some random perturbations
        rng = np.random.RandomState(123)
        for _ in range(5):
            perturbed = best_pts + rng.randn(N, 2) * 0.1
            restarts.append(np.array([_project_to_triangle(p) for p in perturbed]))
        # Add random configurations
        for _ in range(3):
            u = rng.rand(N, 2)
            v = rng.rand(N, 2)
            mask = u + v < 1
            while mask.sum() < N:
                u = rng.rand(N, 2)
                v = rng.rand(N, 2)
                mask = u + v < 1
            pts_rand = np.array([u[i, 0] + 0.5 * v[i, 0], H * v[i, 1]] for i in range(N) if mask[i])
            if len(pts_rand) == N:
                restarts.append(pts_rand)

        for rpts in restarts:
            if time.time() >= deadline - 3:
                break
            x0 = np.array([coord for pt in rpts for coord in pt])

            def obj(x):
                pts_arr = np.array([_project_to_triangle(x[i*2:i*2+2]) for i in range(N)])
                return -_min_triangle_area_fast(pts_arr)

            try:
                res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                if time.time() >= deadline - 2:
                    break
                pts_opt = np.array([_project_to_triangle(res.x[i*2:i*2+2]) for i in range(N)])
                val = _min_triangle_area_fast(pts_opt)
                if val > best_val:
                    best_val = val
                    best_pts = pts_opt
            except Exception:
                pass

    # Final check: ensure all points are inside the triangle
    best_pts = np.array([_project_to_triangle(p) for p in best_pts])

    return best_pts
