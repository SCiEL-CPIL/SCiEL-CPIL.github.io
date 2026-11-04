"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Vertices of the equilateral triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
# Area of this triangle = sqrt(3)/4
A_B = np.array([0.0, 0.0])
A_C = np.array([1.0, 0.0])
A_B2 = np.array([0.5, np.sqrt(3) / 2.0])

# Precompute all triangle index pairs
from itertools import combinations
TRI_IDX = list(combinations(range(N), 3))
TRI_IDX = np.array(TRI_IDX)  # shape (165, 3)

DEADLINE = 110.0  # safety margin


def _min_triangle_area(pts):
    """
    pts: (N, 2) array.
    Returns the minimum area of any triangle formed by 3 of the N points.
    Uses vectorized computation over all C(N,3) = 165 triangles.
    """
    # pts[:, i] for i in TRI_IDX columns
    p1 = pts[TRI_IDX[:, 0]]  # (165, 2)
    p2 = pts[TRI_IDX[:, 1]]  # (165, 2)
    p3 = pts[TRI_IDX[:, 2]]  # (165, 2)
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _in_triangle(pts):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # For triangle (0,0), (1,0), (0.5, h) with h = sqrt(3)/2
    # Edge 1: from (0,0) to (1,0): y >= 0
    # Edge 2: from (1,0) to (0.5, h): line x + (2/sqrt(3))*y <= 1  =>  sqrt(3)*x + 2*y <= sqrt(3)
    # Edge 3: from (0.5, h) to (0,0): line y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
    h = np.sqrt(3) / 2.0
    y = pts[:, 1]
    x = pts[:, 0]
    # Edge 1: y >= 0 (with small tolerance)
    c1 = y >= -1e-9
    # Edge 2: sqrt(3)*x + 2*y <= sqrt(3) + tol
    c2 = np.sqrt(3) * x + 2.0 * y <= np.sqrt(3) + 1e-9
    # Edge 3: y <= sqrt(3)*x + tol
    c3 = y <= np.sqrt(3) * x + 1e-9
    return np.all(c1 & c2 & c3)


def _project_to_triangle(pts):
    """Project points back into the triangle if they're outside (simple clipping)."""
    # For simplicity, just clip: if a point is outside, move it toward the centroid
    # A more robust approach: project onto the nearest edge
    # But for SA, we generate moves that stay inside, so this is rarely needed
    # Fallback: if any point is outside, return the centroid for that point
    h = np.sqrt(3) / 2.0
    x = pts[:, 0]
    y = pts[:, 1]
    c1 = y >= -1e-9
    c2 = np.sqrt(3) * x + 2.0 * y <= np.sqrt(3) + 1e-9
    c3 = y <= np.sqrt(3) * x + 1e-9
    outside = ~(c1 & c2 & c3)
    if np.any(outside):
        centroid = np.array([0.5, h / 3.0])
        pts = pts.copy()
        pts[outside] = centroid
    return pts


def _random_point_in_triangle(rng):
    """Generate a random point uniformly inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    # Barycentric: p = u*A + v*B + (1-u-v)*C
    p = u * A_C + v * A_B2
    return p


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Initial configuration: place points at vertices, edge midpoints, and centroid + a few random
    # Better: use a structured initial guess
    # Vertices (3), edge midpoints (3), centroid (1), and 4 more
    initial = np.array([
        [0.0, 0.0],           # vertex
        [1.0, 0.0],           # vertex
        [0.5, np.sqrt(3)/2],  # vertex
        [0.5, 0.0],           # midpoint of bottom edge
        [0.75, np.sqrt(3)/4], # midpoint of right edge
        [0.25, np.sqrt(3)/4], # midpoint of left edge
        [0.5, np.sqrt(3)/6],  # centroid
    ])
    # Add 4 more points: random but inside
    for _ in range(N - len(initial)):
        p = _random_point_in_triangle(np.random.default_rng(rng.integers(0, 2**31)))
        initial = np.vstack([initial, p])

    # Ensure inside
    initial = _project_to_triangle(initial)
    best_pts = initial.copy()
    best_val = _min_triangle_area(initial)

    # Simulated Annealing
    T_start = 0.001
    T_end = 1e-7
    n_sa_iters = 50000
    # We'll adapt the number of iterations based on time
    # Estimate time per iteration
    t0 = time.time()
    # Do a quick timing
    test_pts = initial.copy()
    for _ in range(100):
        _min_triangle_area(test_pts)
    t1 = time.time()
    time_per_iter = (t1 - t0) / 100.0

    # Determine how many SA iterations we can afford
    time_budget_sa = 60.0  # leave 30s for NM polish and overhead
    max_sa_iters = int(min(n_sa_iters, time_budget_sa / max(time_per_iter, 1e-6)))

    current_pts = initial.copy()
    current_val = best_val

    for it in range(max_sa_iters):
        if time.time() - start_time > DEADLINE - 30.0:
            break

        # Temperature
        alpha = it / max(max_sa_iters - 1, 1)
        T = T_start * (T_end / T_start) ** alpha

        # Generate a random move: pick 1-3 points and perturb
        n_move = 1 + rng.integers(0, 3)  # 1, 2, or 3 points
        idx = rng.choice(N, size=n_move, replace=False)
        new_pts = current_pts.copy()
        step_size = 0.05 * (0.5 + 0.5 * (1 - alpha))  # shrink step over time
        for i in idx:
            new_pts[i] = current_pts[i] + step_size * rng.normal(size=2)

        # Project back into triangle
        new_pts = _project_to_triangle(new_pts)

        new_val = _min_triangle_area(new_pts)

        # Accept if better or with probability exp((new_val - current_val)/T)
        if new_val >= current_val:
            current_pts = new_pts
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_pts = new_pts.copy()
        else:
            delta = new_val - current_val
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_val = new_val

    # Nelder-Mead local polish on best candidate
    if time.time() - start_time < DEADLINE - 10.0:
        # Objective for minimize: we want to maximize min area, so minimize -min area
        def objective(x):
            pts = x.reshape(N, 2)
            if not _in_triangle(pts):
                return 1e6  # penalty
            return -_min_triangle_area(pts)

        # Try NM from best SA result
        try:
            res = minimize(
                objective,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < -best_val:
                cand = res.x.reshape(N, 2)
                if _in_triangle(cand):
                    cv = _min_triangle_area(cand)
                    if cv > best_val:
                        best_val = cv
                        best_pts = cand
        except Exception:
            pass

    # One more round of NM from a few other good candidates if time permits
    if time.time() - start_time < DEADLINE - 5.0:
        # Try a few random restarts of NM from perturbed best
        for restart in range(3):
            if time.time() - start_time > DEADLINE - 5.0:
                break
            try:
                perturbed = best_pts + 0.02 * rng.normal(size=(N, 2))
                perturbed = _project_to_triangle(perturbed)

                def objective2(x):
                    pts = x.reshape(N, 2)
                    if not _in_triangle(pts):
                        return 1e6
                    return -_min_triangle_area(pts)

                res = minimize(
                    objective2,
                    perturbed.ravel(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < -best_val:
                    cand = res.x.reshape(N, 2)
                    if _in_triangle(cand):
                        cv = _min_triangle_area(cand)
                        if cv > best_val:
                            best_val = cv
                            best_pts = cand
            except Exception:
                pass

    # Final safety: ensure all points are strictly inside (with small margin)
    # The problem says "on or inside", so boundary is OK
    # But let's nudge any point very close to boundary inward slightly
    h = np.sqrt(3) / 2.0
    x = best_pts[:, 0]
    y = best_pts[:, 1]
    # Check and nudge
    for i in range(N):
        # Distance to each edge
        # Edge 1 (bottom): y = 0, distance = y[i]
        # Edge 2 (right): sqrt(3)*x + 2*y = sqrt(3), distance = (sqrt(3) - sqrt(3)*x[i] - 2*y[i]) / sqrt(3+4)
        # Edge 3 (left): y = sqrt(3)*x, distance = (sqrt(3)*x[i] - y[i]) / 2
        d1 = y[i]
        d2 = (np.sqrt(3) - np.sqrt(3) * x[i] - 2 * y[i]) / np.sqrt(7)
        d3 = (np.sqrt(3) * x[i] - y[i]) / 2.0
        min_d = min(d1, d2, d3)
        if min_d < 1e-6:
            # Nudge toward centroid
            centroid = np.array([0.5, h / 3.0])
            best_pts[i] = 0.99 * best_pts[i] + 0.01 * centroid

    return best_pts
