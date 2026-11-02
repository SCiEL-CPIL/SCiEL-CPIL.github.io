import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
H = np.sqrt(3)/2


def point_in_triangle(p):
    """Check if point p is inside the equilateral triangle using barycentric coordinates."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > H + 1e-9:
        return False
    # Check below the two slanted edges
    # Edge V0-V2: y <= (2*sqrt(3)/2)*x => y <= sqrt(3)*x
    # Edge V1-V2: y <= -sqrt(3)*(x-1) => y <= sqrt(3)*(1-x)
    if y > np.sqrt(3)*x + 1e-9:
        return False
    if y > np.sqrt(3)*(1-x) + 1e-9:
        return False
    return True


def clip_to_triangle(p):
    """Clip a point to be inside the triangle (approximately)."""
    x, y = p
    if y < 0:
        y = 0
    if x < 0:
        x = 0
    if x > 1:
        x = 1
    if y > H:
        y = H
    # Clip to slanted edges
    y = min(y, np.sqrt(3)*x)
    y = min(y, np.sqrt(3)*(1-x))
    return np.array([x, max(y, 0)])


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a, b, c = points[i], points[j], points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective(params):
    """Objective: negative of minimum triangle area (to be minimized)."""
    points = params.reshape(N, 2)
    # Soft penalty if points go outside triangle
    penalty = 0.0
    for p in points:
        x, y = p
        if y < 0 or x < 0 or x > 1:
            penalty += abs(min(y, 0)) + abs(min(x, 0)) + abs(max(x-1, 0))
        # Check slanted edges
        if y > np.sqrt(3)*x:
            penalty += y - np.sqrt(3)*x
        if y > np.sqrt(3)*(1-x):
            penalty += y - np.sqrt(3)*(1-x)
    if penalty > 1e-6:
        return penalty
    return -min_triangle_area(points)


def make_lattice_init(rows):
    """Create a grid-like initialization inside the triangle."""
    points = []
    # Place points in a structured pattern
    # For 11 points, use a 3-4-4 or similar arrangement
    n_rows = rows
    for r in range(n_rows):
        t = r / max(n_rows - 1, 1)  # 0 to 1
        y = t * H
        # x range at this height
        x_min = y / np.sqrt(3)
        x_max = 1 - y / np.sqrt(3)
        if x_max < x_min:
            continue
        # Determine how many points fit in this row
        # Use spacing based on row
        n_pts = 4 if r < n_rows - 1 else 3
        for c in range(n_pts):
            s = c / max(n_pts - 1, 1)
            x = x_min + s * (x_max - x_min)
            points.append([x, y])
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin

    best_params = None
    best_score = -float('inf')

    # Strategy 1: Lattice-based initialization + local optimization
    inits = []
    # Symmetric lattice
    for rows in [3, 4, 5]:
        pts = make_lattice_init(rows)
        if len(pts) >= N:
            # Take first N or sample
            if len(pts) == N:
                inits.append(pts)
            else:
                # Subsample
                idx = np.linspace(0, len(pts)-1, N, dtype=int)
                inits.append([pts[i] for i in idx])

    # Also add a regular grid
    for grid_n in [3, 4]:
        pts = []
        for i in range(grid_n):
            for j in range(grid_n):
                x = (i + 0.5) / grid_n
                y = (j + 0.5) / grid_n * H
                if point_in_triangle([x, y]):
                    pts.append([x, y])
        if len(pts) >= N:
            idx = np.linspace(0, len(pts)-1, N, dtype=int)
            inits.append([pts[i] for i in idx])

    # Run Nelder-Mead from each initialization
    for init_pts in inits:
        if time.time() > deadline:
            break
        params0 = np.array(init_pts).flatten()
        try:
            res = minimize(objective, params0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 0:  # negative means valid points
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
        except:
            pass

    # If no valid result, try random with many restarts
    if best_params is None:
        rng = np.random.RandomState(42)
        for trial in range(50):
            if time.time() > deadline:
                break
            # Generate random points inside triangle
            pts = []
            while len(pts) < N:
                u, v = rng.random(2)
                if u + v < 1:
                    p = [u + v*0.5, v*H]
                    pts.append(p)
            params0 = np.array(pts).flatten()
            try:
                res = minimize(objective, params0, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 0:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
            except:
                pass

    # If still no valid result, return a default placement
    if best_params is None:
        # Simple grid placement
        pts = []
        for i in range(3):
            for j in range(4):
                x = (i + 0.5) / 3
                y = (j + 0.5) / 4 * H
                if point_in_triangle([x, y]):
                    pts.append([x, y])
        if len(pts) < N:
            # Fill remaining with centroid-ish points
            while len(pts) < N:
                pts.append([0.5, H/3])
        return np.array(pts[:N])

    return best_params.reshape(N, 2)
