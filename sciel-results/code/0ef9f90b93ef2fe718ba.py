import numpy as np
from scipy.optimize import differential_evolution, minimize
import itertools
import time

N = 11
# Triangle vertices
A = np.array([0, 0])
B = np.array([1, 0])
C = np.array([0.5, np.sqrt(3)/2])

def is_inside_triangle(pts):
    # pts: (N, 2)
    # Checks:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * (x - 1)
    x = pts[:, 0]
    y = pts[:, 1]
    return np.all(y >= -1e-9) and np.all(y <= np.sqrt(3) * x + 1e-9) and np.all(y <= -np.sqrt(3) * (x - 1) + 1e-9)

def min_triangle_area(pts):
    # pts: (N, 2)
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(N), 3):
        p1, p2, p3 = pts[i], pts[j], pts[k]
        # Area = 0.5 * |cross(p2-p1, p3-p1)|
        cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area

def penalty(pts):
    # Returns 0 if inside, else large value
    if is_inside_triangle(pts):
        return 0.0
    # Calculate distance to outside
    # Simple penalty: sum of violations
    x = pts[:, 0]
    y = pts[:, 1]
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    v3 = np.maximum(0, y - (-np.sqrt(3) * (x - 1)))
    return 100.0 * (np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2))

def objective(x):
    pts = x.reshape(N, 2)
    area = min_triangle_area(pts)
    pen = penalty(pts)
    # We want to maximize area, so minimize -area + penalty
    return -area + pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Bounds for DE
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Phase 1: Differential Evolution
    best_x = None
    best_val = float('inf')
    
    # Try DE
    try:
        # popsize=20, maxiter=100
        # tol=0.01
        res = differential_evolution(objective, bounds, maxiter=100, popsize=20, seed=42, polish=False, tol=0.01)
        if time.time() > deadline:
            return np.array(res.x).reshape(N, 2)
            
        best_x = res.x
        best_val = res.fun
    except Exception:
        pass

    # Phase 2: Local Polish
    # If we have a candidate, polish it
    if best_x is not None:
        # Nelder-Mead
        try:
            res_nm = minimize(objective, best_x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < best_val:
                best_x = res_nm.x
                best_val = res_nm.fun
        except Exception:
            pass

    # Check time
    if time.time() > deadline:
        return np.array(best_x).reshape(N, 2)

    # Optional: Multiple restarts if time permits?
    # DE usually covers a good range.
    # Let's add a few random restarts of Nelder-Mead from different points if time allows.
    
    # Generate a few random feasible points
    rng = np.random.default_rng(42)
    feasible_starts = []
    for _ in range(5):
        # Sample uniformly in triangle
        # u, v in [0,1], u+v<1 -> point = u*A + v*B + (1-u-v)*C? No, barycentric
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            p = u * A + v * B + (1 - u - v) * C
            # Expand to N points? No, just one point is not enough.
            # We need 11 points.
            # Let's just perturb the best_x found by DE slightly.
            perturbed = best_x + rng.normal(0, 0.05, size=2*N)
            # Ensure bounds
            perturbed = np.clip(perturbed.reshape(N, 2), 0, [1, np.sqrt(3)/2]).reshape(-1)
            feasible_starts.append(perturbed)
    
    for start in feasible_starts:
        if time.time() > deadline:
            break
        try:
            res_nm = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res_nm.fun < best_val:
                best_x = res_nm.x
                best_val = res_nm.fun
        except Exception:
            pass

    return np.array(best_x).reshape(N, 2)
