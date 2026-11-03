"""Heilbronn triangles: 11 points, maximize min triangle area (normalized)."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3.0) / 2.0  # height of unit-base equilateral triangle

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _area(p, q, r):
    return 0.5 * abs(np.cross(q - p, r - p))

def _min_triangle_area(points):
    n = len(points)
    best = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _area(points[i], points[j], points[k])
                if a < best:
                    best = a
    return best

def _inside_triangle(p, eps=1e-9):
    # barycentric check for triangle (0,0),(1,0),(0.5,H)
    # p = b1*V1 + b2*V2, b1,b2>=0, b1+b2<=1
    # solve: x = b1 + 0.5*b2 ; y = H*b2  => b2 = y/H ; b1 = x - 0.5*b2
    b2 = p[1] / H
    b1 = p[0] - 0.5 * b2
    return (b1 >= -eps) and (b2 >= -eps) and (b1 + b2 <= 1.0 + eps)

def _clip_to_triangle(p):
    # project/clip a point to be inside the triangle
    b2 = p[1] / H
    b1 = p[0] - 0.5 * b2
    if b1 < 0:
        b1 = 0.0
    if b2 < 0:
        b2 = 0.0
    if b1 + b2 > 1.0:
        s = b1 + b2
        b1 /= s
        b2 /= s
    x = b1 + 0.5 * b2
    y = H * b2
    return np.array([x, y])

def _unflatten(flat):
    pts = np.empty((N, 2))
    for i in range(N):
        pts[i] = _clip_to_triangle(flat[2*i:2*i+2])
    return pts

def _neg_min_area(flat):
    pts = _unflatten(flat)
    return -_min_triangle_area(pts)

# Structured initial guess: 3 rows mimicking a lattice
def _initial_guess():
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 3 points
    # Row 2 (top): 2 points
    # plus 2 extra to reach 11 (distribute)
    # Let's do a clean hexagonal-ish packing
    pts = []
    # Bottom row y ~ 0.08
    y0 = 0.10
    for i in range(4):
        x = 0.12 + i * 0.24
        pts.append([x, y0])
    # Middle row y ~ H/2
    y1 = H / 2.0
    for i in range(3):
        x = 0.24 + i * 0.24
        pts.append([x, y1])
    # Top row y ~ 0.8*H
    y2 = 0.75 * H
    for i in range(2):
        x = 0.36 + i * 0.24
        pts.append([x, y2])
    # Two more: near centroid and one more
    pts.append([0.5, H * 0.55])
    pts.append([0.5, H * 0.30])
    pts = np.array(pts)
    # clip
    for i in range(len(pts)):
        pts[i] = _clip_to_triangle(pts[i])
    return pts

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def _elapsed():
        return time.time() - t0

    # Start with a structured guess
    best_pts = _initial_guess()
    best_val = _min_triangle_area(best_pts)

    # Bounds for DE: x in [0,1], y in [0,H]
    bounds = [(0.0, 1.0), (0.0, H)] * N

    def _time_check():
        return _elapsed() > DEADLINE

    # Differential evolution
    try:
        de_result = differential_evolution(
            _neg_min_area,
            bounds=bounds,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            mutation=(0.4, 1.0),
            recombination=0.7,
            init='latinhypercube',
            polish=False,
            callback=lambda x, convergence: (_ for _ in ()).throw(
                TimeoutError() if _time_check() else None
            ),
        )
        if de_result.success or (de_result.fun < -best_val):
            cand = _unflatten(de_result.x)
            v = _min_triangle_area(cand)
            if v > best_val:
                best_val = v
                best_pts = cand
    except TimeoutError:
        pass
    except Exception:
        pass

    # Nelder-Mead refinement on best
    if not _time_check():
        try:
            start_flat = best_pts.flatten()
            nm_result = minimize(
                _neg_min_area,
                start_flat,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True},
            )
            cand = _unflatten(nm_result.x)
            v = _min_triangle_area(cand)
            if v > best_val:
                best_val = v
                best_pts = cand
        except Exception:
            pass

    # Final safety: ensure all points are strictly inside
    final = np.empty((N, 2))
    for i in range(N):
        final[i] = _clip_to_triangle(best_pts[i])
    # If clipping changed things, recompute
    v = _min_triangle_area(final)
    if v <= 0:
        # fallback to initial guess
        final = _initial_guess()
    return final
