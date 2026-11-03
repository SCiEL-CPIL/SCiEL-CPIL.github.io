import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

# Precompute unit hexagon vertices (side 1, centered at origin)
_HEX_VERTS = np.array([[math.cos(math.pi/6 + k*math.pi/3),
                        math.sin(math.pi/6 + k*math.pi/3)] for k in range(6)]) * 1.0
# Actually vertices of unit regular hexagon: distance from center to vertex = 1
_HEX_VERTS = np.array([[math.cos(k*math.pi/3), math.sin(k*math.pi/3)] for k in range(6)])

# For collision: two unit hexagons don't overlap iff distance between centers >= SQRT3
# (the minimal distance between parallel edges of two touching unit hexagons is SQRT3)
MIN_DIST = SQRT3

def _hexagon_support(pts, angle_deg):
    """For each point (center of a unit hex), compute max projection onto 6 normals
    of the OUTER hexagon orientation. Returns max of |proj| over all points and normals."""
    beta = math.radians(angle_deg)
    normals = np.array([[math.cos(beta + math.pi/6 + k*math.pi/3),
                         math.sin(beta + math.pi/6 + k*math.pi/3)] for k in range(6)])
    # pts shape (n, 2)
    projs = pts @ normals.T  # (n, 6)
    max_abs = np.max(np.abs(projs))
    return max_abs

def _calc_side_length(pts, angle_deg):
    D = _hexagon_support(pts, angle_deg)
    return (D + APOTHEM) / APOTHEM

def _overlap_penalty(pts, weight=50.0):
    """Sum of squared violations of min distance."""
    n = len(pts)
    if n < 2:
        return 0.0
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    violations = np.maximum(0, MIN_DIST - dists[iu])
    return weight * np.sum(violations**2)

def _force_relax(pts_init, angle_deg, n_iters=2000, lr=0.05):
    """Iterative force-directed relaxation to minimize enclosing side length."""
    pts = pts_init.copy()
    n = len(pts)
    best_pts = pts.copy()
    best_S = _calc_side_length(pts, angle_deg)
    
    for it in range(n_iters):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Gradient of enclosing side length w.r.t. each point
        beta = math.radians(angle_deg)
        normals = np.array([[math.cos(beta + math.pi/6 + k*math.pi/3),
                             math.sin(beta + math.pi/6 + k*math.pi/3)] for k in range(6)])
        projs = pts @ normals.T  # (n, 6)
        abs_projs = np.abs(projs)
        max_abs = np.max(abs_projs)
        
        # Gradient: for each point, the normal that gives max |proj|
        # d(max|proj|)/d(pts[i]) = sign(proj) * normal_j where j = argmax|proj| for that point
        # But it's the global max, so only points achieving the max contribute
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(6):
                if abs(abs_projs[i, j] - max_abs) < 1e-12:
                    sign = 1.0 if projs[i, j] > 0 else -1.0
                    grad[i] += sign * normals[j]
        
        # Repulsive force between overlapping hexagons
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2)) + 1e-12
        # Force on i from j: repulsive if dist < MIN_DIST
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < MIN_DIST:
                    # Repulsive force proportional to overlap
                    f = (MIN_DIST - d) / d
                    grad[i] += f * diffs[i, j]
        
        # Update
        pts = pts - lr * grad
        lr *= 0.9995  # slow decay
        
        S = _calc_side_length(pts, angle_deg)
        if S < best_S:
            best_S = S
            best_pts = pts.copy()
    
    return best_pts, best_S

def _try_angles(pts, angles):
    """Try multiple outer hexagon angles, return best."""
    best_S = float('inf')
    best_pts = pts
    best_angle = angles[0]
    for ang in angles:
        p, S = _force_relax(pts, ang, n_iters=1500)
        if S < best_S:
            best_S = S
            best_pts = p
            best_angle = ang
    return best_pts, best_S, best_angle

def _random_init(n, spread=3.0):
    """Random cluster initialization."""
    pts = np.random.randn(n, 2) * spread
    return pts

def _grid_init(n):
    """Hexagonal grid initialization."""
    pts = []
    # Place in hex grid pattern
    row = 0
    while len(pts) < n:
        cols_needed = n - len(pts)
        for c in range(cols_needed):
            x = c * SQRT3
            y = row * 1.5
            if row % 2 == 1:
                x += SQRT3 / 2
            pts.append([x, y])
        row += 1
    return np.array(pts[:n])

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_angle = 0.0
    
    # Try a set of promising outer angles
    angle_candidates = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 105.0, 120.0, 135.0, 150.0, 165.0]
    
    # Multiple random restarts with force relaxation
    n_restarts = 8
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Different initialization strategies
        if restart == 0:
            pts_init = _grid_init(N)
        elif restart == 1:
            # Tight hex cluster: 1 center + 6 around + 4 more
            pts_init = np.array([
                [0, 0],
                [SQRT3, 0], [0, 1.5], [-SQRT3/2, 1.5],
                [-SQRT3, 0], [0, -1.5], [SQRT3/2, -1.5],
                [2*SQRT3, 0], [0, 3], [-SQRT3, 3], [SQRT3, -3]
            ])
        elif restart == 2:
            # Linear-ish arrangement
            pts_init = np.array([[i * SQRT3, 0] for i in range(N)])
        elif restart == 3:
            # Two rows
            pts_init = np.array([
                [i * SQRT3, 0] for i in range(6)
            ] + [
                [i * SQRT3 + SQRT3/2, 1.5] for i in range(5)
            ])
        else:
            pts_init = rng.randn(N, 2) * (2.0 + restart * 0.5)
        
        # Center the init
        pts_init -= np.mean(pts_init, axis=0)
        
        # Try a subset of angles for this restart
        angles_to_try = angle_candidates[::max(1, len(angle_candidates)//4)]
        if restart < 2:
            angles_to_try = angle_candidates  # more angles for first restarts
        
        pts_opt, S_opt, ang_opt = _try_angles(pts_init, angles_to_try)
        
        if S_opt < best_global_S:
            best_global_S = S_opt
            best_global_pts = pts_opt
            best_global_angle = ang_opt
    
    # If we found something, do a final polish with more iterations
    if best_global_pts is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts_polish, S_polish = _force_relax(best_global_pts, best_global_angle, n_iters=3000, lr=0.02)
        if S_polish < best_global_S:
            best_global_S = S_polish
            best_global_pts = pts_polish
    
    # Fallback
    if best_global_pts is None:
        best_global_pts = _grid_init(N)
        best_global_angle = 0.0
        best_global_S = _calc_side_length(best_global_pts, best_global_angle)
    
    # Compute final side length
    cx = float(np.mean(best_global_pts[:, 0]))
    cy = float(np.mean(best_global_pts[:, 1]))
    
    # Recompute side length relative to center
    pts_centered = best_global_pts - np.array([cx, cy])
    D = _hexagon_support(pts_centered, best_global_angle)
    S_final = (D + APOTHEM) / APOTHEM
    
    # Build output
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(best_global_angle)])
    
    center = [cx, cy]
    side = float(S_final)
    rotation = float(best_global_angle)
    
    return inner, center, side, rotation
