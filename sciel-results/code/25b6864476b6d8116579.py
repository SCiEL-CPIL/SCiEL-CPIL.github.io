"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length containing all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check no overlaps and all inside outer hexagon."""
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            if polygons_overlap(v1, v2, tol=1e-8):
                return False
    ox, oy = outer_center
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % 6] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    return True

def objective(params):
    """Objective: outer side length + overlap penalty."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                if polygons_overlap(v1, v2, tol=1e-8):
                    penalty += (2.0 - d) ** 2 * 200
    return S + penalty

def construct():
    """Main entry point: find optimal packing using differential evolution + local refinement."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    # Hexagonal lattice initial positions
    hex_positions = [(0, 0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]

    # Build bounds for differential evolution
    # Inner hexagons: positions in [-8, 8], angles in [0, 2*pi]
    # Outer: center in [-10, 10], angle in [0, 2*pi]
    bounds = []
    for i in range(N):
        bounds.append((-8, 8))   # cx
        bounds.append((-8, 8))   # cy
        bounds.append((0, 2*np.pi))  # angle
    bounds.append((-10, 10))  # outer cx
    bounds.append((-10, 10))  # outer cy
    bounds.append((0, 2*np.pi))  # outer angle

    def de_objective(params):
        return objective(params)

    best_result = None
    best_score = float('inf')

    # Phase 1: Differential Evolution (global search)
    # Use seed for reproducibility
    np.random.seed(123)
    
    # Try DE with moderate population
    try:
        if time.time() < deadline - 40:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=123,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if de_result.fun < best_score:
                params = de_result.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_score = S
                    best_result = (inner, (ox, oy), S, outer_angle)
    except Exception:
        pass

    # Phase 2: Local refinement with L-BFGS-B from best DE result
    if best_result is not None and time.time() < deadline - 20:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_score = S
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass

    # Phase 3: Additional local refinements from perturbed best
    if best_result is not None and time.time() < deadline - 10:
        try:
            params_base = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_base[3*i] = cx
                params_base[3*i+1] = cy
                params_base[3*i+2] = np.radians(ang)
            params_base[3*N] = best_result[1][0]
            params_base[3*N+1] = best_result[1][1]
            params_base[3*N+2] = np.radians(best_result[3])
            
            np.random.seed(777)
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                perturbation = np.random.randn(3*N+3) * 0.05
                params_trial = params_base + perturbation
                res = minimize(objective, params_trial, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_score:
                    params = res.x
                    inner = []
                    for i in range(N):
                        cx = params[3*i]
                        cy = params[3*i+1]
                        ang = np.degrees(params[3*i+2]) % 360
                        inner.append([cx, cy, ang])
                    ox = params[3*N]
                    oy = params[3*N+1]
                    outer_angle = np.degrees(params[3*N+2]) % 360
                    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass

    # Fallback if nothing valid found
    if best_result is None:
        inner = []
        for i, (px, py) in enumerate(hex_positions):
            inner.append([px, py, 0.0])
        ox, oy = 0.0, 0.0
        outer_angle = 0.0
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        S += 0.1
        best_result = (inner, (ox, oy), S, outer_angle)

    return best_result

# EVOLVE-BLOCK-END
