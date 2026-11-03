import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 135.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    return [(cx + s * math.cos(a + i * math.pi / 3),
             cy + s * math.sin(a + i * math.pi / 3)) for i in range(6)]


def support_func(center, angle_deg, nx, ny):
    """Support function of unit hexagon at center with rotation angle_deg in direction (nx,ny)."""
    a = math.radians(angle_deg)
    best = 0.0
    for k in range(6):
        theta = a + k * math.pi / 3
        v = center[0] * nx + center[1] * ny
        h = math.cos(theta) * nx + math.sin(theta) * ny
        val = v + h
        if val > best:
            best = val
    return best


def containment_constraint(center, angle_deg, outer_side, outer_angle):
    """Max violation of containment (positive = outside)."""
    a_out = math.radians(outer_angle)
    max_viol = 0.0
    for k in range(6):
        theta = a_out + (k + 0.5) * math.pi / 3
        nx, ny = math.cos(theta), math.sin(theta)
        apothem = outer_side * math.sqrt(3) / 2
        h = support_func(center, angle_deg, nx, ny)
        if h > apothem:
            max_viol = max(max_viol, h - apothem)
    return max_viol


def non_overlap(i, j, centers, angles):
    """Max violation of non-overlap (positive = overlapping)."""
    # Critical directions for two hexagons
    a1, a2 = math.radians(angles[i]), math.radians(angles[j])
    normals = set()
    for k in range(6):
        theta1 = a1 + math.pi / 6 + k * math.pi / 3
        theta2 = a2 + math.pi / 6 + k * math.pi / 3
        normals.add((round(math.cos(theta1), 10), round(math.sin(theta1), 10)))
        normals.add((round(math.cos(theta2), 10), round(math.sin(theta2), 10)))
    
    max_viol = 0.0
    for (nx, ny) in normals:
        h1 = support_func(centers[i], angles[i], nx, ny)
        h2 = support_func(centers[j], angles[j], -nx, -ny)
        dx = centers[j][0] - centers[i][0]
        dy = centers[j][1] - centers[i][1]
        proj = dx * nx + dy * ny
        if proj < h1 + h2:
            max_viol = max(max_viol, h1 + h2 - proj)
    return max_viol


def compute_outer_side(centers, angles, outer_angle=0.0):
    """Compute minimum outer side needed."""
    a_out = math.radians(outer_angle)
    max_proj = 0.0
    for k in range(6):
        theta = a_out + (k + 0.5) * math.pi / 3
        nx, ny = math.cos(theta), math.sin(theta)
        for i in range(N):
            h = support_func(centers[i], angles[i], nx, ny)
            if h > max_proj:
                max_proj = h
    return (2.0 / math.sqrt(3.0)) * max_proj


def validity_penalty(centers, angles, outer_side, outer_angle=0.0):
    """Total penalty for violations."""
    pen = 0.0
    for i in range(N):
        pen += containment_constraint(centers[i], angles[i], outer_side, outer_angle)
    for i in range(N):
        for j in range(i + 1, N):
            pen += non_overlap(i, j, centers, angles)
    return pen


def objective(params, outer_angle=0.0):
    """Objective: outer_side + lambda * penalty."""
    centers = [(params[2 * i], params[2 * i + 1]) for i in range(N)]
    angles = [math.degrees(params[2 * i + 2]) for i in range(N)]
    
    outer_side = compute_outer_side(centers, angles, outer_angle)
    pen = validity_penalty(centers, angles, outer_side, outer_angle)
    
    return outer_side + 100.0 * pen


def pack_honeycomb():
    """Place hexagons in a honeycomb pattern."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        positions.append((2.0 * math.cos(theta), 2.0 * math.sin(theta)))
    for k in range(5):
        theta = math.radians(k * 60.0)
        positions.append((4.0 * math.cos(theta), 4.0 * math.sin(theta)))
    centers = [p for p in positions]
    angles = [0.0] * N
    return centers, angles


def optimize_topology(centers, angles, max_time=60.0):
    """Optimize positions and angles to minimize outer side."""
    t0 = time.time()
    
    # Initial parameters
    params = []
    for i in range(N):
        params.append(centers[i][0])
        params.append(centers[i][1])
        params.append(math.radians(angles[i]))
    
    bounds = [(-5, 5)] * (2 * N) + [(-0.52, 0.52)] * N
    
    # Use differential evolution for global search
    def obj_func(p, oa):
        return objective(p, oa)
    
    try:
        result = differential_evolution(
            lambda p: obj_func(p, 0.0),
            bounds,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False
        )
        best_params = result.x
        best_val = result.fun
    except Exception:
        best_params = params
        best_val = objective(params, 0.0)
    
    # Local refinement with Nelder-Mead
    if time.time() - t0 < max_time - 5:
        try:
            local_res = minimize(
                lambda p: objective(p, 0.0),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if local_res.fun < best_val:
                best_params = local_res.x
                best_val = local_res.fun
        except Exception:
            pass
    
    # Extract results
    centers_out = [(best_params[2 * i], best_params[2 * i + 1]) for i in range(N)]
    angles_out = [math.degrees(best_params[2 * i + 2]) for i in range(N)]
    outer_side = compute_outer_side(centers_out, angles_out, 0.0)
    
    return outer_side, centers_out, angles_out


def construct():
    t0 = time.time()
    
    # Start with honeycomb packing
    centers, angles = pack_honeycomb()
    
    # Optimize
    best_side, best_centers, best_angles = optimize_topology(centers, angles, max_time=TIME_LIMIT - 10)
    
    # Try a few different outer angles
    for oa in [0.0, 15.0, 30.0]:
        if time.time() - t0 > TIME_LIMIT - 5:
            break
        side, c, a = optimize_topology(centers, angles, max_time=TIME_LIMIT - 5)
        # Recompute with this outer angle
        centers2 = c
        angles2 = a
        outer_side2 = compute_outer_side(centers2, angles2, oa)
        if outer_side2 < best_side:
            best_side = outer_side2
            best_centers = centers2
            best_angles = angles2
    
    inner = [(best_centers[i][0], best_centers[i][1], best_angles[i]) for i in range(N)]
    outer_center = (0.0, 0.0)
    
    return inner, outer_center, best_side, 0.0
