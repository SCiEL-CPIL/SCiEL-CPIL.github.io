"""Improved packing: Nelder-Mead with smooth penalties and multi-start."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon (flat-top when angle=0)."""
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def sat_distance(v1, v2):
    """
    Fast SAT distance check.
    Returns negative if overlapping (penetration depth), positive if separated.
    """
    min_overlap = float('inf')
    for poly in [v1, v2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return float('inf')
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
    return -min_overlap

def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    max_required = 0.0
    theta = outer_angle_rad
    normals = np.column_stack([
        np.cos(theta + np.radians(np.arange(0, 360, 60))),
        np.sin(theta + np.radians(np.arange(0, 360, 60)))
    ])
    for hex_verts in inner_hexes:
        rel = hex_verts - np.array([outer_cx, outer_cy])
        projections = np.dot(rel, normals.T)
        max_projs = np.max(projections, axis=0)
        required_sides = max_projs * 2.0 / np.sqrt(3.0)
        current_max = np.max(required_sides)
        if current_max > max_required:
            max_required = current_max
    return max_required

def objective(params, outer_angle_rad=0.0):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: [x0, y0, a0, ..., x11, y11, a11] (36 values)
    """
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    # Smooth overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = sat_distance(inner_hexes[i], inner_hexes[j])
            if d < 0:
                penalty += 50.0 * (-d) ** 2
            elif d < 0.02:
                penalty += 5.0 * (0.02 - d) ** 2
    
    # Outer side (centered at origin)
    side = compute_outer_side(inner_hexes, 0.0, 0.0, outer_angle_rad)
    
    return side + penalty

def init_honeycomb():
    """Honeycomb-like initial configuration."""
    hx = []
    hy = []
    ha = []
    sqrt3 = np.sqrt(3)
    for r in range(3):
        for c in range(4):
            x = c * sqrt3
            y = r * 1.5
            if r % 2 == 1:
                x += sqrt3 / 2
            hx.append(x)
            hy.append(y)
            ha.append(0.0)
    hx = np.array(hx)
    hy = np.array(hy)
    ha = np.array(ha)
    hx -= np.mean(hx)
    hy -= np.mean(hy)
    return np.concatenate([hx, hy, ha])

def init_random(scale=3.0):
    """Random initial configuration."""
    rng = np.random.RandomState(np.random.randint(0, 10000))
    hx = rng.uniform(-scale, scale, N)
    hy = rng.uniform(-scale, scale, N)
    ha = rng.uniform(0, 2*np.pi, N)
    hx -= np.mean(hx)
    hy -= np.mean(hy)
    return np.concatenate([hx, hy, ha])

def construct():
    start_time = time.time()
    
    best_params = None
    best_score = float('inf')
    
    # Try multiple initializations
    inits = [
        init_honeycomb(),
        init_random(2.5),
        init_random(3.0),
        init_random(3.5),
    ]
    
    for init_idx, init_params in enumerate(inits):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Nelder-Mead optimization
        result = minimize(
            objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    if best_params is None:
        best_params = init_honeycomb()
    
    # Final refinement with smaller step
    if time.time() - start_time < DEADLINE - 5:
        result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    inner_x = best_params[0:12]
    inner_y = best_params[12:24]
    inner_a = best_params[24:36]
    
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    cx = np.mean(inner_x)
    cy = np.mean(inner_y)
    
    # Find best outer angle
    best_side = np.inf
    best_outer_angle = 0.0
    for angle_deg in np.arange(0, 60, 0.5):
        angle_rad = np.radians(angle_deg)
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad
    
    # Fine-tune outer angle
    for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.05):
        angle_rad = np.radians(angle_deg)
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad
    
    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            d = sat_distance(inner_hexes[i], inner_hexes[j])
            if d < -1e-5:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: spread hexagons far apart
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_side
    
    return inner, (cx, cy), outer_side, np.degrees(best_outer_angle)
