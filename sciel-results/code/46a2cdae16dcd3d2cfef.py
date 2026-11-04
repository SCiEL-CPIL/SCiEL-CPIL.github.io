import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, angle_deg):
    """Return the 6 vertices of a unit regular hexagon centered at (cx, cy)
    with the given rotation angle in degrees."""
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    # Base vertices (flat-to-flat, i.e., vertex at angle 0)
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    base_x = np.cos(base_angles)
    base_y = np.sin(base_angles)
    # Rotate
    rx = base_x * cos_a - base_y * sin_a
    ry = base_x * sin_a + base_y * cos_a
    return cx + rx, cy + ry

def _container_normals(angle_deg):
    """Return the 6 outward normal directions of a regular hexagon container
    with the given rotation angle in degrees. For a flat-to-flat hexagon
    (vertex at angle 0), the outward normals point at angles 30, 90, 150, 210, 270, 330."""
    base_normals = (30 + np.arange(6) * 60) * np.pi / 180.0
    a = np.radians(angle_deg)
    normals = base_normals + a
    return np.stack([np.cos(normals), np.sin(normals)], axis=1)

def _max_projection(inner_positions, inner_angles, container_angle_deg):
    """Compute the minimum side length s of a regular hexagon container (centered at origin)
    with the given rotation angle such that all inner hexagon vertices are inside.
    
    For a regular hexagon of side s centered at origin with outward normals n_i,
    the half-width in direction n_i is s * cos(30°) = s * sqrt(3)/2.
    So we need: max over all vertices v of (v · n_i) <= s * sqrt(3)/2 for all i.
    Thus s >= max_i (max_v (v · n_i)) / (sqrt(3)/2).
    """
    normals = _container_normals(container_angle_deg)
    max_proj = 0.0
    for i in range(N):
        cx, cy = inner_positions[i]
        ang = inner_angles[i]
        vx, vy = _hex_vertices(cx, cy, ang)
        for k in range(6):
            proj = np.max(vx * normals[k, 0] + vy * normals[k, 1])
            if proj > max_proj:
                max_proj = proj
    return max_proj / (np.sqrt(3) / 2.0)

def _objective(params):
    """Objective function for Nelder-Mead.
    params: [x0, y0, x1, y1, ..., x11, y11, angle0, angle1, ..., angle11, container_angle]
    Total: 2*N + N + 1 = 3*N + 1 = 37 parameters.
    """
    positions = params[:2 * N].reshape(N, 2)
    inner_angles = params[2 * N:3 * N]
    container_angle = params[3 * N]
    
    # Check for overlaps: no two hexagons should overlap.
    # Two unit hexagons with centers at distance d apart and relative rotation theta
    # are non-overlapping if d >= 2 (for same orientation) or more generally if
    # the minimum distance between any vertex of one and any vertex of the other is >= 0.
    # A simpler sufficient condition: d >= 2.0 (works for same orientation).
    # For rotated hexagons, the minimum separation can be less than 2, but let's use
    # a conservative check: d >= 2.0 * 0.95 to allow small rotations.
    # Actually, let's just check pairwise vertex-vertex distances.
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1.95:  # Conservative threshold
                penalty += (1.95 - dist) ** 2
    
    s = _max_projection(positions, inner_angles, container_angle)
    return s + 100.0 * penalty

def construct():
    sqrt3 = np.sqrt(3)
    
    # Initial guess: 3 rows of 4, hexagonal lattice
    init_positions = np.zeros((N, 2))
    init_angles = np.zeros(N)
    idx = 0
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            init_positions[idx, 0] = x
            init_positions[idx, 1] = y
            idx += 1
    
    init_container_angle = 0.0
    
    x0 = np.concatenate([init_positions.flatten(), init_angles, [init_container_angle]])
    
    # Run Nelder-Mead optimization
    result = minimize(_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5,
                               'initial_simplex': 0.1})
    
    best_params = result.x
    best_s = result.fun - 100.0 * 0  # The penalty should be ~0 at optimum
    
    # Try a few different initial container angles
    for trial_angle in [15.0, 30.0, 45.0, 60.0]:
        x0_trial = x0.copy()
        x0_trial[3 * N] = trial_angle
        result_trial = minimize(_objective, x0_trial, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5,
                                         'initial_simplex': 0.1})
        if result_trial.fun < best_s:
            best_s = result_trial.fun
            best_params = result_trial.x
    
    # Extract final parameters
    positions = best_params[:2 * N].reshape(N, 2)
    inner_angles = best_params[2 * N:3 * N]
    container_angle = best_params[3 * N]
    
    # Center the configuration
    positions -= np.mean(positions, axis=0)
    
    # Compute the actual required side length
    s = _max_projection(positions, inner_angles, container_angle)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(inner_angles[i])])
    
    center = [0.0, 0.0]
    side = float(s)
    angle = float(container_angle)
    
    return inner, center, side, angle
