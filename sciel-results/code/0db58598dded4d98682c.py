import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    angle = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack([x, y])

def polygon_separation(p1, p2):
    """Check if two convex polygons are separated. Returns minimum separation distance (negative if overlapping).
    
    p1, p2: (6, 2) arrays of vertices
    """
    # Get all edge normals
    def get_normals(p):
        normals = []
        for i in range(6):
            v1 = p[i]
            v2 = p[(i+1) % 6]
            edge = v2 - v1
            # Normal (perpendicular, pointing outward)
            n = np.array([edge[1], -edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n = n / n_norm
            # Ensure outward: check against center
            center = p.mean(axis=0)
            mid = (v1 + v2) / 2
            if np.dot(n, mid - center) < 0:
                n = -n
            normals.append(n)
        return np.array(normals)
    
    n1 = get_normals(p1)
    n2 = get_normals(p2)
    all_normals = np.vstack([n1, n2])
    
    min_sep = np.inf
    for n in all_normals:
        proj1 = p1 @ n
        proj2 = p2 @ n
        # Separation in this direction
        sep = max(proj2.min() - proj1.max(), proj1.min() - proj2.max())
        # If sep < 0, they overlap in this direction
        min_sep = min(min_sep, sep)
    
    return min_sep

def overlap_penalty(params, penalty_weight=1000):
    """Compute penalty for overlapping inner hexagons."""
    positions = np.array([[params[3*k], params[3*k+1]] for k in range(N)])
    angles_deg = np.array([params[3*k+2] for k in range(N)])
    
    penalty = 0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles_deg[j])
            sep = polygon_separation(v1, v2)
            if sep < 0:
                penalty += penalty_weight * sep * sep  # Quadratic penalty
    
    return penalty

def min_enclosing_hexagon_side(positions, angles_deg, outer_angle):
    """Find minimum side length of regular hexagon with given outer_angle that contains all inner hexagons."""
    all_verts = []
    for pos, ang in zip(positions, angles_deg):
        verts = hexagon_vertices(pos[0], pos[1], 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    theta = np.radians(outer_angle)
    r_min = 0
    for k in range(3):
        a = np.radians(30 + k * 60) + theta
        n = np.array([np.cos(a), np.sin(a)])
        proj = all_verts @ n
        width = proj.max() - proj.min()
        r_min = max(r_min, width / 2)
    
    s = r_min * 2 / np.sqrt(3)
    
    # Find a valid center
    midpoints = []
    for k in range(3):
        a = np.radians(30 + k * 60) + theta
        n = np.array([np.cos(a), np.sin(a)])
        proj = all_verts @ n
        midpoints.append((proj.max() + proj.min()) / 2)
    
    A = np.array([[np.cos(np.radians(30 + k*60) + theta), np.sin(np.radians(30 + k*60) + theta)] for k in range(3)])
    b = np.array(midpoints)
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    return s, (c[0], c[1])

def objective(params):
    """Objective: minimize enclosing hexagon side length + overlap penalty."""
    positions = np.array([[params[3*k], params[3*k+1]] for k in range(N)])
    angles_deg = np.array([params[3*k+2] for k in range(N)])
    outer_angle = params[3*N]
    
    s, center = min_enclosing_hexagon_side(positions, angles_deg, outer_angle)
    penalty = overlap_penalty(params, penalty_weight=50)
    
    return s + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: hexagonal lattice
    spacing = np.sqrt(3)
    init_params = []
    for row in range(3):
        for col in range(4):
            x = col * spacing + (row % 2) * spacing / 2
            y = row * 1.5
            init_params.extend([x, y, 0.0])
    init_params.append(0.0)
    init_params = np.array(init_params)
    
    # Center
    positions_init = np.array([[init_params[3*k], init_params[3*k+1]] for k in range(N)])
    centroid = positions_init.mean(axis=0)
    for k in range(N):
        init_params[3*k] -= centroid[0]
        init_params[3*k+1] -= centroid[1]
    
    best_params = init_params.copy()
    best_s = objective(best_params)
    
    # Try different outer angles
    for outer_ang in [0, 30, 60, 90, 120, 150]:
        test_params = init_params.copy()
        test_params[3*N] = outer_ang
        s_test = objective(test_params)
        if s_test < best_s:
            best_s = s_test
            best_params = test_params.copy()
    
    # Optimize
    if time.time() < deadline:
        result = minimize(objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        if result.fun < best_s:
            best_s = result.fun
            best_params = result.x
    
    # Random restarts
    np.random.seed(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        
        params = init_params.copy()
        for k in range(N):
            params[3*k] += np.random.randn() * 0.3
            params[3*k+1] += np.random.randn() * 0.3
            params[3*k+2] = np.random.uniform(0, 60)
        params[3*N] = np.random.uniform(0, 60)
        
        s_test = objective(params)
        if s_test < best_s:
            best_s = s_test
            best_params = params.copy()
        
        if time.time() < deadline:
            result = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_s:
                best_s = result.fun
                best_params = result.x
    
    # Extract result - remove penalty contribution
    positions = np.array([[best_params[3*k], best_params[3*k+1]] for k in range(N)])
    angles_deg = np.array([best_params[3*k+2] for k in range(N)])
    outer_angle = best_params[3*N]
    
    s, center = min_enclosing_hexagon_side(positions, angles_deg, outer_angle)
    
    # Verify no overlaps
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles_deg[i])
            v2 = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles_deg[j])
            sep = polygon_separation(v1, v2)
            if sep < -0.01:  # Significant overlap
                # This configuration is invalid; fall back to initial
                break
    
    inner = []
    for k in range(N):
        inner.append([positions[k, 0], positions[k, 1], angles_deg[k]])
    
    return inner, center, s, outer_angle

