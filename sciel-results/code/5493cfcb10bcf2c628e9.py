"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts_array(cx, cy, side=1.0, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon as (6,2) array."""
    a = math.radians(angle_deg)
    angles = a + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(positions, angles, side=1.0):
    """Return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = math.radians(angles[i])
        base_angles = a + math.pi / 6 + np.arange(6) * math.pi / 3
        verts[6*i:6*(i+1), 0] = positions[i][0] + side * np.cos(base_angles)
        verts[6*i:6*(i+1), 1] = positions[i][1] + side * np.sin(base_angles)
    return verts

def _min_enclosing_hex_side(positions, angles, outer_center=(0.0, 0.0), outer_rotation_deg=0.0):
    """Compute minimum side length of outer hexagon to contain all inner hexagons."""
    verts = _all_vertices(positions, angles) - np.array(outer_center)
    
    phi = math.radians(outer_rotation_deg)
    thetas = phi + np.array([(2*k+1)*math.pi/6 for k in range(6)])
    
    max_proj = 0.0
    for theta in thetas:
        proj = verts[:, 0] * math.cos(theta) + verts[:, 1] * math.sin(theta)
        mp = proj.max()
        if mp > max_proj:
            max_proj = mp
    
    return max_proj / (math.sqrt(3) / 2)

def _sat_gap(p1, a1, p2, a2, side=1.0):
    """Compute minimum gap between two hexagons using SAT."""
    v1 = _hex_verts_array(p1[0], p1[1], side, a1)
    v2 = _hex_verts_array(p2[0], p2[1], side, a2)
    
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    min_gap = float('inf')
    for e in list(edges1) + list(edges2):
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_all_disjoint(positions, angles, tol=1e-10):
    """Check all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_gap(positions[i], angles[i], positions[j], angles[j]) < -tol:
                return False
    return True

def _support_function_obj(params, outer_rot_rad):
    """
    Differentiable objective: minimize max projection onto 6 outer normals.
    params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10] (3N values)
    Returns (value, gradient) where value = max_proj / (sqrt(3)/2)
    """
    n = N
    positions = [(params[3*i], params[3*i+1]) for i in range(n)]
    angles = [math.degrees(params[3*i+2]) for i in range(n)]
    
    # Compute all vertices
    verts = _all_vertices(positions, angles)
    
    phi = outer_rot_rad
    thetas = phi + np.array([(2*k+1)*math.pi/6 for k in range(6)])
    
    # Find the maximum projection and which direction/vertex achieves it
    max_proj = 0.0
    max_theta_idx = 0
    max_vert_idx = 0
    for k, theta in enumerate(thetas):
        proj = verts[:, 0] * math.cos(theta) + verts[:, 1] * math.sin(theta)
        idx = proj.argmax()
        if proj[idx] > max_proj:
            max_proj = proj[idx]
            max_theta_idx = k
            max_vert_idx = idx
    
    value = max_proj / (math.sqrt(3) / 2)
    
    # Gradient: only the vertex that achieves max contributes
    grad = np.zeros(3 * n)
    hex_idx = max_vert_idx // 6
    vert_idx = max_vert_idx % 6
    
    theta = thetas[max_theta_idx]
    ct = math.cos(theta)
    st = math.sin(theta)
    
    # Vertex position: center + side * (cos(a + pi/6 + vert_idx*pi/3), sin(a + pi/6 + vert_idx*pi/3))
    a_rad = params[3*hex_idx + 2]  # angle in radians
    base_angle = a_rad + math.pi / 6 + vert_idx * math.pi / 3
    
    # d(vertex_x)/d(cx) = 1, d(vertex_y)/d(cy) = 1
    # d(vertex_x)/d(a) = -side * sin(base_angle), d(vertex_y)/d(a) = side * cos(base_angle)
    side = 1.0
    
    grad[3*hex_idx] += ct        # d(proj)/d(cx)
    grad[3*hex_idx + 1] += st    # d(proj)/d(cy)
    grad[3*hex_idx + 2] += side * (-st * math.sin(base_angle) + ct * math.cos(base_angle))  # d(proj)/d(a)
    
    # Scale by 1/(sqrt(3)/2)
    scale = 1.0 / (math.sqrt(3) / 2)
    grad *= scale
    
    return value, grad

def _penalized_obj(params, outer_rot_rad, penalty_strength=1000.0):
    """
    Penalized objective: support function + penalty for overlaps.
    This makes the objective differentiable everywhere.
    """
    n = N
    positions = [(params[3*i], params[3*i+1]) for i in range(n)]
    angles_rad = params[3:3*n:3]  # angles in radians
    angles_deg = np.degrees(angles_rad)
    
    # Support function value
    value, grad = _support_function_obj(params, outer_rot_rad)
    
    # Overlap penalty: for each pair, compute SAT gap and penalize if negative
    penalty = 0.0
    pgrad = np.zeros(3 * n)
    
    for i in range(n):
        for j in range(i + 1, n):
            gap = _sat_gap(positions[i], angles_deg[i], positions[j], angles_deg[j])
            if gap < 0:
                # Quadratic penalty on penetration
                penalty += penalty_strength * gap * gap
                # Approximate gradient: push apart along the direction of minimum gap
                # For simplicity, use the direction between centers
                diff = np.array([positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]])
                d = np.linalg.norm(diff)
                if d > 1e-8:
                    dir_vec = diff / d
                    pgrad[3*i] += 2 * penalty_strength * gap * dir_vec[0]
                    pgrad[3*i + 1] += 2 * penalty_strength * gap * dir_vec[1]
                    pgrad[3*j] -= 2 * penalty_strength * gap * dir_vec[0]
                    pgrad[3*j + 1] -= 2 * penalty_strength * gap * dir_vec[1]
    
    return value + penalty, grad + pgrad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    def axial_to_xy(q, r):
        return (1.5 * q, sqrt3 * (r + q / 2.0))
    
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)]
    
    best_side = float('inf')
    best_positions = None
    best_angles = [0.0] * N
    best_outer_rotation = 0.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config: center + ring1 + various subsets of ring2
    # Opposite pairs
    for i in range(6):
        for j in range(i+1, 6):
            subset = [ring2[i], ring2[(i+6)%12], ring2[j], ring2[(j+6)%12]]
            axial_coords = [(0,0)] + ring1 + subset
            positions = [axial_to_xy(q, r) for q, r in axial_coords]
            initial_configs.append((positions, [0.0]*N))
    
    # Consecutive 4
    for start in range(0, 12, 2):
        subset = [ring2[(start + k) % 12] for k in range(4)]
        axial_coords = [(0,0)] + ring1 + subset
        positions = [axial_to_xy(q, r) for q, r in axial_coords]
        initial_configs.append((positions, [0.0]*N))
    
    # Symmetric cross
    subset = [ring2[0], ring2[6], ring2[3], ring2[9]]
    axial_coords = [(0,0)] + ring1 + subset
    positions = [axial_to_xy(q, r) for q, r in axial_coords]
    initial_configs.append((positions, [0.0]*N))
    
    # Deduplicate
    unique_configs = []
    seen = set()
    for cfg, ang in initial_configs:
        key = tuple((round(p[0], 4), round(p[1], 4)) for p in cfg)
        if key not in seen:
            seen.add(key)
            unique_configs.append((cfg, ang))
    
    # Evaluate initial configs
    for positions, angles in unique_configs:
        if time.time() > deadline - 30.0:
            break
        if not _check_all_disjoint(positions, angles):
            continue
        for outer_rot in [0.0, 30.0]:
            side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_positions = positions[:]
                best_angles = angles[:]
                best_outer_rotation = outer_rot
    
    if best_positions is None:
        best_positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        best_angles = [0.0] * N
        best_side = _min_enclosing_hex_side(best_positions, best_angles)
        best_outer_rotation = 0.0
    
    # Direct differentiable optimization on the support function objective
    # Try multiple outer rotations
    if time.time() < deadline - 20.0:
        for outer_rot_deg in [0.0, 15.0, 30.0]:
            if time.time() > deadline - 15.0:
                break
            outer_rot_rad = math.radians(outer_rot_deg)
            
            # Start from best known configuration
            x0 = np.zeros(3 * N)
            for i in range(N):
                x0[3*i] = best_positions[i][0]
                x0[3*i+1] = best_positions[i][1]
                x0[3*i+2] = math.radians(best_angles[i])
            
            def obj_and_grad(params):
                return _support_function_obj(params, outer_rot_rad)
            
            try:
                result = minimize(
                    lambda p: _support_function_obj(p, outer_rot_rad)[0],
                    x0,
                    jac=lambda p: _support_function_obj(p, outer_rot_rad)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if time.time() > deadline - 10.0:
                    continue
                
                opt_positions = [(result.x[3*i], result.x[3*i+1]) for i in range(N)]
                opt_angles = [math.degrees(result.x[3*i+2]) for i in range(N)]
                
                if _check_all_disjoint(opt_positions, opt_angles):
                    side = _min_enclosing_hex_side(opt_positions, opt_angles, (0.0, 0.0), outer_rot_deg)
                    if side < best_side:
                        best_side = side
                        best_positions = opt_positions
                        best_angles = opt_angles
                        best_outer_rotation = outer_rot_deg
            except Exception:
                pass
    
    # Nelder-Mead polish on the actual objective (non-differentiable due to max)
    if time.time() < deadline - 10.0:
        def nm_objective(params):
            positions = [(params[2*i], params[2*i+1]) for i in range(N)]
            angles = best_angles[:]
            if not _check_all_disjoint(positions, angles):
                return 1e6
            side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), best_outer_rotation)
            return side
        
        x0 = np.array([best_positions[i][0], best_positions[i][1]] for i in range(N)).flatten()
        
        try:
            result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if time.time() > deadline - 5.0:
                pass
            else:
                opt_positions = [(result.x[2*i], result.x[2*i+1]) for i in range(N)]
                if _check_all_disjoint(opt_positions, best_angles):
                    side = _min_enclosing_hex_side(opt_positions, best_angles, (0.0, 0.0), best_outer_rotation)
                    if side < best_side:
                        best_side = side
                        best_positions = opt_positions
        except Exception:
            pass
    
    # Also try optimizing angles with Nelder-Mead
    if time.time() < deadline - 5.0:
        def angle_obj(angle_params):
            positions = best_positions[:]
            angles = [math.degrees(angle_params[i]) for i in range(N)]
            if not _check_all_disjoint(positions, angles):
                return 1e6
            side = _min_enclosing_hex_side(positions, angles, (0.0, 0.0), best_outer_rotation)
            return side
        
        x0 = np.array([math.radians(a) for a in best_angles])
        
        try:
            result = minimize(
                angle_obj,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if time.time() > deadline - 2.0:
                pass
            else:
                opt_angles = [math.degrees(result.x[i]) for i in range(N)]
                if _check_all_disjoint(best_positions, opt_angles):
                    side = _min_enclosing_hex_side(best_positions, opt_angles, (0.0, 0.0), best_outer_rotation)
                    if side < best_side:
                        best_side = side
                        best_angles = opt_angles
        except Exception:
            pass
    
    # Final sweep of outer rotations
    if time.time() < deadline - 2.0:
        for outer_rot in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0]:
            side = _min_enclosing_hex_side(best_positions, best_angles, (0.0, 0.0), outer_rot)
            if side < best_side:
                best_side = side
                best_outer_rotation = outer_rot
    
    # Add tiny epsilon for strict containment
    best_side *= (1.0 + 1e-9)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, best_outer_rotation
# EVOLVE-BLOCK-END
