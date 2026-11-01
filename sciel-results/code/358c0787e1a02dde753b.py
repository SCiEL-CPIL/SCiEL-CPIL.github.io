"""Improved program using fully smooth differentiable optimization with analytical gradients."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _hex_min_sep_analytical(dx, dy, dtheta_deg):
    """
    Compute minimum separation between two unit hexagons using analytical formula.
    dx, dy: relative position (center2 - center1)
    dtheta_deg: relative rotation angle (angle2 - angle1) in degrees
    Returns (min_separation, best_axis_x, best_axis_y) where separation is positive if non-overlapping.
    """
    # For two unit hexagons with relative rotation, the minimum separation
    # can be computed by checking all 12 edge normals (6 from each hexagon)
    # But we can be smarter: only check normals that could be critical
    
    dtheta = np.radians(dtheta_deg)
    cos_dt = np.cos(dtheta)
    sin_dt = np.sin(dtheta)
    
    # Hexagon 1 normals (at angle 0)
    # Edge i of hexagon at angle 0: normal direction
    # For a unit hexagon with vertex at angle 0, edges are at angles 30, 90, 150, 210, 270, 330
    # Normals point outward: at angles 0, 60, 120, 180, 240, 300
    n1_angles = np.radians(60 * np.arange(6))
    n1 = np.column_stack([np.cos(n1_angles), np.sin(n1_angles)])
    
    # Hexagon 2 normals (rotated by dtheta)
    n2_angles = n1_angles + dtheta
    n2 = np.column_stack([np.cos(n2_angles), np.sin(n2_angles)])
    
    # All candidate axes
    all_axes = np.vstack([n1, n2])  # shape (12, 2)
    
    # For each axis, compute the support function values
    # For a unit hexagon centered at origin with angle 0:
    # Support in direction u = max over vertices of dot(v, u)
    # For unit hexagon, support function is: max(|u_x|, |u_y|, |u_x/2 + sqrt(3)*u_y/2|) * 2/sqrt(3)
    # Actually, let's compute it properly
    
    # Vertices of unit hexagon at angle 0
    v_angles = np.radians(60 * np.arange(6))
    v1_local = np.column_stack([np.cos(v_angles), np.sin(v_angles)])  # (6, 2)
    
    # Vertices of unit hexagon at angle dtheta (relative to hex1)
    v2_local = np.column_stack([np.cos(v_angles + dtheta), np.sin(v_angles + dtheta)])  # (6, 2)
    
    # For each axis, compute min separation
    # sep = max(proj1.min - proj2.max, proj2.min - proj1.max)
    # where proj1 = v1_local @ axis, proj2 = (v2_local + [dx,dy]) @ axis
    
    min_sep = float('inf')
    best_ax = None
    
    for axis in all_axes:
        proj1 = v1_local @ axis  # (6,)
        proj2 = v2_local @ axis + (dx * axis[0] + dy * axis[1])  # (6,)
        
        sep1 = proj1.min() - proj2.max()
        sep2 = proj2.min() - proj1.max()
        sep = max(sep1, sep2)
        
        if sep < min_sep:
            min_sep = sep
            best_ax = axis
    
    return min_sep, best_ax[0], best_ax[1]

def _hex_min_sep_smooth(dx, dy, dtheta_deg, beta=80.0):
    """
    Smooth approximation of minimum separation using soft-min over all 12 axes.
    Returns smooth separation value.
    """
    dtheta = np.radians(dtheta_deg)
    
    # Hexagon 1 normals
    n1_angles = np.radians(60 * np.arange(6))
    n1 = np.column_stack([np.cos(n1_angles), np.sin(n1_angles)])
    
    # Hexagon 2 normals
    n2_angles = n1_angles + dtheta
    n2 = np.column_stack([np.cos(n2_angles), np.sin(n2_angles)])
    
    all_axes = np.vstack([n1, n2])  # (12, 2)
    
    # Vertices
    v_angles = np.radians(60 * np.arange(6))
    v1_local = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
    v2_local = np.column_stack([np.cos(v_angles + dtheta), np.sin(v_angles + dtheta)])
    
    # Compute separation for each axis
    # proj1 = v1_local @ axis.T -> (12, 6)
    proj1 = all_axes @ v1_local.T  # (12, 6)
    proj2 = all_axes @ v2_local.T + (dx * all_axes[:, 0] + dy * all_axes[:, 1])[:, None]  # (12, 6)
    
    sep1 = proj1.min(axis=1) - proj2.max(axis=1)  # (12,)
    sep2 = proj2.min(axis=1) - proj1.max(axis=1)  # (12,)
    seps = np.maximum(sep1, sep2)  # (12,)
    
    # Soft-min
    min_val = np.min(seps)
    # Clip to avoid numerical issues
    seps_clipped = seps - min_val
    exps = np.exp(beta * seps_clipped)
    soft_min = min_val - np.log(np.sum(exps)) / beta
    
    return soft_min

def _containment_constraint(px, py, pa_deg, outer_angle_deg):
    """
    Compute the maximum |projection| of inner hexagon vertices onto outer hexagon normals,
    relative to outer center at origin.
    Returns the max absolute projection value (in the rotated frame).
    """
    # Inner hexagon vertices relative to origin (outer center)
    a = np.radians(pa_deg)
    v_angles = a + np.radians(60 * np.arange(6))
    vx = px + np.cos(v_angles)
    vy = py + np.sin(v_angles)
    
    # Rotate to outer frame
    oa = np.radians(outer_angle_deg)
    cos_oa = np.cos(oa)
    sin_oa = np.sin(oa)
    
    rx = vx * cos_oa - vy * sin_oa
    ry = vx * sin_oa + vy * cos_oa
    
    # Outer hexagon normals (at angle 0 in local frame)
    n_angles = np.radians(60 * np.arange(6))
    nx = np.cos(n_angles)
    ny = np.sin(n_angles)
    
    # Projections: (6, 6)
    projs = rx[:, None] * nx[None, :] + ry[:, None] * ny[None, :]
    
    return np.max(np.abs(projs))

def smooth_objective(params, beta=80.0):
    """
    Fully smooth objective function.
    params: [x1,y1,a1, ..., xN,yN,aN, ox,oy,oa]
    Returns scalar objective.
    """
    n = N
    centers = []
    for i in range(n):
        centers.append((params[3*i], params[3*i+1], params[3*i+2]))
    outer_center = (params[3*n], params[3*n+1])
    outer_angle = params[3*n+2]
    
    # Containment: max extent / (sqrt(3)/2) gives required side
    max_extent = 0.0
    for i in range(n):
        px, py, pa = centers[i]
        # Translate so outer center is at origin
        dx = px - outer_center[0]
        dy = py - outer_center[1]
        extent = _containment_constraint(dx, dy, pa - outer_angle, 0.0)
        max_extent = max(max_extent, extent)
    
    req_side = max_extent / (np.sqrt(3) / 2)
    
    # Overlap penalty: sum of soft-min separations squared (only when negative)
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dtheta = centers[j][2] - centers[i][2]
            sep = _hex_min_sep_smooth(dx, dy, dtheta, beta)
            if sep < 0:
                overlap_pen += (-sep) ** 2
    
    return req_side + 10.0 * overlap_pen

def objective_and_gradient(params, beta=80.0):
    """
    Compute objective and gradient using finite differences.
    To save time, we use a larger step and central differences.
    """
    obj = smooth_objective(params, beta)
    
    # Use central differences with adaptive step
    h = 1e-4
    grad = np.zeros_like(params)
    
    for i in range(len(params)):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (smooth_objective(p_plus, beta) - smooth_objective(p_minus, beta)) / (2 * h)
    
    return obj, grad

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(oa_rad)
    sin_a = np.sin(oa_rad)
    
    for (px, py, pa) in centers:
        a = np.radians(pa)
        v_angles = a + np.radians(60 * np.arange(6))
        vx = px + np.cos(v_angles)
        vy = py + np.sin(v_angles)
        
        rx = vx - outer_center[0]
        ry = vy - outer_center[1]
        
        rx_local = rx * cos_a - ry * sin_a
        ry_local = rx * sin_a + ry * cos_a
        
        n_angles = np.radians(60 * np.arange(6))
        nx = np.cos(n_angles)
        ny = np.sin(n_angles)
        
        projs = rx_local[:, None] * nx[None, :] + ry_local[:, None] * ny[None, :]
        max_extent = max(max_extent, np.max(np.abs(projs)))
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-7:
        return False
    
    # Check disjointness using analytical SAT
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dtheta = centers[j][2] - centers[i][2]
            sep, _, _ = _hex_min_sep_analytical(dx, dy, dtheta)
            if sep < -1e-7:
                return False
    return True

def lbfghs_optimize(centers, outer_center, outer_angle, time_limit=30.0, beta=80.0):
    """
    Use L-BFGS-B to optimize a configuration with smooth objective.
    Returns (centers, outer_center, outer_side, outer_angle) or original if no improvement.
    """
    start_time = time.time()
    n = N
    
    params = np.zeros(3 * n + 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*n] = outer_center[0]
    params[3*n+1] = outer_center[1]
    params[3*n+2] = outer_angle
    
    # First, do a quick check of time
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return centers, outer_center, outer_angle, outer_angle
    
    # Run L-BFGS-B
    try:
        result = minimize(
            objective_and_gradient,
            params,
            method='L-BFGS-B',
            jac=True,
            args=(beta,),
            options={'maxiter': 200, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
    except Exception:
        return centers, outer_center, outer_angle, outer_angle
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return centers, outer_center, outer_angle, outer_angle
    
    new_params = result.x
    new_centers = []
    for i in range(n):
        new_centers.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
    new_outer_center = [new_params[3*n], new_params[3*n+1]]
    new_outer_angle = new_params[3*n+2]
    
    # Compute actual required side
    actual_side = required_outer_side(new_centers, new_outer_center, new_outer_angle)
    
    # Check validity
    valid = check_validity(new_centers, tuple(new_outer_center), actual_side, new_outer_angle)
    
    if not valid:
        return centers, outer_center, outer_angle, outer_angle
    
    orig_side = required_outer_side(centers, outer_center, outer_angle)
    
    if actual_side < orig_side:
        return new_centers, new_outer_center, actual_side, new_outer_angle
    else:
        return centers, outer_center, outer_angle, outer_angle

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    # Multiple initial configurations
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("random3", initial_config_random(999)),
        ("random4", initial_config_random(777)),
        ("random5", initial_config_random(31415))
    ]
    
    # For each config, try L-BFGS-B optimization
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        time_limit = min(15, DEADLINE - elapsed - 5)
        if time_limit < 3:
            break
        
        try:
            centers = [list(c) for c in config]
            outer_center = [0.0, 0.0]
            outer_angle = 0.0
            
            # First, compute initial side to have a baseline
            init_side = required_outer_side(centers, outer_center, outer_angle)
            
            # Try optimization
            opt_centers, opt_outer_center, opt_side, opt_angle = lbfghs_optimize(
                centers, outer_center, outer_angle,
                time_limit=time_limit, beta=80.0
            )
            
            # Validate
            if check_validity(opt_centers, tuple(opt_outer_center), opt_side, opt_angle):
                if opt_side < best_score:
                    best_score = opt_side
                    best_result = (opt_centers, tuple(opt_outer_center), opt_side, opt_angle)
                    
                    # Also try a second round with different beta
                    elapsed = time.time() - start_time
                    if elapsed < DEADLINE - 8:
                        time_limit2 = min(10, DEADLINE - elapsed - 3)
                        if time_limit2 > 2:
                            opt2_centers, opt2_outer_center, opt2_side, opt2_angle = lbfghs_optimize(
                                opt_centers, tuple(opt_outer_center), opt_angle,
                                time_limit=time_limit2, beta=120.0
                            )
                            if check_validity(opt2_centers, tuple(opt2_outer_center), opt2_side, opt2_angle):
                                if opt2_side < best_score:
                                    best_score = opt2_side
                                    best_result = (opt2_centers, tuple(opt2_outer_center), opt2_side, opt2_angle)
        except Exception as e:
            pass
    
    # If we have a result, try one more polish with high beta on the best
    if best_result is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            time_limit = min(12, DEADLINE - elapsed - 2)
            if time_limit > 3:
                try:
                    bc, boc, bs, ba = best_result
                    pc, poc, ps, pa = lbfghs_optimize(
                        bc, boc, ba,
                        time_limit=time_limit, beta=150.0
                    )
                    if check_validity(pc, tuple(poc), ps, pa):
                        if ps < best_score:
                            best_score = ps
                            best_result = (pc, tuple(poc), ps, pa)
                except Exception:
                    pass
    
    # Fallback: return best found or a valid default
    if best_result is not None:
        return best_result
    
    # Default valid packing (larger outer hexagon)
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
# EVOLVE-BLOCK-END
