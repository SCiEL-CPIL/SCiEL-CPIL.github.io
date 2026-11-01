"""Improved program using smooth surrogate + L-BFGS-B polish on top of SA."""
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

def sat_min_separation(c1, c2):
    """
    Compute minimum separation between two hexagons using SAT.
    Returns positive if separated, negative if overlapping.
    Also returns the axis of minimum separation.
    """
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    best_axis = None
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
            best_axis = axis
    
    return min_sep, best_axis

def _compute_hex_axes(cx, cy, angle_deg):
    """Compute the 6 edge normals of a unit hexagon at given center/angle."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Precompute cos/sin for all 6 edges
    thetas = a + np.radians(60 * np.arange(6))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # Edge i goes from vertex i to vertex i+1
    # Normal is perpendicular to edge, pointing outward
    # Edge direction: (cos(theta_{i+1}) - cos(theta_i), sin(theta_{i+1}) - sin(theta_i))
    # For unit hexagon, edge length = 1, so normal = (-dy, dx) normalized
    dx = cos_t[1:] - cos_t[:-1]
    dy = sin_t[1:] - sin_t[:-1]
    # Wrap around for last edge
    dx = np.append(dx, cos_t[0] - cos_t[-1])
    dy = np.append(dy, sin_t[0] - sin_t[-1])
    # Normal = (-dy, dx), already unit length for unit hexagon edges
    normals = np.column_stack([-dy, dx])
    return normals

def _compute_hex_vertex_projections(cx, cy, angle_deg, axes):
    """Compute projections of hexagon vertices onto given axes. Returns (min_proj, max_proj) per axis."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    vx = cx + np.cos(thetas)
    vy = cy + np.sin(thetas)
    # Project vertices onto each axis
    # axes shape: (12, 2), vertices: (6, 2)
    # proj[i, j] = dot(vertex_j, axis_i)
    proj = axes[:, 0] * vx[None, :] + axes[:, 1] * vy[None, :]
    min_proj = proj.min(axis=1)
    max_proj = proj.max(axis=1)
    return min_proj, max_proj

def smooth_min_separation(c1, c2, beta=50.0):
    """
    Compute a smooth approximation of minimum separation using soft-min.
    Returns (smooth_sep, smooth_axis_approx).
    """
    axes1 = _compute_hex_axes(c1[0], c1[1], c1[2])
    axes2 = _compute_hex_axes(c2[0], c2[1], c2[2])
    all_axes = np.vstack([axes1, axes2])
    
    min1, max1 = _compute_hex_vertex_projections(c1[0], c1[1], c1[2], all_axes)
    min2, max2 = _compute_hex_vertex_projections(c2[0], c2[1], c2[2], all_axes)
    
    sep1 = min1 - max2
    sep2 = min2 - max1
    seps = np.maximum(sep1, sep2)
    
    min_val = np.min(seps)
    exps = np.exp(beta * (seps - min_val))
    soft_min = min_val - np.log(np.sum(exps)) / beta
    
    weights = exps / np.sum(exps)
    approx_axis = np.sum(weights[:, np.newaxis] * all_axes, axis=0)
    approx_axis_norm = np.linalg.norm(approx_axis)
    if approx_axis_norm > 1e-12:
        approx_axis = approx_axis / approx_axis_norm
    
    return soft_min, approx_axis

def smooth_required_outer_side(centers, outer_center, outer_angle, beta=30.0):
    """
    Compute a smooth approximation of the required outer side length.
    """
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    # Outer hexagon normals (in local frame)
    outer_thetas = np.radians(60 * np.arange(6))
    outer_normals = np.column_stack([np.cos(outer_thetas), np.sin(outer_thetas)])
    
    d_values = []
    
    for (px, py, pa) in centers:
        a = np.radians(pa)
        thetas = a + np.radians(60 * np.arange(6))
        vx = px + np.cos(thetas)
        vy = py + np.sin(thetas)
        
        # Relative to outer center
        rx = vx - outer_center[0]
        ry = vy - outer_center[1]
        
        # Rotate to outer frame
        rx_local = rx * cos_a - ry * sin_a
        ry_local = rx * sin_a + ry * cos_a
        
        # Project onto outer normals
        d = np.abs(rx_local[:, None] * outer_normals[None, :, 0] + ry_local[:, None] * outer_normals[None, :, 1])
        d_values.append(d.flatten())
    
    d_arr = np.concatenate(d_values)
    min_d = np.min(d_arr)
    exps = np.exp(beta * (d_arr - min_d))
    soft_max = min_d + np.log(np.sum(exps)) / beta
    
    return soft_max / (np.sqrt(3) / 2)

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    # Check containment
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep, _ = sat_min_separation(centers[i], centers[j])
            if sep < -1e-9:
                return False
    return True

def energy(centers, outer_center, outer_angle):
    """
    Compute energy of configuration.
    Returns (required_side, overlap_penalty, forces).
    forces is a list of (fx, fy, fa) for each inner hexagon.
    """
    n = len(centers)
    forces = [(0.0, 0.0, 0.0)] * n
    
    # Compute required outer side
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    # Compute overlap forces using SAT
    overlap_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep, axis = sat_min_separation(centers[i], centers[j])
            if sep < 0:
                # Overlapping: push apart along axis
                overlap_penalty += (-sep) ** 2
                push = -sep * 2.0  # force magnitude
                if axis is not None:
                    # i should move in -axis direction, j in +axis direction
                    fi = (forces[i][0] - axis[0] * push, 
                          forces[i][1] - axis[1] * push, 
                          forces[i][2])
                    fj = (forces[j][0] + axis[0] * push, 
                          forces[j][1] + axis[1] * push, 
                          forces[j][2])
                    forces[i] = fi
                    forces[j] = fj
    
    # Compute containment forces
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for i in range(n):
        px, py, pa = centers[i]
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        fx = 0.0
        fy = 0.0
        fa = 0.0
        
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = rx * nx + ry * ny
                limit = req_side * np.sqrt(3) / 2
                
                if d > limit:
                    # Push back toward center
                    excess = d - limit
                    # Force in direction opposite to normal, rotated back
                    fx_local = -nx * excess
                    fy_local = -ny * excess
                    # Rotate back to world coordinates
                    fx += fx_local * np.cos(oa_rad) - fy_local * np.sin(oa_rad)
                    fy += fx_local * np.sin(oa_rad) + fy_local * np.cos(oa_rad)
        
        forces[i] = (forces[i][0] + fx, forces[i][1] + fy, forces[i][2] + fa)
    
    return req_side, overlap_penalty, forces

def smooth_objective(params, beta=50.0):
    """
    Smooth objective function for L-BFGS-B optimization.
    params: flat array of length 3*N + 3 = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, ox,oy,oa]
    Returns scalar objective: smooth_req_side + lambda * smooth_overlap_penalty
    """
    n = N
    centers = []
    for i in range(n):
        centers.append([params[3*i], params[3*i+1], params[3*i+2]])
    outer_center = [params[3*n], params[3*n+1]]
    outer_angle = params[3*n+2]
    
    # Smooth required side
    smooth_side = smooth_required_outer_side(centers, outer_center, outer_angle, beta)
    
    # Smooth overlap penalty
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = smooth_min_separation(centers[i], centers[j], beta)
            if sep < 0:
                overlap_pen += (-sep) ** 2
    
    return smooth_side + 5.0 * overlap_pen

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    # Hexagonal lattice spacing: 2*sqrt(3) for non-overlapping unit hexagons
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    # Center at origin, arrange in hexagonal pattern
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

def simulated_annealing(init_centers, max_iter=50000, T_start=2.0, T_end=0.01, 
                        cooling_rate=0.99995, time_limit=140.0):
    """
    Run simulated annealing to optimize the packing.
    Returns (best_centers, best_outer_center, best_outer_side, best_outer_angle).
    """
    start_time = time.time()
    n = len(init_centers)
    
    # Initial state
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Compute initial energy
    req_side, overlap_pen, forces = energy(centers, outer_center, outer_angle)
    current_score = req_side + 10.0 * overlap_pen
    
    best_centers = [list(c) for c in centers]
    best_outer_center = list(outer_center)
    best_outer_angle = outer_angle
    best_score = current_score
    
    T = T_start
    
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Generate perturbation
        new_centers = [list(c) for c in centers]
        new_outer_center = list(outer_center)
        new_outer_angle = outer_angle
        
        # Perturb a random inner hexagon
        i = np.random.randint(n)
        step_size = T * 0.5
        new_centers[i][0] += np.random.normal(0, step_size)
        new_centers[i][1] += np.random.normal(0, step_size)
        new_centers[i][2] += np.random.normal(0, step_size * 30)
        new_centers[i][2] = new_centers[i][2] % 60
        
        # Occasionally perturb outer parameters
        if np.random.random() < 0.1:
            new_outer_center[0] += np.random.normal(0, step_size * 0.3)
            new_outer_center[1] += np.random.normal(0, step_size * 0.3)
        if np.random.random() < 0.05:
            new_outer_angle += np.random.normal(0, step_size * 10)
            new_outer_angle = new_outer_angle % 360
        
        # Compute new energy
        new_req_side, new_overlap_pen, new_forces = energy(new_centers, new_outer_center, new_outer_angle)
        new_score = new_req_side + 10.0 * new_overlap_pen
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            centers = new_centers
            outer_center = new_outer_center
            outer_angle = new_outer_angle
            current_score = new_score
            req_side, overlap_pen, forces = new_req_side, new_overlap_pen, new_forces
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_centers = [list(c) for c in centers]
                best_outer_center = list(outer_center)
                best_outer_angle = outer_angle
        else:
            # Use forces to guide next move (optional refinement)
            pass
        
        # Cool down
        T = max(T_end, T * cooling_rate)
        
        # Periodically recenter to prevent drift
        if it % 1000 == 0 and it > 0:
            cx = np.mean([c[0] for c in centers])
            cy = np.mean([c[1] for c in centers])
            for c in centers:
                c[0] -= cx
                c[1] -= cy
            outer_center[0] -= cx
            outer_center[1] -= cy
    
    # Final validation and cleanup
    final_req_side = required_outer_side(best_centers, best_outer_center, best_outer_angle)
    
    # Verify validity
    valid = True
    for i in range(len(best_centers)):
        for j in range(i + 1, len(best_centers)):
            sep, _ = sat_min_separation(best_centers[i], best_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try to fix by slightly expanding
        return None
    
    return (best_centers, best_outer_center, final_req_side, best_outer_angle)

def lbfghs_polish(centers, outer_center, outer_angle, time_limit=30.0):
    """
    Use L-BFGS-B to polish a near-optimal configuration.
    Returns improved (centers, outer_center, outer_side, outer_angle) or original if no improvement.
    """
    start_time = time.time()
    n = N
    
    # Build initial params
    params = np.zeros(3 * n + 3)
    for i in range(n):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*n] = outer_center[0]
    params[3*n+1] = outer_center[1]
    params[3*n+2] = outer_angle
    
    # Compute initial objective
    init_obj = smooth_objective(params, beta=80.0)
    
    # Define gradient using finite differences with larger step for speed
    h = 1e-5
    def objective_with_grad(p):
        obj = smooth_objective(p, beta=80.0)
        
        # Numerical gradient - use one-sided differences for speed
        grad = np.zeros_like(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += h
            grad[i] = (smooth_objective(p_plus, beta=80.0) - obj) / h
        
        return obj, grad
    
    # Run L-BFGS-B with more iterations allowed
    result = minimize(
        objective_with_grad,
        params,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
    )
    
    # Check time
    if time.time() - start_time > time_limit:
        return centers, outer_center, outer_angle, outer_angle
    
    # Extract result
    new_params = result.x
    new_centers = []
    for i in range(n):
        new_centers.append([new_params[3*i], new_params[3*i+1], new_params[3*i+2]])
    new_outer_center = [new_params[3*n], new_params[3*n+1]]
    new_outer_angle = new_params[3*n+2]
    
    # Compute actual required side (not smooth)
    actual_side = required_outer_side(new_centers, new_outer_center, new_outer_angle)
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            sep, _ = sat_min_separation(new_centers[i], new_centers[j])
            if sep < -1e-9:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return centers, outer_center, outer_angle, outer_angle
    
    # Compare with original
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
        ("random3", initial_config_random(999))
    ]
    
    # Phase 1: Simulated annealing with faster settings
    sa_configs = []
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 45:
            break
        
        try:
            time_limit = min(20, DEADLINE - elapsed - 40)
            if time_limit < 5:
                break
            result = simulated_annealing(
                config, 
                max_iter=50000,
                T_start=2.0,
                T_end=0.005,
                cooling_rate=0.99998,
                time_limit=time_limit
            )
            
            if result is not None:
                centers, outer_center, outer_side, outer_angle = result
                
                # Final validation
                if check_validity(centers, tuple(outer_center), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (centers, tuple(outer_center), outer_side, outer_angle)
                    sa_configs.append((centers, outer_center, outer_side, outer_angle))
        except Exception as e:
            pass
    
    # Phase 2: L-BFGS-B polish on best SA results
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        # Polish top 3 results if time permits
        sa_configs_sorted = sorted(sa_configs, key=lambda x: x[2])
        for centers, outer_center, outer_side, outer_angle in sa_configs_sorted[:3]:
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 8:
                break
            
            try:
                time_limit = min(25, DEADLINE - elapsed - 5)
                if time_limit < 5:
                    break
                polished_centers, polished_outer_center, polished_side, polished_angle = lbfghs_polish(
                    centers, outer_center, outer_angle,
                    time_limit=time_limit
                )
                
                if check_validity(polished_centers, tuple(polished_outer_center), polished_side, polished_angle):
                    if polished_side < best_score:
                        best_score = polished_side
                        best_result = (polished_centers, tuple(polished_outer_center), polished_side, polished_angle)
            except Exception as e:
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
    
    # Compute required side for default
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    
    return inner_default, (0.0, 0.0), required_side, 0.0
# EVOLVE-BLOCK-END
