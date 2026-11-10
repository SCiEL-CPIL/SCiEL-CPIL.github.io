# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all, outer_angle_deg=0.0):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, with given angle) that contains all vertices.
    """
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using vectorized SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    
    # Precompute vertices and normals for all hexagons
    all_verts = np.zeros((n, 6, 2))
    all_normals = np.zeros((n, 6, 2))
    
    for i in range(n):
        cx, cy = centers[i]
        a = np.radians(angles_deg[i])
        angles = a + np.arange(6) * np.pi / 3.0
        verts = np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
        all_verts[i] = verts
        
        for k in range(6):
            p1 = verts[k]
            p2 = verts[(k+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                all_normals[i, k] = [nx/norm, ny/norm]
            else:
                all_normals[i, k] = [1.0, 0.0]
    
    # Check all pairs
    for i in range(n):
        for j in range(i+1, n):
            # Combine normals from both hexagons
            normals = np.vstack([all_normals[i], all_normals[j]])  # (12, 2)
            
            proj_i = all_verts[i] @ normals.T  # (6, 12)
            proj_j = all_verts[j] @ normals.T  # (6, 12)
            
            min_i = np.min(proj_i, axis=0)
            max_i = np.max(proj_i, axis=0)
            min_j = np.min(proj_j, axis=0)
            max_j = np.max(proj_j, axis=0)
            
            separated = (max_i + tol < min_j) | (max_j + tol < min_i)
            
            if not np.any(separated):
                return False
    
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _initial_ring():
    """
    Create a ring arrangement: 1 center + 6 around + 5 more spread out.
    """
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # First ring: 6 hexagons at distance sqrt(3)
    d1 = np.sqrt(3.0)
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        x = d1 * np.cos(angle)
        y = d1 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # Second ring: 5 hexagons at distance ~2*sqrt(3)
    d2 = 2.0 * np.sqrt(3.0)
    for k in range(5):
        angle = k * 2 * np.pi / 5 + np.pi / 6
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    return np.array(positions), np.array(angles)

def _initial_compact():
    """
    Create a more compact arrangement using hexagonal grid coordinates.
    """
    positions = []
    angles = []
    
    # Use axial coordinates for hex grid, convert to pixel
    # s = sqrt(3) is the distance between adjacent hex centers
    s = np.sqrt(3.0)
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 neighbors
    for k in range(6):
        angle = k * np.pi / 3
        x = s * np.cos(angle)
        y = s * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    # 5 more: pick positions that fill gaps
    # These are at distance sqrt(3)*sqrt(3) = 3 in some directions
    extra_positions = [
        [s * np.cos(np.pi/3), s * np.sin(np.pi/3) + s * np.cos(0)],  # not quite right
        [s * np.cos(0), s * np.sin(0) + s * np.cos(np.pi/6)],
        [s * np.cos(2*np.pi/3), s * np.sin(2*np.pi/3) + s * np.cos(0)],
        [s * np.cos(4*np.pi/3), s * np.sin(4*np.pi/3) + s * np.cos(0)],
        [s * np.cos(5*np.pi/3), s * np.sin(5*np.pi/3) + s * np.cos(0)],
    ]
    
    # Better: use proper hex grid neighbors
    # In a hex grid with flat-top orientation, neighbors of center are at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Second ring neighbors are at distance sqrt(3)*sqrt(3) = 3
    # Let's just place them at various angles at distance 2*s
    d2 = 2 * s
    for k in range(5):
        angle = k * 2 * np.pi / 5
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    return np.array(positions), np.array(angles)

def _objective(params, outer_angle_deg=0.0):
    """
    Objective: minimize outer hex side s, with penalty for overlaps.
    params: (36,) = [x1,y1,theta1, x2,y2,theta2, ..., x12,y12,theta12]
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _soft_min_outer_side(vertices_all, outer_angle_deg=0.0, beta=100.0):
    """Smooth approximation of min outer side using log-sum-exp."""
    k = np.arange(6)
    base_angle = np.radians(outer_angle_deg)
    angles = base_angle + (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T  # (72, 6)
    max_dot = np.max(dots, axis=1)  # (6,)
    
    max_val = np.max(max_dot)
    if max_val < 0:
        max_val = 0.0
    
    shifted = max_dot - max_val
    soft_max = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    s_min = (2.0 / np.sqrt(3.0)) * soft_max
    return s_min

def _smooth_objective(params, outer_angle_deg=0.0, beta=100.0, overlap_penalty=10.0):
    """Smooth objective for scipy optimization."""
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _soft_min_outer_side(all_verts, outer_angle_deg, beta)
    
    # Overlap penalty: minimum gap between pairs
    n = len(centers)
    min_gap = np.inf
    
    for i in range(n):
        for j in range(i+1, n):
            c_diff = centers[j] - centers[i]
            dist = np.linalg.norm(c_diff)
            if dist < 1e-10:
                min_gap = 0.0
                break
            
            a_i = np.radians(angles_deg[i])
            a_j = np.radians(angles_deg[j])
            
            normals_i = np.column_stack([
                np.cos(a_i + np.pi/6 + np.arange(6)*np.pi/3),
                np.sin(a_i + np.pi/6 + np.arange(6)*np.pi/3)
            ])
            normals_j = np.column_stack([
                np.cos(a_j + np.pi/6 + np.arange(6)*np.pi/3),
                np.sin(a_j + np.pi/6 + np.arange(6)*np.pi/3)
            ])
            
            all_normals = np.vstack([normals_i, normals_j])
            
            verts_i = _hex_vertices(centers[i,0], centers[i,1], 1.0, angles_deg[i])
            verts_j = _hex_vertices(centers[j,0], centers[j,1], 1.0, angles_deg[j])
            
            proj_i = verts_i @ all_normals.T
            proj_j = verts_j @ all_normals.T
            
            min_i = np.min(proj_i, axis=0)
            max_i = np.max(proj_i, axis=0)
            min_j = np.min(proj_j, axis=0)
            max_j = np.max(proj_j, axis=0)
            
            gap1 = min_j - max_i
            gap2 = min_i - max_j
            gap = np.minimum(gap1, gap2)
            min_pair_gap = np.max(gap)
            
            if min_pair_gap < min_gap:
                min_gap = min_pair_gap
    
    if min_gap < 0:
        penalty = overlap_penalty * min_gap * min_gap
    else:
        penalty = 0.0
    
    return s + penalty

def _gradient_descent(params, learning_rate=0.01, n_steps=1000, deadline=None, outer_angle_deg=0.0):
    """
    Perform gradient descent on the positions to minimize s.
    We approximate the gradient by finite differences.
    """
    best_s = _objective(params, outer_angle_deg)
    best_params = params.copy()
    
    for step in range(n_steps):
        if deadline and time.time() > deadline:
            break
        
        # Compute gradient for each parameter
        grad = np.zeros(36)
        eps = 0.01
        
        for i in range(36):
            orig_val = params[i]
            params[i] = orig_val + eps
            s_plus = _objective(params, outer_angle_deg)
            params[i] = orig_val - eps
            s_minus = _objective(params, outer_angle_deg)
            params[i] = orig_val
            grad[i] = (s_plus - s_minus) / (2 * eps)
        
        # Update parameters
        params -= learning_rate * grad
        
        # Evaluate new s
        s_new = _objective(params, outer_angle_deg)
        
        if s_new < best_s:
            best_s = s_new
            best_params = params.copy()
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            learning_rate *= 0.95
    
    return best_params, best_s

def _scipy_polish(params, deadline, outer_angle_deg=0.0):
    """Use scipy L-BFGS-B to polish with smooth objective."""
    if time.time() > deadline - 15:
        return params
    
    try:
        from scipy.optimize import minimize
        result = minimize(
            _smooth_objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000},
            args=(outer_angle_deg, 100.0, 10.0)
        )
        
        if result.success:
            c_new = result.x.reshape(12, 3)[:, :2]
            a_new = result.x.reshape(12, 3)[:, 2]
            if _check_nonoverlap(c_new, a_new):
                return result.x
    except Exception:
        pass
    
    return params

def _local_refinement(params, deadline, outer_angle_deg=0.0):
    """
    Perform local refinement using gradient descent, simulated annealing, and scipy polish.
    """
    centers = params.reshape(12, 3)[:, :2]
    angles_deg = params.reshape(12, 3)[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side(all_verts, outer_angle_deg)
    
    # Phase 1: Gradient descent
    params, s = _gradient_descent(params, learning_rate=0.05, n_steps=2000, deadline=deadline,
                                   outer_angle_deg=outer_angle_deg)
    
    # Phase 2: Simulated annealing with smaller steps
    np.random.seed(42)
    T = 0.05
    cooling = 0.999
    step_size = 0.05
    
    current_params = params.copy()
    current_s = s
    
    for it in range(50000):
        if time.time() > deadline - 60:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, :2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < s:
                s = s_new
                params = new_params.copy()
        
        T *= cooling
        if it % 1000 == 0:
            step_size *= 0.9
    
    # Phase 3: SciPy polish
    if time.time() < deadline - 20:
        params = _scipy_polish(params, deadline, outer_angle_deg)
        centers = params.reshape(12, 3)[:, :2]
        angles_deg = params.reshape(12, 3)[:, 2]
        all_verts = _compute_all_vertices(centers, angles_deg)
        s = _min_outer_side(all_verts, outer_angle_deg)
    
    return params, s

def _simulated_annealing(params, deadline, outer_angle_deg=0.0, T0=0.3, cooling=0.9995, step_size0=0.2):
    """
    Simulated annealing for global search.
    """
    np.random.seed(42)
    T = T0
    step_size = step_size0
    
    current_params = params.copy()
    current_s = _objective(params, outer_angle_deg)
    best_params = params.copy()
    best_s = current_s
    
    max_iters = 500000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        c_new = new_params.reshape(12, 3)[:, :2]
        a_new = new_params.reshape(12, 3)[:, 2]
        
        if not _check_nonoverlap(c_new, a_new):
            continue
        
        verts_new = _compute_all_vertices(c_new, a_new)
        s_new = _min_outer_side(verts_new, outer_angle_deg)
        
        delta = s_new - current_s
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        
        if it % 1000 == 0 and it > 0:
            step_size *= 0.95
            step_size = np.clip(step_size, 0.05, 0.5)
    
    return best_params, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_honeycomb(),
        _initial_ring(),
        _initial_compact(),
    ]
    
    best_params = None
    best_s = np.inf
    
    for centers, angles_deg in initial_configs:
        params = np.column_stack([centers, angles_deg]).flatten()
        
        all_verts = _compute_all_vertices(centers, angles_deg)
        s = _min_outer_side(all_verts)
        
        if not _check_nonoverlap(centers, angles_deg):
            scale = 1.1
            while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
                scale += 0.05
            centers = centers * scale
            params = np.column_stack([centers, angles_deg]).flatten()
            all_verts = _compute_all_vertices(centers, angles_deg)
            s = _min_outer_side(all_verts)
        
        if s < best_s:
            best_s = s
            best_params = params.copy()
    
    # Global search: Simulated annealing with larger steps
    remaining_time = deadline - time.time()
    sa_deadline = start_time + remaining_time * 0.5
    
    best_params, best_s = _simulated_annealing(best_params, sa_deadline)
    
    # Local refinement
    best_params, best_s = _local_refinement(best_params, deadline)
    
    # Final verification and cleanup
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    
    # Search for best outer hexagon angle
    best_outer_angle = 0.0
    s_final = _min_outer_side(all_verts_final, 0.0)
    
    # Coarse search over outer angles
    for angle_deg in np.arange(0, 60, 0.5):
        s_candidate = _min_outer_side(all_verts_final, angle_deg)
        if s_candidate < s_final:
            s_final = s_candidate
            best_outer_angle = angle_deg
    
    # Fine search around best angle
    if time.time() < deadline - 5:
        for angle_deg in np.arange(best_outer_angle - 0.5, best_outer_angle + 0.5, 0.01):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
        
        # Even finer search
        for angle_deg in np.arange(best_outer_angle - 0.01, best_outer_angle + 0.01, 0.001):
            s_candidate = _min_outer_side(all_verts_final, angle_deg)
            if s_candidate < s_final:
                s_final = s_candidate
                best_outer_angle = angle_deg
    
    if not _check_nonoverlap(centers_final, angles_final):
        # Fallback to initial if somehow invalid
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
        best_outer_angle = 0.0
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
