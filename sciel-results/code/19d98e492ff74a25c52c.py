# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base_angles = np.arange(6) * np.pi / 3.0
    vx = np.cos(base_angles)
    vy = np.sin(base_angles)
    rx = cos_a * vx - sin_a * vy
    ry = sin_a * vx + cos_a * vy
    return np.column_stack((cx + rx, cy + ry))

def _sat_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if overlap."""
    normals = []
    for verts in [v1, v2]:
        for k in range(len(verts)):
            e = verts[(k+1)%len(verts)] - verts[k]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                normals.append(n / norm)
    
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _sat_penalty(v1, v2):
    """Calculate penalty for overlap between two convex polygons using SAT."""
    normals = []
    for verts in [v1, v2]:
        for k in range(len(verts)):
            e = verts[(k+1)%len(verts)] - verts[k]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                normals.append(n / norm)
    
    penalty = 0.0
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
        if gap < 0:
            penalty += gap * gap
    return penalty

def _outer_side_for_points(points):
    """Given points (N,2), return the minimum outer hexagon side length."""
    sqrt3 = np.sqrt(3)
    d = sqrt3 / 2.0
    normals = np.array([
        [sqrt3/2, 0.5],
        [0.0, 1.0],
        [-sqrt3/2, 0.5],
        [-sqrt3/2, -0.5],
        [0.0, -1.0],
        [sqrt3/2, -0.5]
    ])
    dots = points @ normals.T
    max_dot = np.max(dots)
    return max_dot / d

def _check_validity(centers, angles, tol=1e-9):
    """Check if all hexagons are disjoint."""
    for i in range(N):
        vi = _hex_vertices(centers[i,0], centers[i,1], angles[i])
        for j in range(i+1, N):
            vj = _hex_vertices(centers[j,0], centers[j,1], angles[j])
            if _sat_overlap(vi, vj, tol):
                return False
    return True

def _objective(params):
    """Objective: minimize outer side length + penalty for overlaps."""
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    side = _outer_side_for_points(centers)
    
    penalty = 0.0
    for i in range(N):
        vi = _hex_vertices(centers[i,0], centers[i,1], angles[i])
        for j in range(i+1, N):
            vj = _hex_vertices(centers[j,0], centers[j,1], angles[j])
            penalty += _sat_penalty(vi, vj)
    
    return side + 1000.0 * penalty

def construct():
    global DEADLINE
    start = time.time()
    
    # 1. Generate a valid initial configuration
    # Use hex grid points closest to origin
    sqrt3 = np.sqrt(3)
    max_range = 5
    candidates = []
    for r in range(-max_range, max_range + 1):
        for c in range(-max_range, max_range + 1):
            x = c * sqrt3 + (r % 2) * sqrt3 / 2
            y = r * 1.5
            candidates.append([x, y])
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    sorted_indices = np.argsort(dists)
    selected_indices = sorted_indices[:N]
    
    centers_init = candidates[selected_indices].copy()
    angles_init = np.zeros(N)
    
    # Center the configuration
    center_of_mass = centers_init.mean(axis=0)
    centers_init = centers_init - center_of_mass
    
    # Verify initial validity
    if not _check_validity(centers_init, angles_init):
        # If invalid (should not happen), jitter
        np.random.seed(0)
        centers_init += np.random.randn(N, 2) * 0.1
        if not _check_validity(centers_init, angles_init):
             # Fallback to a simple grid if still invalid
             # This is a safety net
             pass

    params_init = np.concatenate([centers_init.flatten(), angles_init])
    best_params = params_init.copy()
    best_side = _outer_side_for_points(centers_init)
    
    # 2. Optimization
    if time.time() < DEADLINE - 20:
        try:
            result = minimize(_objective, params_init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            test_params = result.x
            test_centers = test_params[:2*N].reshape(N, 2)
            test_angles = test_params[2*N:]
            
            if _check_validity(test_centers, test_angles):
                test_side = _outer_side_for_points(test_centers)
                if test_side < best_side:
                    best_params = test_params
                    best_side = test_side
        except Exception:
            pass
            
    # 3. Final Extraction
    best_centers = best_params[:2*N].reshape(N, 2)
    best_angles = best_params[2*N:]
    
    # Final validity check
    if not _check_validity(best_centers, best_angles):
        # Return initial valid state
        best_centers = centers_init
        best_angles = angles_init
        best_side = _outer_side_for_points(best_centers)

    inner = []
    for i in range(N):
        inner.append([float(best_centers[i, 0]), float(best_centers[i, 1]), float(best_angles[i])])
    
    outer_center = (0.0, 0.0)
    outer_side = float(best_side)
    outer_rotation = 0.0
    
    return inner, outer_center, outer_side, outer_rotation

# EVOLVE-BLOCK-END
