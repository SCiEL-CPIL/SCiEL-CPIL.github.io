import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog, minimize
import warnings

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    x = 0
    y = 0
    for i in range(N):
        centers.append((x, y))
        x += 4.0
        if i % 2 == 1:
            y += 4.0
            x = -4.0 + (i // 2) * 8.0
    
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _hex_min_distance(c1, a1, c2, a2):
    """Compute the minimum gap between two hexagons (positive if separated)."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    min_gap = float('inf')
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle finely
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _is_valid_configuration(centers, inner_angles, outer_center, outer_side, outer_angle):
    """Strictly check if the configuration is valid (non-overlapping and contained)."""
    return _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle)

def _force_directed_relax(centers, angles, max_iter=5000, tol=1e-6):
    """Force-directed relaxation to reduce overlaps and compact the arrangement."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    for iteration in range(max_iter):
        forces = [[0.0, 0.0] for _ in range(N)]
        max_penetration = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                v2 = _hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                
                axes = get_axes(v1) + get_axes(v2)
                min_overlap = float('inf')
                best_axis = None
                
                for ax, ay in axes:
                    proj1 = v1[:, 0] * ax + v1[:, 1] * ay
                    proj2 = v2[:, 0] * ax + v2[:, 1] * ay
                    overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                    if overlap > max_penetration:
                        max_penetration = overlap
                    if overlap > 0 and overlap < min_overlap:
                        min_overlap = overlap
                        best_axis = (ax, ay)
                
                if min_overlap > 0 and best_axis is not None:
                    ax, ay = best_axis
                    forces[i][0] -= ax * min_overlap
                    forces[i][1] -= ay * min_overlap
                    forces[j][0] += ax * min_overlap
                    forces[j][1] += ay * min_overlap
    
        if max_penetration < tol:
            break
    
        step_size = 0.5
        for i in range(N):
            centers[i][0] += forces[i][0] * step_size
            centers[i][1] += forces[i][1] * step_size
    
    return centers, angles

def _contact_graph_optimize(centers, angles, max_iter=2000, tol=1e-8):
    """
    Optimize using contact graph constraints with SLSQP.
    """
    def flatten(centers, angles):
        var = []
        for i in range(N):
            var.append(centers[i][0])
            var.append(centers[i][1])
            var.append(np.radians(angles[i]))
        return np.array(var)
    
    def unflatten(var):
        centers = []
        angles = []
        for i in range(N):
            centers.append((var[3*i], var[3*i+1]))
            angles.append(np.degrees(var[3*i+2]))
        return centers, angles
    
    current_var = flatten(centers, angles)
    
    contact_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < 5.0:
                contact_pairs.append((i, j))
    
    def objective(var):
        c, a = unflatten(var)
        side, _, _ = _optimize_outer(c, a)
        if not np.isfinite(side):
            return 1e10
        penalty = 0.0
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            if gap < 0:
                penalty += gap * gap * 100
        return side + penalty
    
    def constraints(var):
        c, a = unflatten(var)
        cons = []
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            cons.append(gap)
        return cons
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                current_var,
                method='SLSQP',
                constraints=[{'type': 'ineq', 'fun': lambda v: constraints(v)}],
                bounds=None,
                options={'maxiter': max_iter, 'ftol': tol, 'disp': False}
            )
        
        if result.success or result.fun < objective(current_var):
            new_centers, new_angles = unflatten(result.x)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side, center, angle = _optimize_outer(new_centers, new_angles)
                if np.isfinite(side) and _check_validity(new_centers, new_angles, center, side, angle):
                    return new_centers, new_angles, side, center, angle
    except Exception:
        pass
    
    return centers, angles, float('inf'), None, None

def _simulate_annealing(initial_centers, initial_angles, max_time=120.0):
    """
    Simulated Annealing to optimize positions and angles.
    """
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    current_side, current_center, current_angle = _optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.5
    T_min = 0.01
    cooling_rate = 0.999
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        idx = np.random.randint(N)
        delta_x = np.random.normal(0, 0.1)
        delta_y = np.random.normal(0, 0.1)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        if np.random.rand() < 0.5:
            rot_idx = np.random.randint(N)
            delta_ang = np.random.normal(0, 5.0)
            new_angles[rot_idx] += delta_ang
            
        if np.random.rand() < 0.1:
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy
                
        if np.random.rand() < 0.1:
            drot = np.random.normal(0, 2.0)
            for i in range(N):
                new_angles[i] += drot

        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            T *= cooling_rate
            continue
            
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            current_center = new_center
            current_angle = new_angle
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]
                best_angles = list(angles)
                best_center = current_center
                best_angle = current_angle
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                centers = new_centers
                angles = new_angles
                current_side = new_side
                current_center = new_center
                current_angle = new_angle
                
        T *= cooling_rate
        
    return best_centers, best_angles, best_side, best_center, best_angle

def _generate_initial_configurations():
    """Generate diverse initial configurations for multi-start."""
    configs = []
    
    d1 = 2 * np.sqrt(3)
    d2 = 4 * np.sqrt(3)
    
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
        
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    subsets = [
        (0, 2, 4, 6),
        (1, 3, 5, 7),
        (0, 1, 2, 3),
        (0, 3, 6, 9),
    ]
    
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        angles = [0.0] * N
        configs.append((centers, angles))
    
    centers = [(i * 2 * np.sqrt(3), 0.0) for i in range(N)]
    angles = [0.0] * N
    configs.append((centers, angles))
    
    for radius in [3.0, 4.0, 5.0]:
        centers = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
        angles = [0.0] * N
        configs.append((centers, angles))
    
    return configs

def _trust_region_optimize(centers, angles, max_time=60.0):
    """
    Trust-region style optimization that simultaneously maximizes minimum
    pairwise distance and minimizes outer side length.
    """
    start_time = time.time()
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    def flatten(centers, angles):
        var = []
        for i in range(N):
            var.append(centers[i][0])
            var.append(centers[i][1])
            var.append(np.radians(angles[i]))
        return np.array(var)
    
    def unflatten(var):
        centers = []
        angles = []
        for i in range(N):
            centers.append((var[3*i], var[3*i+1]))
            angles.append(np.degrees(var[3*i+2]))
        return centers, angles
    
    current_var = flatten(centers, angles)
    
    # Build contact pairs (pairs close enough to potentially interact)
    contact_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < 8.0:
                contact_pairs.append((i, j))
    
    def objective(var):
        c, a = unflatten(var)
        side, _, _ = _optimize_outer(c, a)
        if not np.isfinite(side):
            return 1e10
        
        # Compute minimum pairwise gap
        min_gap = float('inf')
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            if gap < min_gap:
                min_gap = gap
        
        # Objective: minimize side, penalize overlaps, reward separation
        penalty = 0.0
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            if gap < 0:
                penalty += gap * gap * 500
        
        # Soft max of gaps (encourage uniform spacing)
        gap_vals = []
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            gap_vals.append(max(gap, -1.0))
        
        if gap_vals:
            # Soft-min of gaps (we want to maximize this)
            soft_min = -np.logaddexp.reduce([np.log(1e-10) - g for g in gap_vals]) + np.log(len(gap_vals))
            # Actually let's just use the min gap as a bonus
            min_gap_val = min(gap_vals)
        else:
            min_gap_val = 0.0
        
        return side - 0.1 * min_gap_val + penalty
    
    # Run a local optimization
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                current_var,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
        
        if result.fun < objective(current_var):
            new_centers, new_angles = unflatten(result.x)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side, center, angle = _optimize_outer(new_centers, new_angles)
                if np.isfinite(side) and _check_validity(new_centers, new_angles, center, side, angle):
                    return new_centers, new_angles, side, center, angle
    except Exception:
        pass
    
    return centers, angles, float('inf'), None, None

def _polish_pass(centers, angles, max_iter=1000):
    """
    High-precision polish pass: alternating between position optimization
    and angle optimization with tight tolerances.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    def flatten(centers, angles):
        var = []
        for i in range(N):
            var.append(centers[i][0])
            var.append(centers[i][1])
            var.append(np.radians(angles[i]))
        return np.array(var)
    
    def unflatten(var):
        centers = []
        angles = []
        for i in range(N):
            centers.append((var[3*i], var[3*i+1]))
            angles.append(np.degrees(var[3*i+2]))
        return centers, angles
    
    current_var = flatten(centers, angles)
    
    # Contact pairs
    contact_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < 6.0:
                contact_pairs.append((i, j))
    
    def objective(var):
        c, a = unflatten(var)
        side, _, _ = _optimize_outer(c, a)
        if not np.isfinite(side):
            return 1e10
        
        penalty = 0.0
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            if gap < 0:
                penalty += gap * gap * 1000
        
        return side + penalty
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                current_var,
                method='SLSQP',
                options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
            )
        
        if result.fun < objective(current_var):
            new_centers, new_angles = unflatten(result.x)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                side, center, angle = _optimize_outer(new_centers, new_angles)
                if np.isfinite(side) and _check_validity(new_centers, new_angles, center, side, angle):
                    return new_centers, new_angles, side, center, angle
    except Exception:
        pass
    
    return centers, angles, float('inf'), None, None

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = _generate_initial_configurations()
    
    best_init_centers = None
    best_init_angles = None
    best_init_side = float('inf')
    
    # Evaluate all initial configurations
    for centers, angles in initial_configs:
        if time.time() - start_time > 15:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result
    
    # Also try lattice combinations for better initial configs
    d1 = 2 * np.sqrt(3)
    d2 = 4 * np.sqrt(3)
    
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
        
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    for subset in combinations(range(12), 4):
        if time.time() - start_time > 25:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(inner_angles)
            best_result = result

    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Run Force-Directed Relaxation first
    if best_init_centers and time.time() - start_time < DEADLINE - 40:
        relaxed_centers, relaxed_angles = _force_directed_relax(best_init_centers, best_init_angles)
        
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(relaxed_centers[i], relaxed_angles[i], relaxed_centers[j], relaxed_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side, center, angle = _optimize_outer(relaxed_centers, relaxed_angles)
            if np.isfinite(side) and _check_validity(relaxed_centers, relaxed_angles, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (relaxed_centers, relaxed_angles, center, side, angle)
                    best_init_centers = relaxed_centers
                    best_init_angles = relaxed_angles
    
    # Run Contact Graph Optimization
    if best_init_centers and time.time() - start_time < DEADLINE - 30:
        cg_centers, cg_angles, cg_side, cg_center, cg_angle = _contact_graph_optimize(
            best_init_centers, best_init_angles, max_iter=500
        )
        
        if cg_side < best_side:
            best_side = cg_side
            best_result = (cg_centers, cg_angles, cg_center, cg_side, cg_angle)
            best_init_centers = cg_centers
            best_init_angles = cg_angles
    
    # Run Trust Region Optimization (new: simultaneously optimizes spacing and size)
    if best_init_centers and time.time() - start_time < DEADLINE - 25:
        tr_centers, tr_angles, tr_side, tr_center, tr_angle = _trust_region_optimize(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 25
        )
        
        if tr_side < best_side:
            best_side = tr_side
            best_result = (tr_centers, tr_angles, tr_center, tr_side, tr_angle)
            best_init_centers = tr_centers
            best_init_angles = tr_angles
    
    # Run Simulated Annealing from the best configuration found
    if best_init_centers and time.time() - start_time < DEADLINE - 15:
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 15
        )
        
        if sa_side < float('inf') and _is_valid_configuration(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)
                best_init_centers = sa_centers
                best_init_angles = sa_angles
    
    # Final high-precision polish pass
    if best_init_centers and time.time() - start_time < DEADLINE - 5:
        pol_centers, pol_angles, pol_side, pol_center, pol_angle = _polish_pass(
            best_init_centers, best_init_angles, max_iter=500
        )
        
        if pol_side < best_side:
            best_side = pol_side
            best_result = (pol_centers, pol_angles, pol_center, pol_side, pol_angle)

    # Final verification and formatting
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    if not _is_valid_configuration(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
