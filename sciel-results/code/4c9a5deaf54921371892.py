import numpy as np
import time
import math
from itertools import combinations
from scipy.optimize import linprog, differential_evolution, minimize

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + math.pi * i / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = math.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + math.pi / 6.0 + math.pi * i / 3.0
        normals[i, 0] = math.cos(mid_angle)
        normals[i, 1] = math.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = math.sqrt(3)
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
    # A_matrix shape (6, 3)
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
            norm = math.sqrt(nx**2 + ny**2)
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

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    return _hex_min_distance(c1, a1, c2, a2) < -tol

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = math.sqrt(3)
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

def _optimize_outer(centers, inner_angles, steps=121):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle finely
    for outer_angle in np.linspace(0, 60, steps):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _try_arrangement(centers, inner_angles=None, fast=False):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    steps = 31 if fast else 121
    side, center, angle = _optimize_outer(centers, inner_angles, steps=steps)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    for i in range(N):
        row = i // 5
        col = i % 5
        x = col * 4.0
        y = row * 4.0
        centers.append((x, y))
    
    # Calculate enclosing hexagon
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _force_directed_relax(centers, angles, max_iter=200, tol=1e-6):
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
            norm = math.sqrt(nx**2 + ny**2)
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
                    # Push apart along the axis of minimum overlap
                    forces[i][0] -= ax * min_overlap
                    forces[i][1] -= ay * min_overlap
                    forces[j][0] += ax * min_overlap
                    forces[j][1] += ay * min_overlap
        
        if max_penetration < tol:
            break
        
        # Apply forces with damping
        step_size = 0.5
        for i in range(N):
            centers[i][0] += forces[i][0] * step_size
            centers[i][1] += forces[i][1] * step_size
    
    return centers, angles

def _generate_initial_configurations():
    """Generate diverse initial configurations for multi-start."""
    configs = []
    
    # Configuration 1: Hexagonal lattice (center + 6 ring + 4 from second ring)
    d1 = 2 * math.sqrt(3)
    d2 = 4 * math.sqrt(3)
    
    first_ring = []
    for i in range(6):
        angle = math.pi / 3 * i
        first_ring.append((d1 * math.cos(angle), d1 * math.sin(angle)))
        
    second_ring_positions = []
    for i in range(6):
        angle = math.pi / 3 * i
        second_ring_positions.append((d2 * math.cos(angle), d2 * math.sin(angle)))
        angle2 = math.pi / 3 * i + math.pi / 6
        second_ring_positions.append((d2 * math.cos(angle2), d2 * math.sin(angle2)))
    
    # Try different subsets of second ring
    subsets = [
        (0, 2, 4, 6),    # Axial
        (1, 3, 5, 7),    # Intermediate
        (0, 1, 2, 3),    # Consecutive
        (0, 3, 6, 9),    # Spread
    ]
    
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        angles = [0.0] * N
        configs.append((centers, angles))
    
    # Configuration 2: Linear arrangement
    centers = [(i * 2 * math.sqrt(3), 0.0) for i in range(N)]
    angles = [0.0] * N
    configs.append((centers, angles))
    
    # Configuration 3: Circular arrangement
    for radius in [3.0, 4.0, 5.0]:
        centers = []
        for i in range(N):
            angle = 2 * math.pi * i / N
            centers.append((radius * math.cos(angle), radius * math.sin(angle)))
        angles = [0.0] * N
        configs.append((centers, angles))
    
    # Configuration 4: Random configurations
    np.random.seed(42)
    for trial in range(5):
        centers = []
        angles = []
        for i in range(N):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            a = np.random.uniform(0, 60)
            centers.append((x, y))
            angles.append(a)
        configs.append((centers, angles))
    
    return configs

def _de_objective(var, centers_range, angles_range):
    """
    Objective function for Differential Evolution.
    var: flattened [x0, y0, a0, x1, y1, a1, ...]
    """
    # Unflatten
    centers = []
    angles = []
    for i in range(N):
        x = var[3*i]
        y = var[3*i+1]
        a = var[3*i+2]
        centers.append((x, y))
        angles.append(a)
    
    # Calculate overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            gap = _hex_min_distance(centers[i], angles[i], centers[j], angles[j])
            if gap < 0:
                penalty += gap * gap * 100.0
    
    # Calculate outer side (fast version)
    side, _, _ = _optimize_outer(centers, angles, steps=31)
    if not np.isfinite(side):
        return 1e10
    
    return side + penalty

def _optimize_with_de(initial_centers, initial_angles, max_time=20.0):
    """
    Use Differential Evolution to optimize positions and angles.
    """
    start_time = time.time()
    
    # Define bounds
    # The hexagons should be roughly within a 10x10 box
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 60))   # angle
    
    # Initial point
    x0 = []
    for i in range(N):
        x0.append(initial_centers[i][0])
        x0.append(initial_centers[i][1])
        x0.append(initial_angles[i])
    x0 = np.array(x0)
    
    # Run DE
    try:
        res = differential_evolution(
            _de_objective,
            bounds,
            x0=x0,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False # We will polish later with L-BFGS-B
        )
        
        if res.success or res.fun < 1e10:
            new_var = res.x
            new_centers = []
            new_angles = []
            for i in range(N):
                new_centers.append((new_var[3*i], new_var[3*i+1]))
                new_angles.append(new_var[3*i+2])
            
            # Verify validity
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
    
    return initial_centers, initial_angles, float('inf'), None, None

def _optimize_with_l_bfgsb(centers, angles, max_time=30.0):
    """
    Use L-BFGS-B to optimize positions and angles with smooth overlap penalties.
    """
    start_time = time.time()
    
    def flatten(centers, angles):
        var = []
        for i in range(N):
            var.append(centers[i][0])
            var.append(centers[i][1])
            var.append(math.radians(angles[i]))
        return np.array(var)
    
    def unflatten(var):
        centers = []
        angles = []
        for i in range(N):
            centers.append((var[3*i], var[3*i+1]))
            angles.append(math.degrees(var[3*i+2]))
        return centers, angles
    
    current_var = flatten(centers, angles)
    
    # Build contact graph: pairs that are close
    contact_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            dist = math.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist < 6.0:  # threshold for potential contact
                contact_pairs.append((i, j))
    
    def objective(var):
        if time.time() - start_time > max_time:
            return 1e10
        
        c, a = unflatten(var)
        
        # Check for overlaps and compute penalty
        penalty = 0.0
        for i, j in contact_pairs:
            gap = _hex_min_distance(c[i], a[i], c[j], a[j])
            if gap < 0:
                penalty += gap * gap * 1000
        
        # Get outer hexagon side
        side, _, _ = _optimize_outer(c, a)
        if not np.isfinite(side):
            return 1e10
        
        return side + penalty
    
    # Initial evaluation
    initial_obj = objective(current_var)
    
    # Run L-BFGS-B
    try:
        result = minimize(
            objective,
            current_var,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False,
                     'maxfun': 500}
        )
        
        if result.fun < initial_obj:
            new_centers, new_angles = unflatten(result.x)
            
            # Verify validity
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
        side, result = _try_arrangement(centers, angles, fast=True)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result
    
    # Also try lattice combinations for better initial configs
    d1 = 2 * math.sqrt(3)
    d2 = 4 * math.sqrt(3)
    
    first_ring = []
    for i in range(6):
        angle = math.pi / 3 * i
        first_ring.append((d1 * math.cos(angle), d1 * math.sin(angle)))
        
    second_ring_positions = []
    for i in range(6):
        angle = math.pi / 3 * i
        second_ring_positions.append((d2 * math.cos(angle), d2 * math.sin(angle)))
        angle2 = math.pi / 3 * i + math.pi / 6
        second_ring_positions.append((d2 * math.cos(angle2), d2 * math.sin(angle2)))
    
    for subset in combinations(range(12), 4):
        if time.time() - start_time > 25:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles, fast=True)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(inner_angles)
            best_result = result

    if best_result is None:
        # Fallback: Use a very sparse grid to ensure validity
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        best_init_centers = fb_centers
        best_init_angles = fb_angles
        best_init_side = fb_side

    # Run Force-Directed Relaxation first
    if best_init_centers and time.time() - start_time < DEADLINE - 40:
        relaxed_centers, relaxed_angles = _force_directed_relax(best_init_centers, best_init_angles)
        
        # Check if relaxed configuration is valid and better
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
    
    # Run Differential Evolution
    if best_init_centers and time.time() - start_time < DEADLINE - 30:
        de_centers, de_angles, de_side, de_center, de_angle = _optimize_with_de(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 30
        )
        
        if de_side < best_side:
            best_side = de_side
            best_result = (de_centers, de_angles, de_center, de_side, de_angle)
            best_init_centers = de_centers
            best_init_angles = de_angles
    
    # Run L-BFGS-B optimization
    if best_init_centers and time.time() - start_time < DEADLINE - 10:
        lbfgs_centers, lbfgs_angles, lbfgs_side, lbfgs_center, lbfgs_angle = _optimize_with_l_bfgsb(
            best_init_centers, best_init_angles, max_time=DEADLINE - time.time() - 10
        )
        
        if lbfgs_side < best_side:
            best_side = lbfgs_side
            best_result = (lbfgs_centers, lbfgs_angles, lbfgs_center, lbfgs_side, lbfgs_angle)
            best_init_centers = lbfgs_centers
            best_init_angles = lbfgs_angles
    
    # Final verification and formatting
    if best_result is None:
        # Should not happen, but use fallback
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    # Final strict check
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        # If somehow invalid, use fallback
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
