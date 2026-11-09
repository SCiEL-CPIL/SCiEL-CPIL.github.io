import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog, minimize
import warnings

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    # Place hexagons in a hexagonal lattice pattern
    sqrt3 = np.sqrt(3)
    d1 = 2 * sqrt3  # distance between centers of touching hexagons (flat-top)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # 4 more from second ring
    for i in range(4):
        angle = np.pi / 3 * i
        centers.append((2 * d1 * np.cos(angle), 2 * d1 * np.sin(angle)))
    angles = [0.0] * N
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

def _optimize_outer(centers, inner_angles, max_angle_steps=30):
    """Find optimal outer hexagon for fixed inner positions.
    Uses coarse sweep + fine local refinement for speed."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    for outer_angle in np.linspace(0, 60, max_angle_steps + 1):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Fine local refinement around best angle
    if np.isfinite(best_side):
        for da in np.linspace(-2, 2, 9):
            test_angle = best_angle + da
            side, center = _min_enclosing_side_lp(centers, inner_angles, test_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = test_angle
    
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

def _force_directed_relax(centers, angles, max_iter=3000, tol=1e-6):
    """Force-directed relaxation to reduce overlaps and compact the arrangement."""
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    sqrt3 = np.sqrt(3)
    
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
    
        # Apply forces with adaptive step size
        step_size = 0.5
        for i in range(N):
            centers[i][0] += forces[i][0] * step_size
            centers[i][1] += forces[i][1] * step_size
    
    return centers, angles

def _local_search(centers, angles, max_time=30.0):
    """
    Local search: try small perturbations of individual hexagons
    and accept if the outer side decreases.
    """
    start_time = time.time()
    centers = [list(c) for c in centers]
    angles = list(angles)
    
    current_side, _, _ = _optimize_outer(centers, angles, max_angle_steps=15)
    if not np.isfinite(current_side):
        return centers, angles, float('inf'), None, None
    
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    
    np.random.seed(123)
    step_sizes = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    while time.time() - start_time < max_time:
        improved = False
        for step in step_sizes:
            if time.time() - start_time >= max_time:
                break
            for idx in range(N):
                if time.time() - start_time >= max_time:
                    break
                # Try moving hexagon idx in random directions
                for _ in range(8):
                    dx = np.random.uniform(-step, step)
                    dy = np.random.uniform(-step, step)
                    trial_centers = [list(c) for c in centers]
                    trial_centers[idx][0] += dx
                    trial_centers[idx][1] += dy
                    
                    # Quick overlap check
                    valid = True
                    for j in range(N):
                        if j != idx:
                            if _hex_overlap(trial_centers[idx], angles[idx], trial_centers[j], angles[j]):
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    new_side, _, _ = _optimize_outer(trial_centers, angles, max_angle_steps=10)
                    if np.isfinite(new_side) and new_side < current_side - 1e-8:
                        centers = trial_centers
                        current_side = new_side
                        improved = True
                        if current_side < best_side:
                            best_side = current_side
                            best_centers = [list(c) for c in centers]
                            best_angles = list(angles)
                        break  # restart inner loop for this step
        
        if not improved:
            break
    
    return best_centers, best_angles, best_side, None, None

def _simulate_annealing(initial_centers, initial_angles, max_time=60.0):
    """
    Simulated Annealing to optimize positions and angles.
    """
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    # Initial cost
    current_side, current_center, current_angle = _optimize_outer(centers, angles, max_angle_steps=15)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.3
    T_min = 0.005
    cooling_rate = 0.998
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Generate new state
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Random perturbation
        # 1. Move a random hexagon
        idx = np.random.randint(N)
        scale = 0.05 + 0.15 * (T / T_min)
        delta_x = np.random.normal(0, scale)
        delta_y = np.random.normal(0, scale)
        new_centers[idx][0] += delta_x
        new_centers[idx][1] += delta_y
        
        # 2. Rotate a random hexagon
        if np.random.rand() < 0.4:
            rot_idx = np.random.randint(N)
            delta_ang = np.random.normal(0, 3.0)
            new_angles[rot_idx] += delta_ang
            
        # 3. Global translation
        if np.random.rand() < 0.05:
            dx = np.random.normal(0, 0.03)
            dy = np.random.normal(0, 0.03)
            for i in range(N):
                new_centers[i][0] += dx
                new_centers[i][1] += dy

        # Check validity (overlaps)
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
            
        # Calculate new cost
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles, max_angle_steps=10)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            # Accept
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
            # Accept with probability
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
    sqrt3 = np.sqrt(3)
    d1 = 2 * sqrt3  # touching distance for flat-top hexagons
    
    # Configuration 1: Perfect hexagonal lattice (center + ring1 + 4 from ring2)
    # This is the densest natural packing
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # 4 from second ring - choose the ones that form a compact shape
    ring2_positions = []
    for i in range(12):
        # Second ring has 12 positions
        if i % 2 == 0:
            angle = np.pi / 3 * (i // 2)
            r = 2 * d1
        else:
            angle = np.pi / 3 * (i // 2) + np.pi / 6
            r = 2 * d1 * np.cos(np.pi / 6) / np.cos(np.pi / 6)  # same radius
            r = d1 * np.sqrt(3) * 2 / np.sqrt(3)  # actually 2*d1 for the "corner" ones
            # Actually for hex lattice, ring 2 positions:
            # 6 at distance 2*d1 (along axes) and 6 at distance d1*sqrt(3) (between axes)
            # Let me recalculate:
            pass
        ring2_positions.append((r * np.cos(angle), r * np.sin(angle)))
    
    # Proper hex lattice second ring:
    # Positions at (2*d1, 0), (d1, d1*sqrt(3)), (-d1, d1*sqrt(3)), (-2*d1, 0),
    # (-d1, -d1*sqrt(3)), (d1, -d1*sqrt(3)) — these are the 6 "corner" positions at dist 2*d1
    # And 6 "edge" positions at dist d1*sqrt(3):
    # (d1*sqrt(3)/2 * 2, d1*sqrt(3)/2 * 2)... let me just use the standard formula.
    
    # Standard hex lattice: centers at (i*a + j*a/2, j*a*sqrt(3)/2) where a = 2*sqrt(3) for unit hexagons
    # Actually for unit hexagons (side=1), the center-to-center distance for touching is 2*sqrt(3)
    # Let me use axial coordinates
    a = 2 * sqrt3
    lattice_points = []
    for qi in range(-2, 3):
        for qj in range(-2, 3):
            x = a * (qi + qj * 0.5)
            y = a * qj * sqrt3 / 2
            lattice_points.append((x, y))
    
    # Sort by distance from origin, take closest 11
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    centers_hex = lattice_points[:11]
    angles_hex = [0.0] * 11
    configs.append((centers_hex, angles_hex))
    
    # Configuration 2: Center + 6 ring + 4 spread from ring 2
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # 4 from ring 2, spread evenly
    for i in range(4):
        angle = np.pi / 3 * i + np.pi / 6
        centers.append((2 * d1 * np.cos(angle), 2 * d1 * np.sin(angle)))
    angles = [0.0] * N
    configs.append((centers, angles))
    
    # Configuration 3: Circular arrangement
    for radius in [2.5, 3.5, 4.5]:
        centers = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            centers.append((radius * np.cos(angle), radius * np.sin(angle)))
        angles = [0.0] * N
        configs.append((centers, angles))
    
    # Configuration 4: Two rows
    centers = []
    for i in range(6):
        centers.append((i * d1, 0.0))
    for i in range(5):
        centers.append((i * d1 + d1/2, d1 * sqrt3 / 2))
    angles = [0.0] * N
    configs.append((centers, angles))
    
    return configs

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
        if time.time() - start_time > 10:
            break
        side, result = _try_arrangement(centers, angles)
        if side < best_init_side:
            best_init_side = side
            best_init_centers = [list(c) for c in centers]
            best_init_angles = list(angles)
            best_result = result
    
    # Try more hex lattice subsets
    sqrt3 = np.sqrt(3)
    a = 2 * sqrt3
    lattice_points = []
    for qi in range(-3, 4):
        for qj in range(-3, 4):
            x = a * (qi + qj * 0.5)
            y = a * qj * sqrt3 / 2
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Try various subsets of the lattice
    for k in range(11, min(25, len(lattice_points))):
        if time.time() - start_time > 20:
            break
        subset = lattice_points[:k]
        # Need exactly N points, so if k > N, we need to choose N from k
        if k == N:
            centers = subset
            inner_angles = [0.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_init_side:
                best_init_side = side
                best_init_centers = [list(c) for c in centers]
                best_init_angles = list(inner_angles)
                best_result = result
    
    # Also try random subsets of lattice points
    np.random.seed(7)
    for trial in range(20):
        if time.time() - start_time > 25:
            break
        idxs = np.random.choice(len(lattice_points), N, replace=False)
        centers = [lattice_points[i] for i in idxs]
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
    if best_init_centers and time.time() - start_time < DEADLINE - 60:
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
    
    # Run Local Search
    if best_init_centers and time.time() - start_time < DEADLINE - 40:
        ls_centers, ls_angles, ls_side, _, _ = _local_search(
            best_init_centers, best_init_angles, max_time=min(25.0, DEADLINE - time.time() - 40)
        )
        if np.isfinite(ls_side) and ls_side < best_side:
            # Verify
            side, center, angle = _optimize_outer(ls_centers, ls_angles)
            if np.isfinite(side) and _check_validity(ls_centers, ls_angles, center, side, angle):
                best_side = side
                best_result = (ls_centers, ls_angles, center, side, angle)
                best_init_centers = ls_centers
                best_init_angles = ls_angles
    
    # Run Simulated Annealing from the best configuration found
    if best_init_centers and time.time() - start_time < DEADLINE - 15:
        sa_max_time = max(5.0, DEADLINE - time.time() - 15)
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            best_init_centers, best_init_angles, max_time=sa_max_time
        )
        
        if sa_side < float('inf') and _is_valid_configuration(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)
    
    # Final verification and formatting
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    # Final strict check
    if not _is_valid_configuration(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
