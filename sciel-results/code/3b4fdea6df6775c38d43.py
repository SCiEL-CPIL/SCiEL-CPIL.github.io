"""Hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Force-directed relaxation with overlap penalty, then optimize outer hexagon.
"""
import numpy as np
import time
import math

N = 11
DEADLINE = 142.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Given inner hexagon centers and angles, and outer hexagon angle,
    compute the minimum side length of a regular hexagon (with that angle)
    that encloses all inner hexagons. Also return the optimal center.
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)  # 6x2 array
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        all_points.append(verts)
    all_points = np.vstack(all_points)  # (6*N, 2)
    
    # For each normal direction, compute max projection
    projections = all_points @ normals.T  # (6N, 6)
    A = projections.max(axis=0)  # (6,)
    
    # Now minimize over c: max_i (A_i - dot(c, n_i))
    # Use scipy linprog for exact solution
    from scipy.optimize import linprog
    
    # Minimize t subject to: dot(c, n_i) >= A_i - t for all i
    # Variables: [c_x, c_y, t]
    # Constraints: -dot(c, n_i) + t <= -A_i for all i
    c_matrix = np.zeros((6, 3))
    b_vec = np.zeros(6)
    for i in range(6):
        c_matrix[i, 0] = -normals[i, 0]
        c_matrix[i, 1] = -normals[i, 1]
        c_matrix[i, 2] = 1.0
        b_vec[i] = -A[i]
    
    # Objective: minimize t
    obj = np.array([0.0, 0.0, 1.0])
    
    res = linprog(obj, A_ub=c_matrix, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        c = np.array([res.x[0], res.x[1]])
        side = res.x[2] / (sqrt3 / 2)
    else:
        # Fallback: use center of mass
        c = all_points.mean(axis=0)
        vals = A - c @ normals.T
        side = vals.max() / (sqrt3 / 2)
    
    return side, c


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using separating axis theorem."""
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
    for ax, ay in axes:
        proj1 = v1 @ np.array([ax, ay])
        proj2 = v2 @ np.array([ax, ay])
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _hex_min_distance(c1, a1, c2, a2):
    """Compute minimum distance between two hexagons (positive if separated, negative if overlapping)."""
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
    min_dist = float('inf')
    for ax, ay in axes:
        proj1 = v1 @ np.array([ax, ay])
        proj2 = v2 @ np.array([ax, ay])
        gap = min(proj2.max(), proj1.max()) - max(proj1.min(), proj2.min())
        if gap < min_dist:
            min_dist = gap
    return min_dist


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check if all inner hexagons are inside outer and don't overlap."""
    sqrt3 = np.sqrt(3)
    
    # Check containment: each inner vertex must be inside outer hexagon
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check non-overlap between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True


def _smooth_overlap_penalty(centers, inner_angles):
    """Compute smooth penalty for overlaps between hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = _hex_min_distance(centers[i], inner_angles[i], centers[j], inner_angles[j])
            if dist < 0:
                penalty += dist * dist
    return penalty


def _force_directed_relax(centers, inner_angles, steps=500, lr=0.01):
    """Use force-directed relaxation to minimize overlaps and spread hexagons."""
    centers = [list(c) for c in centers]
    inner_angles = list(inner_angles)
    
    for step in range(steps):
        forces = [np.zeros(2) for _ in range(N)]
        angles_forces = [0.0] * N
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                dist = _hex_min_distance(centers[i], inner_angles[i], centers[j], inner_angles[j])
                if dist < 0.1:  # Close or overlapping
                    # Compute gradient of distance w.r.t. positions
                    # Approximate: push apart along center-to-center vector
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    d = math.sqrt(dx*dx + dy*dy) + 1e-12
                    fx = dx / d
                    fy = dy / d
                    # Force magnitude proportional to overlap
                    mag = max(0, 0.1 - dist) * 2.0
                    forces[i] += np.array([-fx, -fy]) * mag
                    forces[j] += np.array([fx, fy]) * mag
        
        # Attraction to center (keep packing compact)
        for i in range(N):
            cx, cy = centers[i]
            forces[i] += np.array([-cx, -cy]) * 0.01
        
        # Apply forces
        for i in range(N):
            centers[i][0] += lr * forces[i][0]
            centers[i][1] += lr * forces[i][1]
        
        # Decay learning rate
        lr *= 0.999
    
    return [tuple(c) for c in centers], inner_angles


def _generate_initial_configurations():
    """Generate diverse initial configurations for 11 hexagons."""
    configs = []
    d1 = np.sqrt(3)
    d2 = 2 * np.sqrt(3)
    
    # Config 1: Center + 6 ring + 4 second ring (standard)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # Add 4 from second ring
    for i in range(4):
        angle = np.pi / 6 + np.pi / 3 * i
        centers.append((d2 * np.cos(angle), d2 * np.sin(angle)))
    configs.append((centers[:N], [0.0] * N))
    
    # Config 2: 3-4-3-1 row pattern
    centers2 = []
    for i in range(3):
        centers2.append((i * d1, 0.0))
    for i in range(4):
        centers2.append((i * d1 - d1/2, 1.5))
    for i in range(3):
        centers2.append((i * d1, 3.0))
    centers2.append((d1, 1.5))
    configs.append((centers2[:N], [0.0] * N))
    
    # Config 3: Random perturbation of config 1
    rng = np.random.default_rng(42)
    centers3 = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers3.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    for i in range(4):
        angle = np.pi / 6 + np.pi / 3 * i
        centers3.append((d2 * np.cos(angle), d2 * np.sin(angle)))
    centers3 = [(x + rng.normal(0, 0.1), y + rng.normal(0, 0.1)) for x, y in centers3[:N]]
    configs.append((centers3, [0.0] * N))
    
    # Config 4: Different second ring selection
    centers4 = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers4.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # Pick different 4 from second ring
    for i in [0, 2, 4, 6]:
        angle = np.pi / 3 * i
        centers4.append((d2 * np.cos(angle), d2 * np.sin(angle)))
    configs.append((centers4[:N], [0.0] * N))
    
    # Config 5: Rotated inner hexagons
    centers5 = [(0.0, 0.0)]
    for i in range(6):
        angle = np.pi / 3 * i
        centers5.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    for i in range(4):
        angle = np.pi / 6 + np.pi / 3 * i
        centers5.append((d2 * np.cos(angle), d2 * np.sin(angle)))
    configs.append((centers5[:N], [30.0] * N))
    
    return configs


def _optimize_outer(centers, inner_angles):
    """For fixed inner positions, find optimal outer hexagon (center, side, angle)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try a range of outer angles
    for outer_angle in np.linspace(0, 60, 61):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return the best side length found."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Verify no overlaps
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


def _simulate_annealing(centers, inner_angles, steps=200, T0=0.5, T1=0.01):
    """Simulated annealing to optimize hexagon positions."""
    centers = [list(c) for c in centers]
    inner_angles = list(inner_angles)
    best_centers = [list(c) for c in centers]
    best_penalty = _smooth_overlap_penalty(centers, inner_angles)
    
    for step in range(steps):
        T = T0 * (T1 / T0) ** (step / steps)
        
        # Propose random move
        i = np.random.randint(N)
        move_type = np.random.randint(3)
        
        if move_type == 0:
            # Move position
            dx = np.random.normal(0, 0.1)
            dy = np.random.normal(0, 0.1)
            old_x, old_y = centers[i]
            centers[i] = [old_x + dx, old_y + dy]
        elif move_type == 1:
            # Rotate
            da = np.random.normal(0, 5)
            old_angle = inner_angles[i]
            inner_angles[i] = old_angle + da
        else:
            # Small position jitter
            dx = np.random.normal(0, 0.05)
            dy = np.random.normal(0, 0.05)
            old_x, old_y = centers[i]
            centers[i] = [old_x + dx, old_y + dy]
        
        new_penalty = _smooth_overlap_penalty(centers, inner_angles)
        delta = new_penalty - best_penalty
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            best_penalty = new_penalty
            best_centers = [list(c) for c in centers]
        else:
            # Revert
            if move_type in [0, 2]:
                centers[i] = [old_x, old_y]
            else:
                inner_angles[i] = old_angle
    
    return [tuple(c) for c in best_centers], inner_angles, best_penalty


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate initial configurations
    configs = _generate_initial_configurations()
    
    for centers, inner_angles in configs:
        if time.time() > DEADLINE:
            break
        
        # Force-directed relaxation
        centers_relaxed, angles_relaxed = _force_directed_relax(centers, inner_angles, steps=200, lr=0.02)
        
        # Try the relaxed configuration
        side, result = _try_arrangement(centers_relaxed, angles_relaxed)
        if side < best_side:
            best_side = side
            best_result = result
        
        # Simulated annealing
        if time.time() < DEADLINE - 30:
            centers_sa, angles_sa, penalty = _simulate_annealing(centers_relaxed, angles_relaxed, steps=100, T0=0.3, T1=0.005)
            side_sa, result_sa = _try_arrangement(centers_sa, angles_sa)
            if side_sa < best_side:
                best_side = side_sa
                best_result = result_sa
    
    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
