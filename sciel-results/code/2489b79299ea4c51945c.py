"""Hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Generate diverse initial configurations (hexagonal lattice, ring patterns),
then use force-directed relaxation and local search to minimize the enclosing hexagon.
The outer hexagon is optimized via exact projection computation for fixed orientation,
with a sweep over orientations.
"""
import numpy as np
import time
import math
from itertools import combinations

N = 11
DEADLINE = 142.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3.0)


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
    """Return 6 outward normal vectors of a regular hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi * (i + 0.5) / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side(centers, inner_angles, outer_angle_deg):
    """
    Given inner hexagon centers and angles, and outer hexagon angle,
    compute the minimum side length of a regular hexagon that encloses all inner hexagons.
    Returns (side, center).
    """
    normals = _hex_edge_normals(outer_angle_deg)  # 6x2
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        all_points.append(verts)
    all_points = np.vstack(all_points)  # (6*N, 2)
    
    # A_i = max_p dot(p, n_i) for each normal direction
    projections = all_points @ normals.T  # (6N, 6)
    A = projections.max(axis=0)  # (6,)
    
    # Minimize over c: max_i (A_i - dot(c, n_i))
    # Use iterative balancing
    c = np.array([0.0, 0.0])
    for _ in range(500):
        vals = A - c @ normals.T  # (6,)
        max_val = vals.max()
        active = vals >= max_val - 1e-10
        if active.sum() == 0:
            break
        # Move c toward the average of active normals
        grad = normals[active].mean(axis=0)
        # Adaptive step size
        step = 0.05
        for _trial in range(20):
            c_new = c + step * grad
            new_vals = A - c_new @ normals.T
            if new_vals.max() < max_val:
                c = c_new
                break
            step *= 0.5
        else:
            break
    
    vals = A - c @ normals.T
    max_proj = vals.max()
    side = max_proj / (SQRT3 / 2)
    
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
            norm = math.sqrt(nx * nx + ny * ny)
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


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check if all inner hexagons are inside outer and don't overlap."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True


def _optimize_outer(centers, inner_angles):
    """For fixed inner positions, find optimal outer hexagon (center, side, angle)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep
    for outer_angle in np.linspace(0, 60, 31):
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Fine refinement around best
    for outer_angle in np.linspace(best_angle - 2, best_angle + 2, 41):
        outer_angle = outer_angle % 60
        side, center = _min_enclosing_side(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle


def _force_directed_relax(centers, inner_angles, max_iters=2000, tol=1e-8):
    """
    Force-directed relaxation: push overlapping hexagons apart,
    and pull them toward the center to minimize spread.
    Returns relaxed centers.
    """
    centers = [list(c) for c in centers]
    n = len(centers)
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        forces = [[0.0, 0.0] for _ in range(n)]
        
        # Repulsion between overlapping or close hexagons
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = 2.0 * SQRT3  # minimum distance for non-overlap
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i][0] -= fx
                    forces[i][1] -= fy
                    forces[j][0] += fx
                    forces[j][1] += fy
                elif dist <= 1e-10:
                    # Random nudge
                    forces[i][0] += np.random.randn() * 0.01
                    forces[i][1] += np.random.randn() * 0.01
                    forces[j][0] += np.random.randn() * 0.01
                    forces[j][1] += np.random.randn() * 0.01
        
        # Attraction toward center (to minimize spread)
        cx_mean = sum(c[0] for c in centers) / n
        cy_mean = sum(c[1] for c in centers) / n
        for i in range(n):
            forces[i][0] += (cx_mean - centers[i][0]) * 0.01
            forces[i][1] += (cy_mean - centers[i][1]) * 0.01
        
        # Apply forces with decreasing step size
        step = 0.1 * (1.0 - it / max_iters) + 0.001
        moved = 0.0
        for i in range(n):
            centers[i][0] += forces[i][0] * step
            centers[i][1] += forces[i][1] * step
            moved += abs(forces[i][0]) + abs(forces[i][1])
        
        if moved < tol:
            break
    
    return centers


def _local_search(centers, inner_angles, n_iterations=500):
    """
    Local search: randomly perturb positions and accept if it improves the enclosing side.
    """
    best_centers = [list(c) for c in centers]
    best_side, _, _ = _optimize_outer(best_centers, inner_angles)
    
    for it in range(n_iterations):
        if time.time() > DEADLINE:
            break
        
        # Random perturbation
        trial_centers = [list(c) for c in best_centers]
        idx = np.random.randint(0, N)
        dx = np.random.randn() * 0.3
        dy = np.random.randn() * 0.3
        trial_centers[idx][0] += dx
        trial_centers[idx][1] += dy
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(trial_centers[i], inner_angles[i], trial_centers[j], inner_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        side, _, _ = _optimize_outer(trial_centers, inner_angles)
        if side < best_side - 1e-10:
            best_centers = trial_centers
            best_side = side
    
    return best_centers, best_side


def _generate_hexagonal_lattice():
    """Generate a hexagonal lattice arrangement: center + 6 first ring + 4 from second ring."""
    d1 = SQRT3
    d2 = 2 * SQRT3
    
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append([d1 * math.cos(angle), d1 * math.sin(angle)])
    
    second_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring.append([d2 * math.cos(angle), d2 * math.sin(angle)])
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring.append([d2 * math.cos(angle2), d2 * math.sin(angle2)])
    
    centers = [[0.0, 0.0]] + first_ring + second_ring[:4]
    return centers


def _generate_ring_pattern():
    """Generate a pattern with center + 6 inner ring + 4 outer ring (different angles)."""
    d1 = SQRT3 * 1.05
    d2 = SQRT3 * 2.1
    
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i + np.pi / 6
        first_ring.append([d1 * math.cos(angle), d1 * math.sin(angle)])
    
    second_ring = []
    for i in range(4):
        angle = np.pi / 2 * i
        second_ring.append([d2 * math.cos(angle), d2 * math.sin(angle)])
    
    centers = [[0.0, 0.0]] + first_ring + second_ring
    return centers


def _generate_linear():
    """Linear arrangement."""
    d = 2 * SQRT3
    centers = [[d * i, 0.0] for i in range(N)]
    return centers


def _generate_spiral():
    """Spiral arrangement."""
    centers = [[0.0, 0.0]]
    d = SQRT3
    for i in range(1, N):
        angle = i * 2.5  # golden angle-ish
        r = d * math.sqrt(i)
        centers.append([r * math.cos(angle), r * math.sin(angle)])
    return centers


def _try_configuration(centers, inner_angles=None):
    """Try a configuration: relax, optimize outer, check validity."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Force-directed relaxation
    relaxed = _force_directed_relax(centers, inner_angles, max_iters=1000)
    
    # Check overlaps after relaxation
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(relaxed[i], inner_angles[i], relaxed[j], inner_angles[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try more relaxation
        relaxed = _force_directed_relax(centers, inner_angles, max_iters=3000)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(relaxed[i], inner_angles[i], relaxed[j], inner_angles[j]):
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return float('inf'), None
    
    # Optimize outer hexagon
    side, center, angle = _optimize_outer(relaxed, inner_angles)
    
    # Verify validity
    if _check_validity(relaxed, inner_angles, center, side, angle):
        return side, (relaxed, inner_angles, center, side, angle)
    
    return float('inf'), None


def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_side = float('inf')
    best_result = None
    
    # Generate diverse initial configurations
    initial_configs = [
        _generate_hexagonal_lattice(),
        _generate_ring_pattern(),
        _generate_linear(),
        _generate_spiral(),
    ]
    
    # Also add some random configurations
    for _ in range(5):
        random_centers = [[np.random.randn() * 5, np.random.randn() * 5] for _ in range(N)]
        initial_configs.append(random_centers)
    
    # Try each configuration
    for config in initial_configs:
        if time.time() > DEADLINE:
            break
        
        # Try with angle 0
        side, result = _try_configuration(config)
        if side < best_side:
            best_side = side
            best_result = result
        
        # Try with 30-degree rotation
        if time.time() > DEADLINE:
            break
        inner_angles_30 = [30.0] * N
        side, result = _try_configuration(config, inner_angles_30)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Local search refinement on best result
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, inner_angles, center, side, angle = best_result
        refined_centers, refined_side = _local_search(centers, inner_angles, n_iterations=300)
        if refined_side < best_side:
            new_center, new_angle = _optimize_outer(refined_centers, inner_angles)[1:]
            if _check_validity(refined_centers, inner_angles, new_center, refined_side, new_angle):
                best_side = refined_side
                best_result = (refined_centers, inner_angles, new_center, refined_side, new_angle)
    
    # Fallback
    if best_result is None:
        centers = _generate_linear()
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
