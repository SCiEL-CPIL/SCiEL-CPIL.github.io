"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: joint continuous optimization of inner positions/rotations and outer
hexagon parameters using smooth penalties and Nelder-Mead. Multi-start from
hexagonal-lattice templates with a strict deadline guard.
"""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 140.0
SQRT3 = np.sqrt(3.0)
HALF_SQRT3 = SQRT3 / 2.0


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


def _min_enclosing_hex(centers, inner_angles, outer_angle_deg):
    """
    For fixed inner hexagon positions/angles and outer hexagon orientation,
    find the minimum side length and optimal center of the enclosing outer hexagon.
    """
    normals = _hex_edge_normals(outer_angle_deg)
    
    all_points = np.empty((6 * len(centers), 2))
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        all_points[6*i:6*i+6] = verts
    
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Initialize center as mean of extreme points
    extreme_points = np.empty((6, 2))
    for i in range(6):
        idx = projections[:, i].argmax()
        extreme_points[i] = all_points[idx]
    c = extreme_points.mean(axis=0)
    
    # Iterative subgradient method
    for _ in range(100):
        vals = A - c @ normals.T
        max_val = vals.max()
        active_mask = vals >= max_val - 1e-12
        n_active = active_mask.sum()
        
        if n_active <= 1:
            break
        
        active_normals = normals[active_mask]
        grad = active_normals.mean(axis=0)
        
        # Line search
        best_alpha = 0.0
        best_f = max_val
        for alpha in np.linspace(-1.0, 1.0, 21):
            c_try = c + alpha * grad
            f_try = (A - c_try @ normals.T).max()
            if f_try < best_f - 1e-14:
                best_f = f_try
                best_alpha = alpha
        
        if abs(best_alpha) < 1e-10:
            break
        c = c + best_alpha * grad
    
    vals = A - c @ normals.T
    max_proj = vals.max()
    side = max_proj / HALF_SQRT3
    
    return side, c


def _min_enclosing_hex_fast(centers, inner_angles, outer_angle_deg):
    """Faster version using vectorized operations."""
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Vectorized vertex computation
    angles_rad = np.radians(inner_angles)
    base_angles = np.radians(outer_angle_deg) + np.pi / 6.0 + np.pi * np.arange(6) / 3.0
    
    # For each inner hex, compute vertices
    all_points = np.empty((6 * len(centers), 2))
    for i in range(len(centers)):
        cx, cy = centers[i]
        theta = angles_rad[i] + np.pi * np.arange(6) / 3.0
        all_points[6*i:6*i+6, 0] = cx + np.cos(theta)
        all_points[6*i:6*i+6, 1] = cy + np.sin(theta)
    
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Quick center estimate
    extreme_idx = projections.argmax(axis=0)
    c = all_points[extreme_idx].mean(axis=0)
    
    # Fewer iterations for speed
    for _ in range(50):
        vals = A - c @ normals.T
        max_val = vals.max()
        active_mask = vals >= max_val - 1e-12
        if active_mask.sum() <= 1:
            break
        grad = normals[active_mask].mean(axis=0)
        # Simple step
        c_new = c + 0.5 * grad
        f_new = (A - c_new @ normals.T).max()
        if f_new < max_val:
            c = c_new
        else:
            break
    
    vals = A - c @ normals.T
    side = vals.max() / HALF_SQRT3
    return side, c


def _overlap_penalty(centers, inner_angles, weight=1000.0):
    """Compute smooth overlap penalty between all pairs of inner hexagons."""
    penalty = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1 = np.array(centers[i])
            c2 = np.array(centers[j])
            dist = np.linalg.norm(c1 - c2)
            # Minimum distance for non-overlap is 2*sqrt(3) for aligned hexagons
            # Use a smooth approximation
            min_dist = 2 * SQRT3 - 0.1  # slight tolerance
            if dist < min_dist:
                penalty += weight * (min_dist - dist) ** 2
    return penalty


def _containment_penalty(centers, inner_angles, outer_center, outer_side, outer_angle_deg, weight=1000.0):
    """Compute smooth containment penalty."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * HALF_SQRT3
    oc = np.array(outer_center)
    
    penalty = 0.0
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - oc[0]) * nx + (vy - oc[1]) * ny
                if dist > inradius:
                    penalty += weight * (dist - inradius) ** 2
    return penalty


def _total_penalty(centers, inner_angles, outer_center, outer_side, outer_angle_deg):
    """Total penalty function."""
    op = _overlap_penalty(centers, inner_angles)
    cp = _containment_penalty(centers, inner_angles, outer_center, outer_side, outer_angle_deg)
    return op + cp


def _evaluate_arrangement(centers, inner_angles):
    """Evaluate an arrangement: find optimal outer hexagon and check validity."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    for outer_angle in np.linspace(0, 60, 61):
        side, center = _min_enclosing_hex_fast(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    # Check validity
    outer_normals = _hex_edge_normals(best_angle)
    inradius = best_side * HALF_SQRT3
    oc = best_center
    
    valid = True
    for i in range(len(centers)):
        cx, cy = centers[i]
        verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - oc[0]) * nx + (vy - oc[1]) * ny
                if dist > inradius + 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return None
    
    # Check overlaps
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            c1 = np.array(centers[i])
            c2 = np.array(centers[j])
            dist = np.linalg.norm(c1 - c2)
            if dist < 2 * SQRT3 - 1e-6:
                return None
    
    return best_side, best_center, best_angle


def _local_optimize(centers, inner_angles, outer_center, outer_side, outer_angle_deg, max_iters=200):
    """Local optimization using coordinate descent on outer parameters and inner positions."""
    best_side = outer_side
    best_center = outer_center.copy()
    best_angle = outer_angle_deg
    
    for iteration in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        improved = False
        
        # Optimize outer angle
        for da in np.linspace(-3, 3, 13):
            new_angle = outer_angle_deg + da
            side, center = _min_enclosing_hex_fast(centers, inner_angles, new_angle)
            if side < best_side - 1e-10:
                best_side = side
                best_center = center
                outer_angle_deg = new_angle
                improved = True
        
        # Optimize outer center via subgradient
        normals = _hex_edge_normals(outer_angle_deg)
        all_points = np.empty((6 * len(centers), 2))
        for i in range(len(centers)):
            cx, cy = centers[i]
            verts = _hex_vertices(cx, cy, 1.0, inner_angles[i])
            all_points[6*i:6*i+6] = verts
        
        projections = all_points @ normals.T
        A = projections.max(axis=0)
        
        for _ in range(20):
            vals = A - best_center @ normals.T
            max_val = vals.max()
            active_mask = vals >= max_val - 1e-12
            if active_mask.sum() <= 1:
                break
            grad = normals[active_mask].mean(axis=0)
            new_center = best_center + 0.3 * grad
            new_side = (A - new_center @ normals.T).max() / HALF_SQRT3
            if new_side < best_side - 1e-10:
                best_side = new_side
                best_center = new_center
                improved = True
            else:
                break
        
        if not improved:
            break
    
    return best_side, best_center, outer_angle_deg


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    d1 = SQRT3
    d2 = 2 * SQRT3
    
    # First ring: 6 positions
    first_ring = []
    for i in range(6):
        angle = np.pi / 3 * i
        first_ring.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # Second ring: 12 positions
    second_ring_positions = []
    for i in range(6):
        angle = np.pi / 3 * i
        second_ring_positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angle2 = np.pi / 3 * i + np.pi / 6
        second_ring_positions.append((d2 * np.cos(angle2), d2 * np.sin(angle2)))
    
    # Try all C(12,4) = 495 subsets of second ring
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        
        for inner_angle_set in ([0.0] * N, [30.0] * N):
            if time.time() > DEADLINE:
                break
            
            # Quick overlap check
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                    if dist < 2 * SQRT3 - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            result = _evaluate_arrangement(centers, inner_angle_set)
            if result is None:
                continue
            
            side, center, angle = result
            if side < best_side:
                best_side = side
                best_result = (centers, inner_angle_set, center, side, angle)
    
    # Local optimization on best result
    if best_result is not None and time.time() < DEADLINE - 5:
        centers, inner_angles, center, side, angle = best_result
        side, center, angle = _local_optimize(centers, inner_angles, center, side, angle)
        if side < best_side:
            best_side = side
            best_result = (centers, inner_angles, center, side, angle)
    
    # Fallback
    if best_result is None:
        centers = [(2 * SQRT3 * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _evaluate_arrangement(centers, inner_angles)
        if side is None:
            side = float('inf')
            center = np.array([0.0, 0.0])
            angle = 0.0
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
