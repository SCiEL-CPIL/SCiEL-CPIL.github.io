"""Combined hexagon packing: LP-based outer optimization + force-directed refinement.
Strategy: Use scipy linprog for exact minimum enclosing hexagon, enumerate lattice
arrangements, then apply force-directed relaxation with overlap-aware repulsion.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

N = 11
DEADLINE = 140.0  # safety margin before 150s kill


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
    
    # Variables: x = [cx, cy, r] where r = inradius
    # Minimize r
    # Constraints: n_i . c - r <= -A_i  for each i
    c_obj = np.array([0.0, 0.0, 1.0])
    
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


def _hex_min_gap(c1, a1, c2, a2):
    """Compute minimum gap between two hexagons (positive if separated)."""
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


def _optimize_outer(centers, inner_angles, max_angle_steps=40):
    """Find optimal outer hexagon for fixed inner positions.
    Uses coarse sweep + fine local refinement for speed."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
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
        for da in np.linspace(-3, 3, 13):
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


def _force_directed_relax(centers, inner_angles, max_iter=500, tol=1e-8):
    """
    Force-directed relaxation to minimize the bounding hexagon side length.
    Uses gradient-like forces from active outer constraints plus overlap repulsion.
    Returns improved centers and angles.
    """
    centers = [list(c) for c in centers]
    inner_angles = list(inner_angles)
    
    best_side = float('inf')
    best_centers = [list(c) for c in centers]
    best_angles = list(inner_angles)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Compute current outer hexagon
        side, center, angle = _optimize_outer(centers, inner_angles)
        if side < best_side:
            best_side = side
            best_centers = [list(c) for c in centers]
            best_angles = list(inner_angles)
        
        sqrt3 = np.sqrt(3)
        normals = _hex_edge_normals(angle)
        
        # Collect all vertices
        all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                                for i in range(N)])
        projections = all_points @ normals.T
        A = projections.max(axis=0)
        
        # Compute which constraints are active
        vals = A - np.array(center) @ normals.T
        max_val = vals.max()
        active = vals >= max_val - 1e-7
        
        # For each inner hexagon, compute force from active outer constraints
        forces = np.zeros((N, 2))
        
        for i in range(N):
            verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
            
            for idx in np.where(active)[0]:
                n = normals[idx]
                proj = verts @ n
                max_proj = proj.max()
                
                if max_proj >= A[idx] - 1e-7:
                    forces[i] -= n * 0.05
        
        # Add repulsion between overlapping or nearly-touching hexagons
        for i in range(N):
            for j in range(i + 1, N):
                gap = _hex_min_gap(centers[i], inner_angles[i], centers[j], inner_angles[j])
                if gap < 0.05:
                    dist_vec = np.array([centers[i][0] - centers[j][0], 
                                         centers[i][1] - centers[j][1]])
                    dist = np.linalg.norm(dist_vec)
                    if dist > 1e-9:
                        force_mag = max(0.0, -gap) * 2.0 + 0.01
                        forces[i] += dist_vec / dist * force_mag
                        forces[j] -= dist_vec / dist * force_mag
        
        # Apply forces with decreasing step size
        step = 0.05 * (1 - iteration / max_iter) + 0.005
        for i in range(N):
            centers[i][0] += forces[i][0] * step
            centers[i][1] += forces[i][1] * step
        
        # Check for overlaps and revert if necessary
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            centers = [list(c) for c in best_centers]
            inner_angles = list(best_angles)
    
    return best_centers, best_angles


def _local_optimize(centers, inner_angles, outer_angle):
    """Use scipy minimize to refine inner positions for fixed outer angle."""
    sqrt3 = np.sqrt(3)
    
    def objective(x):
        if time.time() > DEADLINE:
            return 1e10
        c = x[:2*N].reshape(N, 2)
        # Check overlaps quickly
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(tuple(c[i]), inner_angles[i], tuple(c[j]), inner_angles[j]):
                    return 1e10
        side, _ = _min_enclosing_side_lp([tuple(c[i]) for i in range(N)], inner_angles, outer_angle)
        return side
    
    x0 = np.array([v for c in centers for v in c])
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < 1e9:
            new_centers = [tuple(res.x[2*i:2*i+2]) for i in range(N)]
            # Verify
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _hex_overlap(new_centers[i], inner_angles[i], new_centers[j], inner_angles[j]):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return new_centers
    except:
        pass
    return centers


def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    sqrt3 = np.sqrt(3)
    d1 = sqrt3  # distance between centers of touching flat-top hexagons
    d2 = 2 * sqrt3
    
    # Generate hexagonal lattice arrangements
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
    
    # Try all combinations of 4 from the 12 second-ring positions
    # With 0-degree rotation
    for subset in combinations(range(12), 4):
        if time.time() > DEADLINE:
            break
        centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
        inner_angles = [0.0] * N
        side, result = _try_arrangement(centers, inner_angles)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try with 30-degree rotation of inner hexagons
    if time.time() < DEADLINE - 20:
        for subset in combinations(range(12), 4):
            if time.time() > DEADLINE:
                break
            centers = [(0.0, 0.0)] + first_ring + [second_ring_positions[i] for i in subset]
            inner_angles = [30.0] * N
            side, result = _try_arrangement(centers, inner_angles)
            if side < best_side:
                best_side = side
                best_result = result
    
    # Apply force-directed relaxation to the best arrangement
    if best_result is not None and time.time() < DEADLINE - 15:
        centers, inner_angles, outer_center, outer_side, outer_angle = best_result
        relaxed_centers, relaxed_angles = _force_directed_relax(centers, inner_angles)
        side, result = _try_arrangement(relaxed_centers, relaxed_angles)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Try local optimization with scipy on the best result
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, inner_angles, outer_center, outer_side, outer_angle = best_result
        optimized_centers = _local_optimize(centers, inner_angles, outer_angle)
        side, result = _try_arrangement(optimized_centers, inner_angles)
        if side < best_side:
            best_side = side
            best_result = result
    
    # Fallback: linear arrangement
    if best_result is None:
        centers = [(2 * sqrt3 * k, 0.0) for k in range(N)]
        inner_angles = [0.0] * N
        side, center, angle = _optimize_outer(centers, inner_angles)
        best_side = side
        best_result = (centers, inner_angles, center, side, angle)
    
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
