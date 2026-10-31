"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    # Vectorized vertex calculation
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack((
        cx + side * np.cos(thetas),
        cy + side * np.sin(thetas)
    ))

def compute_required_side_vectorized(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Vectorized for speed.
    
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normal_angles = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # Rotation matrix for outer frame
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Pre-compute all vertices of all inner hexagons
    # Shape: (N, 6, 2)
    all_verts = np.array([
        get_hex_vertices(ix, iy, 1.0, math.radians(iangle)) 
        for (ix, iy, iangle) in inner
    ])
    
    # Translate to outer center
    # Shape: (N, 6, 2)
    translated = all_verts - np.array([ox, oy])
    
    # Rotate points by -outer_angle to align with standard hex axes
    # x' = x cos + y sin
    # y' = -x sin + y cos
    # Note: Standard rotation by -theta is [cos theta, sin theta; -sin theta, cos theta]
    # Let's verify. If outer angle is alpha, we rotate coordinates by -alpha.
    # x_local = x cos(-alpha) - y sin(-alpha) = x cos(alpha) + y sin(alpha)
    # y_local = x sin(-alpha) + y cos(-alpha) = -x sin(alpha) + y cos(alpha)
    
    x_coords = translated[:, :, 0]
    y_coords = translated[:, :, 1]
    
    lx = x_coords * cos_a + y_coords * sin_a
    ly = -x_coords * sin_a + y_coords * cos_a
    
    # Local coordinates shape: (N, 6)
    local_coords = np.column_stack((lx, ly))
    
    # Compute projections onto all 6 normals
    # local_coords: (N, 6, 2)
    # normals: (6, 2)
    # We want proj[i, j, k] = dot(local_coords[i, j], normals[k])
    # Using einsum: 'ijk,kl->ijl'
    projections = np.einsum('ijk,kl->ijl', local_coords, normals)
    
    # Find the maximum projection across all vertices and all normals
    max_projection = np.max(projections)
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return float(2.0 * max_projection / SQRT3)

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        # Convert to list of tuples for sat_overlap_check
        vi_list = [(float(v[0]), float(v[1])) for v in vi]
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            vj_list = [(float(v[0]), float(v[1])) for v in vj]
            if sat_overlap_check(vi_list, vj_list, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """
    Compute a smooth penalty for overlaps based on center distances.
    This is a proxy for actual overlap, valid for aligned hexagons.
    Vectorized for speed.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    # Extract positions
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    # Compute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    # Only consider i < j pairs (upper triangle)
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    # Penalty: sum of (min_dist - dist)^2 where dist < min_dist
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    # This specific configuration (center + 6 neighbors + 4 more) is known to be efficient
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Standard hex grid: 
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Add a tiny epsilon to ensure strict disjointness in floating point if needed, 
    # though SAT check will verify. For optimization, exact lattice is better start.
    scale = 1.0 
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate centroid
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # 3. Optimize Outer Hexagon parameters (cx, cy, angle)
    # The inner hexagons are fixed in this phase (rigid body assumption)
    # Using vectorized compute_required_side
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side_vectorized(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    # The optimal angle is likely related to the symmetry of the cluster.
    # Let's try a range of angles.
    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 25:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_vectorized(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint Optimization: Allow small perturbations of inner hexagons
    # This step refines the "rigid" assumption by allowing local adjustments
    # that might squeeze the container slightly tighter.
    
    def objective_joint(params):
        # Unpack params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_vectorized(inner_list, (ocx, ocy), oangle)
        
        # Penalty for overlaps
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    # Initial params: current best inner + best outer
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Use Nelder-Mead for joint optimization as it handles non-smooth penalties reasonably well
    # and doesn't require gradients.
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Run a few iterations of Nelder-Mead with tight tolerances
    if time.time() < deadline - 15:
        try:
            res = minimize(objective_joint, current_params, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 10000})
            if res.fun < best_val_joint:
                best_params = res.x
                best_val_joint = res.fun
        except Exception:
            pass

    # Reconstruct best joint solution
    best_inner_joint = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    # Verify with SAT
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side_vectorized(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to initial if SAT fails
        best_inner = initial_inner
        best_side = compute_required_side_vectorized(best_inner, best_outer_center, best_outer_angle)

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    # Use a high-precision local optimization on the outer parameters only
    
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side_vectorized(best_inner, (cx, cy), angle_deg)
    
    # Try a few local starts around the current best with tighter tolerances
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 3],
        [best_outer_center[0] + 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.03, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.03, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            # Use Nelder-Mead with very tight tolerances
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    # Second round of polish with even tighter tolerances around the best found
    if time.time() < deadline - 1:
        fine_starts = [
            [best_polish_params[0], best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1], best_polish_params[2] + 0.5],
            [best_polish_params[0], best_polish_params[1], best_polish_params[2] - 0.5],
            [best_polish_params[0] + 0.01, best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0] - 0.01, best_polish_params[1], best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1] + 0.01, best_polish_params[2]],
            [best_polish_params[0], best_polish_params[1] - 0.01, best_polish_params[2]],
        ]
        
        for ps in fine_starts:
            if time.time() > deadline - 0.5:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 40000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
