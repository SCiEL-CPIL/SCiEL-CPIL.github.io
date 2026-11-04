"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_rad=0.0, center=None):
    """Return 6 vertices of a unit regular hexagon rotated by angle_rad, optionally translated by center."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = math.cos(theta)
        y = math.sin(theta)
        if center is not None:
            x += center[0]
            y += center[1]
        verts.append([x, y])
    return np.array(verts)

def _outer_hexagon_vertices(side_length, angle_rad=0.0, center=None):
    """Return 6 vertices of a regular hexagon with given side length."""
    # Circumradius = side_length for regular hexagon
    R = side_length
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        if center is not None:
            x += center[0]
            y += center[1]
        verts.append([x, y])
    return np.array(verts)

def _compute_min_side_length(centers, angles, outer_angle_rad=0.0):
    """
    Compute the minimum side length S such that all inner hexagons fit inside
    the outer hexagon centered at origin with rotation outer_angle_rad.
    
    The outer hexagon with side length S and rotation phi has the property that
    a point (x,y) is inside iff for all 6 normals n_k:
        n_k . (x,y) <= S * sqrt(3) / 2
    
    where n_k are unit vectors at angles phi + k*pi/3.
    """
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    
    max_abs_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        # Project all vertices onto each normal
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            max_abs_proj = max(max_abs_proj, np.max(np.abs(projs)))
    
    # S * sqrt(3)/2 >= max_abs_proj
    # S >= max_abs_proj * 2 / sqrt(3)
    S = max_abs_proj * 2.0 / math.sqrt(3.0)
    return S

def _sat_distance(poly1, poly2):
    """
    Compute minimum distance between two convex polygons using SAT.
    Returns 0 if overlapping, positive if separated.
    poly1, poly2 are (6, 2) arrays of vertices in order.
    """
    min_dist = float('inf')
    
    # Get all edge normals from both polygons
    def get_normals(poly):
        normals = []
        n = len(poly)
        for i in range(n):
            edge = poly[(i+1) % n] - poly[i]
            # Normal is perpendicular to edge
            norm = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(norm)
            if norm_len > 1e-12:
                normals.append(norm / norm_len)
        return normals
    
    normals1 = get_normals(poly1)
    normals2 = get_normals(poly2)
    all_normals = normals1 + normals2
    
    for n in all_normals:
        # Project both polygons onto this normal
        proj1 = poly1 @ n
        proj2 = poly2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for overlap on this axis
        if max1 < min2 or max2 < min1:
            # Gap exists
            if max1 < min2:
                gap = min2 - max1
            else:
                gap = min1 - max2
            min_dist = min(min_dist, gap)
        else:
            # Overlap on this axis, but we need to check if polygons actually overlap
            # For SAT, if all axes have overlap, polygons intersect
            # We'll handle this by checking if any axis has a gap
            pass
    
    # If we found a gap on any axis, return it
    if min_dist < float('inf'):
        return min_dist
    
    # If no gap found, polygons overlap
    return 0.0

def _total_overlap_penalty(centers, angles, eps=0.01):
    """Compute total squared penalty for overlaps between inner hexagons."""
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _sat_distance(polys[i], polys[j])
            if d < eps:
                penalty += (eps - d) ** 2
    return penalty

def _objective(params, outer_angle_rad):
    """
    Objective function: minimize outer side length + overlap penalty.
    params: 3N array [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    """
    centers = params[0::3].copy()
    ys = params[1::3].copy()
    angles = params[2::3].copy()
    
    # Reconstruct centers
    c = np.column_stack([centers, ys])
    
    # Compute side length
    S = _compute_min_side_length(c, angles, outer_angle_rad)
    
    # Compute overlap penalty
    pen = _total_overlap_penalty(c, angles)
    
    return S + 50.0 * pen

def _is_valid(centers, angles, S, outer_angle_rad=0.0, tol=1e-4):
    """Check if configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps
    polys = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = _sat_distance(polys[i], polys[j])
            if d < tol:
                return False
    
    # Check containment
    outer_normals = np.array([
        [math.cos(outer_angle_rad + k * math.pi / 3.0), 
         math.sin(outer_angle_rad + k * math.pi / 3.0)]
        for k in range(6)
    ])
    limit = S * math.sqrt(3) / 2.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        for k in range(6):
            n = outer_normals[k]
            projs = verts @ n
            if np.max(np.abs(projs)) > limit + tol:
                return False
    return True

def _seed_solution():
    """Generate a valid seed solution using a hexagonal lattice with 30-degree rotations for outer ring."""
    d = math.sqrt(3)  # Distance between centers of touching hexagons
    centers = []
    angles = []
    
    # Central hexagon
    centers.append([0, 0])
    angles.append(0.0)
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        cx = d * math.cos(theta)
        cy = d * math.sin(theta)
        centers.append([cx, cy])
        # Rotate to fit better
        angles.append(0.0)
    
    # 4 more from second ring - pick well-separated positions
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    
    # Select 4 that are far apart (alternating)
    # ring2 has 12 positions, pick every 3rd starting from 0
    selected_indices = [0, 3, 6, 9]
    for idx in selected_indices:
        if idx < len(ring2):
            centers.append(list(ring2[idx]))
            angles.append(0.0)
    
    centers = np.array(centers)
    angles = np.array(angles)
    
    S = _compute_min_side_length(centers, angles)
    S *= 1.02  # Small margin
    
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    centers0, angles0, S0 = _seed_solution()
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    best_outer_angle = 0.0
    
    # Verify seed
    if not _is_valid(best_centers, best_angles, best_S, best_outer_angle):
        # Fallback: spread out more
        d = math.sqrt(3) * 1.3
        centers_fb = []
        angles_fb = []
        centers_fb.append([0, 0])
        angles_fb.append(0.0)
        for i in range(6):
            theta = i * math.pi / 3.0
            centers_fb.append([d * math.cos(theta), d * math.sin(theta)])
            angles_fb.append(0.0)
        for i in range(4):
            theta = (2*i + 1) * math.pi / 12.0
            centers_fb.append([2*d * math.cos(theta), 2*d * math.sin(theta)])
            angles_fb.append(0.0)
        best_centers = np.array(centers_fb)
        best_angles = np.array(angles_fb)
        best_S = _compute_min_side_length(best_centers, best_angles)
        best_S *= 1.02
    
    # Define objective for optimization
    def objective_full(params):
        """
        params: 3N + 1 array [x1, y1, a1, ..., xN, yN, aN, outer_angle]
        """
        if time.time() > deadline - 5.0:
            return 1e10
        
        centers = np.column_stack([params[0::3], params[1::3]])
        angles = params[2::3]
        outer_angle = params[3*N]
        
        S = _compute_min_side_length(centers, angles, outer_angle)
        pen = _total_overlap_penalty(centers, angles)
        
        return S + 100.0 * pen
    
    # Initial parameters
    x0 = np.zeros(3*N + 1)
    for i in range(N):
        x0[3*i] = best_centers[i][0]
        x0[3*i+1] = best_centers[i][1]
        x0[3*i+2] = best_angles[i]
    x0[3*N] = best_outer_angle
    
    # Try Nelder-Mead with multiple restarts
    bounds_x = (-4, 4)
    bounds_y = (-4, 4)
    
    # Local refinement
    for restart in range(5):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb initial guess for diversity
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            # Add small random perturbation
            np.random.seed(42 + restart)
            perturbation = np.random.randn(len(x_start)) * 0.5
            x_start += perturbation
        
        try:
            res = minimize(objective_full, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-6, 
                                   'maxiter': 3000, 'maxfev': 5000})
            
            if res.fun < best_S:
                centers_new = np.column_stack([res.x[0::3], res.x[1::3]])
                angles_new = res.x[2::3]
                outer_angle_new = res.x[3*N]
                
                S_val = _compute_min_side_length(centers_new, angles_new, outer_angle_new)
                if _is_valid(centers_new, angles_new, S_val * 1.001, outer_angle_new):
                    best_S = S_val
                    best_centers = centers_new
                    best_angles = angles_new
                    best_outer_angle = outer_angle_new
        except Exception:
            pass
    
    # Final refinement with tighter tolerance
    if time.time() < deadline - 10.0:
        x_start = np.zeros(3*N + 1)
        for i in range(N):
            x_start[3*i] = best_centers[i][0]
            x_start[3*i+1] = best_centers[i][1]
            x_start[3*i+2] = best_angles[i]
        x_start[3*N] = best_outer_angle
        
        try:
            res = minimize(objective_full, x_start, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7, 
                                   'maxiter': 5000, 'maxfev': 10000})
            
            if res.fun < best_S:
                centers_new = np.column_stack([res.x[0::3], res.x[1::3]])
                angles_new = res.x[2::3]
                outer_angle_new = res.x[3*N]
                
                S_val = _compute_min_side_length(centers_new, angles_new, outer_angle_new)
                if _is_valid(centers_new, angles_new, S_val * 1.001, outer_angle_new):
                    best_S = S_val
                    best_centers = centers_new
                    best_angles = angles_new
                    best_outer_angle = outer_angle_new
        except Exception:
            pass
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
