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
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

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
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def hexagon_distance(c1, a1, c2, a2):
    """
    Compute the minimum distance between two unit hexagons with centers c1, c2
    and angles a1, a2 (in radians). Returns 0 if overlapping.
    Uses support function approach: for two convex polygons, the distance is
    the minimum over all edge normals of (projection gap).
    """
    v1 = get_hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = get_hex_vertices(c2[0], c2[1], 1.0, a2)
    
    # Collect all edge normals from both hexagons
    axes = []
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            norm_len = math.sqrt(edge[0]**2 + edge[1]**2)
            if norm_len > 1e-12:
                nx = -edge[1] / norm_len
                ny = edge[0] / norm_len
                axes.append((nx, ny))
    
    min_gap = float('inf')
    for nx, ny in axes:
        min1 = min(v[0] * nx + v[1] * ny for v in v1)
        max1 = max(v[0] * nx + v[1] * ny for v in v1)
        min2 = min(v[0] * nx + v[1] * ny for v in v2)
        max2 = max(v[0] * nx + v[1] * ny for v in v2)
        
        # Gap on this axis (positive means separated)
        gap = max(min2 - max1, min1 - max2)
        if gap < min_gap:
            min_gap = gap
    
    # If min_gap > 0, polygons are separated by at least min_gap
    # If min_gap <= 0, they overlap or touch
    return max(0.0, min_gap)

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on actual hexagon distances.
    Uses the minimum distance between hexagon boundaries.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    penalty = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        ai_rad = math.radians(ai)
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            aj_rad = math.radians(aj)
            
            dist = hexagon_distance((xi, yi), ai_rad, (xj, yj), aj_rad)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    # Use a compact cluster: center + 6 ring + 4 more from second ring
    # Axial coordinates (q, r) for a more symmetric/compact arrangement
    axial_coords = [
        (0, 0),          # center
        (1, 0), (1, -1), (0, -1),  # 3 neighbors
        (-1, 0), (-1, 1), (0, 1),  # 3 neighbors
        (2, -1), (2, -2), (1, -2),  # 3 from second ring
        (0, 2)           # 1 more from second ring
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Standard hex grid with pointy-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    epsilon = 1e-4
    scale = 1.0 + epsilon
    
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
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    # The optimal angle is likely related to the symmetry of the cluster.
    # The cluster has some symmetry, but not full 6-fold.
    # Let's try a range of angles.
    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 30:
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
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

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
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        
        # Penalty for overlaps
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    # Initial params: current best inner + best outer
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Use Nelder-Mead for the joint optimization as it handles non-smooth penalties reasonably well
    # and is robust for low-dimensional-ish problems if we restrict the search space.
    # However, 36 dimensions is high. Let's use a limited step size or just stick to the rigid body if time is tight.
    # Given the 150s limit and that the rigid body optimization took very little time, we can afford a local search.
    
    # We will use a simple local search with random restarts or just one Nelder-Mead run.
    # Nelder-Mead on 36 dims might be slow. Let's use a gradient-free method with constraints?
    # Better: Use the rigid body result as the starting point and do a "polish" with small steps.
    
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Simulated Annealing or Random Local Search
    # Let's do a simple local search with decreasing step sizes.
    step_size = 0.1
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        
        # Randomly pick a few dimensions to perturb
        # To keep it fast, perturb only 2-3 dimensions at a time
        num_perturb = 3
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            # Determine step size based on dimension type
            # Inner positions: ~0.1, Inner angles: ~0.1 rad, Outer: ~0.1
            delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
            
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
        else:
            # Occasionally accept worse solutions to escape local minima?
            # For packing, usually greedy is fine if we start good.
            pass
        
        # Decay step size
        step_size *= 0.99

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
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to rigid body result if joint search broke disjointness
        best_inner = initial_inner
        best_side = best_val
        # Re-optimize outer for the rigid body if needed? No, we already did that.

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try multiple local starts around the current best with finer angle sampling
    polish_starts = []
    for da in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        polish_starts.append([best_outer_center[0], best_outer_center[1], best_outer_angle + da])
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 20000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    # Final high-precision polish with very tight tolerances
    if time.time() < deadline - 1:
        try:
            res = minimize(objective_outer_polish, 
                          [best_outer_center[0], best_outer_center[1], best_outer_angle],
                          method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 50000})
            if res.fun < best_polish_val:
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = res.x[2]
                best_side = res.fun
        except Exception:
            pass
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
