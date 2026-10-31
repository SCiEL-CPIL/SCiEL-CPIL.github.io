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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
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
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    epsilon = 1e-6
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

    # 3. Optimize Outer Hexagon using scipy with multiple starts
    # Variables: [cx, cy, angle_deg]
    
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 25:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
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
        # Fallback to grid search if scipy fails or times out
        best_side = float('inf')
        for outer_angle_deg in range(0, 360, 5):
            current_cx = cx_avg
            current_cy = cy_avg
            step = 0.5
            for _ in range(3):
                improved = False
                for dx in np.linspace(-step, step, 5):
                    for dy in np.linspace(-step, step, 5):
                        oc = (current_cx + dx, current_cy + dy)
                        side = compute_required_side(initial_inner, oc, outer_angle_deg)
                        if side < best_side - 1e-6:
                            best_side = side
                            best_outer_center = oc
                            best_outer_angle = outer_angle_deg
                            current_cx = oc[0]
                            current_cy = oc[1]
                            improved = True
                if not improved:
                    break
                step *= 0.5
            if time.time() > deadline - 5:
                break

    # 4. Joint Optimization of Inner Positions and Outer Container
    # We treat the outer center and angle as part of the optimization variables
    # to allow the container to adapt to the inner configuration.
    
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
    
    # Simulated Annealing for joint optimization with adaptive step sizes
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9997
    
    # Adaptive step sizes: start larger, shrink as we converge
    base_step_inner = 0.3
    base_step_outer = 0.2
    
    max_iter = 20000
    no_improve_count = 0
    patience = 2000
    
    for i in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Generate neighbor
        neighbor = current_params.copy()
        
        # Adaptive temperature scaling for step size
        temp_scale = max(0.1, T)
        
        # 80% chance to move an inner hexagon, 20% chance to move outer
        if np.random.random() < 0.8:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim == 2:  # angle
                delta = np.random.normal(0, 5.0 * temp_scale)
            else:  # position
                delta = np.random.normal(0, base_step_inner * temp_scale)
            neighbor[idx*3 + dim] += delta
        else:
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2: # angle
                delta = np.random.normal(0, 4.0 * temp_scale)
            else:
                delta = np.random.normal(0, base_step_outer * temp_scale)
            neighbor[dim] += delta
        
        neighbor_val = objective_joint(neighbor)
        
        # Accept if better or with probability exp(-delta/T)
        if neighbor_val < current_val or (T > 1e-10 and np.random.random() < math.exp(-(neighbor_val - current_val) / T)):
            current_params = neighbor
            current_val = neighbor_val
            
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Reduce base step sizes if we've been making progress
        if no_improve_count % 500 == 0 and no_improve_count > 0:
            base_step_inner *= 0.95
            base_step_outer *= 0.95
        
        T *= cooling
        if T < T_min:
            break
        
        # Early termination if no improvement for a long time and T is low
        if no_improve_count > patience and T < 0.05:
            break
    
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
        # Fallback to initial if SAT fails
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    # Use a high-precision local optimization on the outer parameters only
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try a few local starts around the current best with tighter tolerances
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 3],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 7],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 7],
        [best_outer_center[0] + 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.03, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.03, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.03, best_outer_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] + 0.02, best_outer_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] - 0.02, best_outer_angle],
        [best_outer_center[0] + 0.02, best_outer_center[1] - 0.02, best_outer_angle],
        [best_outer_center[0] - 0.02, best_outer_center[1] + 0.02, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            # Use COBYLA for better handling of the non-smooth max objective
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 15000})
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
                              options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 20000})
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
