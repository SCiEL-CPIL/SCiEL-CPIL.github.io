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

def compute_required_side_lse(inner, outer_center, outer_angle_deg, temp=0.01):
    """
    Compute a smooth approximation of the minimum outer side length using log-sum-exp.
    This provides gradients for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for max
    max_p = max(projections)
    # If max_p is very negative, handle overflow/underflow
    if max_p > 1000:
        max_p = 1000
    if max_p < -1000:
        max_p = -1000
        
    # Sum of exp((p - max_p) / temp)
    sum_exp = 0.0
    for p in projections:
        sum_exp += math.exp((p - max_p) / temp)
    
    # soft_max = max_p + temp * log(sum_exp)
    soft_max = max_p + temp * math.log(sum_exp)
    
    # S >= 2 * soft_max / sqrt(3)
    return 2.0 * soft_max / SQRT3

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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.001, beta=30.0):
    """
    Compute a smooth penalty for overlaps based on center distances.
    Uses a smoother exponential-based penalty for better optimization behavior.
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
    
    # Penalty: smooth approximation using exp
    dists = np.sqrt(np.maximum(0, dists_sq_upper))
    diff = min_dist - dists
    
    # Use softplus-like smooth max for better gradient behavior
    positive_diff = np.maximum(0.0, diff)
    # Smooth penalty: beta * (exp(-beta * dist) - exp(-beta * min_dist)) for overlapping
    # Simpler: use quadratic but with smooth transition
    penalty = np.sum(positive_diff * positive_diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    # More compact cluster: center + 6 neighbors + 4 from next ring
    # Using a 3-4-3 arrangement for better radial symmetry
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-1, 2), (2, -2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Standard hex grid: 
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Use a very small epsilon to ensure initial disjointness for SAT checks, 
    # but the optimizer will move them.
    epsilon = 1e-5
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
    # Using the smooth LSE surrogate for better convergence
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_lse(initial_inner, (cx, cy), angle_deg, temp=0.03)
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
        
    best_result = None
    best_val_smooth = float('inf')
    
    # The optimal angle is likely related to the symmetry of the cluster.
    # Try a finer range of angles around likely optimal values
    init_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350]
    
    for init_angle in init_angles:
        if time.time() > deadline - 25:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # Use the smooth objective for optimization
            res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
            if res.fun < best_val_smooth:
                best_val_smooth = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        # Evaluate with exact objective to get the true side length
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = objective_outer_exact(best_result)
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
        
        # Use smooth objective for the container part to help gradient-free methods
        side = compute_required_side_lse(inner_list, (ocx, ocy), oangle, temp=0.015)
        
        # Penalty for overlaps - use stronger penalty to ensure disjointness
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 50.0 * penalty
    
    # Initial params: current best inner + best outer
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params)
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Adaptive Step Size Local Search with restarts
    # Use multiple phases with different step sizes
    phases = [
        {'step_size': 0.1, 'decay': 0.99, 'min_step': 0.005, 'max_iter': 2000},
        {'step_size': 0.05, 'decay': 0.99, 'min_step': 0.002, 'max_iter': 2000},
        {'step_size': 0.02, 'decay': 0.995, 'min_step': 0.001, 'max_iter': 3000},
    ]
    
    for phase in phases:
        if time.time() > deadline - 12:
            break
            
        step_size = phase['step_size']
        min_step_size = phase['min_step']
        decay = phase['decay']
        max_iter = phase['max_iter']
        
        iter_count = 0
        no_improve_count = 0
        max_no_improve = 200
        
        while iter_count < max_iter:
            if time.time() > deadline - 12:
                break
            if no_improve_count > max_no_improve:
                break
                
            # Randomly pick a few dimensions to perturb
            num_perturb = 3
            indices = np.random.choice(len(current_params), num_perturb, replace=False)
            
            neighbor = current_params.copy()
            for idx in indices:
                delta = np.random.normal(0, step_size)
                neighbor[idx] += delta
                
            neighbor_val = objective_joint(neighbor)
            
            if neighbor_val < current_val:
                current_params = neighbor
                current_val = neighbor_val
                no_improve_count = 0
                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
            else:
                no_improve_count += 1
                # Occasionally accept worse solutions to escape local minima
                T = 0.005 * (1.0 - iter_count / max_iter)
                if T > 0:
                    delta_val = neighbor_val - current_val
                    if np.random.rand() < math.exp(-delta_val / T):
                        current_params = neighbor
                        current_val = neighbor_val
            
            # Decay step size
            step_size = max(min_step_size, step_size * decay)
            
            iter_count += 1
        
        # Restart from best if we got stuck
        if no_improve_count > max_no_improve:
            current_params = best_params.copy()
            current_val = best_val_joint
    
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
        # Calculate exact side length
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to rigid body result if joint search broke disjointness
        best_inner = initial_inner
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    # Use the smooth objective for a few more iterations, then exact
    def objective_outer_polish_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_lse(best_inner, (cx, cy), angle_deg, temp=0.008)
        
    def objective_outer_polish_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try more local starts around the current best with finer angle steps
    polish_starts = []
    for da in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
        polish_starts.append([best_outer_center[0], best_outer_center[1], best_outer_angle + da])
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 2:
            break
        try:
            # First pass with smooth objective
            res_smooth = minimize(objective_outer_polish_smooth, ps, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
            
            # Second pass with exact objective starting from smooth result
            res_exact = minimize(objective_outer_polish_exact, res_smooth.x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 8000})
            
            if res_exact.fun < best_polish_val:
                best_polish_val = res_exact.fun
                best_polish_params = res_exact.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
