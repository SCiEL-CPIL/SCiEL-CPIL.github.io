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

def compute_overlap_penalty_smooth(inner, min_dist=SQRT3 - 0.001, beta=50.0):
    """
    Compute a smooth penalty for overlaps based on center distances.
    Uses a smoother function than quadratic for better optimization behavior.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    dists = np.sqrt(dists_sq_upper)
    diff = min_dist - dists
    
    # Smooth penalty: use exp(-beta * max(0, diff)) style
    # For overlapping pairs (diff > 0), penalty grows smoothly
    positive_diff = np.maximum(0.0, diff)
    # Use a smooth approximation: penalty = sum of (1 - exp(-beta * positive_diff))
    # This is bounded and smooth
    penalty = np.sum(1.0 - np.exp(-beta * positive_diff))
    
    # Also add a quadratic term for strong penalty at large overlaps
    quad_penalty = np.sum(positive_diff * positive_diff)
    
    return float(5.0 * penalty + quad_penalty)

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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.001):
    """
    Compute a smooth penalty for overlaps based on center distances.
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
    deadline = start_time + 148.0  # Safety margin for 150s limit

    # 1. Try multiple hexagonal cluster configurations
    # Each config is a list of axial coordinates (q, r)
    configs = [
        # Config 1: Center + 6 neighbors + 4 more (original)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ],
        # Config 2: More radially symmetric
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 2), (-2, 1), (-1, -2)
        ],
        # Config 3: Compact diamond-like shape
        [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (0, 2)
        ],
        # Config 4: Hexagonal ring with center
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ],
        # Config 5: Tighter arrangement
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (0, 2)
        ],
    ]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # For each config, try to optimize
    for config_idx, axial_coords in enumerate(configs):
        if time.time() > deadline - 40:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Convert axial to Cartesian with slight expansion
        epsilon = 1e-4
        scale = 1.0 + epsilon
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
            
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Calculate centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # 2. Optimize Outer Hexagon parameters for this config
        def objective_outer_smooth(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side_lse(inner, (cx, cy), angle_deg, temp=0.05)
        
        def objective_outer_exact(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side(inner, (cx, cy), angle_deg)
            
        best_result = None
        best_val_smooth = float('inf')
        
        # Try multiple initial angles
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 35:
                break
                
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
                if res.fun < best_val_smooth:
                    best_val_smooth = res.fun
                    best_result = res.x
            except Exception:
                pass
                
        if best_result is None:
            continue
            
        best_outer_center_cfg = (best_result[0], best_result[1])
        best_outer_angle_cfg = best_result[2]
        best_side_cfg = objective_outer_exact(best_result)
        
        # 3. Joint Optimization for this config
        def objective_joint(params, inner_init=initial_inner):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side_lse(inner_list, (ocx, ocy), oangle, temp=0.02)
            penalty = compute_overlap_penalty_smooth(inner_list)
            
            return side + 10.0 * penalty
        
        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([best_outer_center_cfg[0], best_outer_center_cfg[1], best_outer_angle_cfg])
        
        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params_cfg = current_params.copy()
        best_val_joint_cfg = current_val
        
        # Adaptive Step Size Local Search with better parameters
        step_size = 0.1
        min_step_size = 0.0005
        decay = 0.985
        max_iter = 3000
        
        iter_count = 0
        while iter_count < max_iter:
            if time.time() > deadline - 20:
                break
                
            # Perturb 2-4 dimensions
            num_perturb = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            indices = np.random.choice(len(current_params), num_perturb, replace=False)
            
            neighbor = current_params.copy()
            for idx in indices:
                # Different step sizes for position vs angle
                if idx % 3 == 2:  # Angle
                    delta = np.random.normal(0, step_size * 0.5)
                else:  # Position
                    delta = np.random.normal(0, step_size)
                neighbor[idx] += delta
                
            neighbor_val = objective_joint(neighbor)
            
            if neighbor_val < current_val:
                current_params = neighbor
                current_val = neighbor_val
                if current_val < best_val_joint_cfg:
                    best_params_cfg = current_params.copy()
                    best_val_joint_cfg = current_val
            else:
                # Simulated annealing acceptance
                T = 0.005 * max(0.1, 1.0 - iter_count / max_iter)
                if T > 0:
                    delta_val = neighbor_val - current_val
                    if delta_val < 5.0 and np.random.rand() < math.exp(-delta_val / T):
                        current_params = neighbor
                        current_val = neighbor_val
            
            step_size = max(min_step_size, step_size * decay)
            iter_count += 1
            
        # Reconstruct best joint solution for this config
        best_inner_joint_cfg = []
        for i in range(N):
            x = best_params_cfg[i*3]
            y = best_params_cfg[i*3+1]
            a = best_params_cfg[i*3+2]
            best_inner_joint_cfg.append([x, y, a])
        
        best_ocx_cfg = best_params_cfg[N*3]
        best_ocy_cfg = best_params_cfg[N*3+1]
        best_oangle_cfg = best_params_cfg[N*3+2]
        
        # Verify with SAT
        if check_all_disjoint(best_inner_joint_cfg, eps=1e-5):
            # Calculate exact side length
            exact_side = compute_required_side(best_inner_joint_cfg, (best_ocx_cfg, best_ocy_cfg), best_oangle_cfg)
            
            # Update global best if this config is better
            if exact_side < best_side:
                best_side = exact_side
                best_inner = best_inner_joint_cfg
                best_outer_center = (best_ocx_cfg, best_ocy_cfg)
                best_outer_angle = best_oangle_cfg
        else:
            # If joint search broke disjointness, use the rigid body result
            if best_side_cfg < best_side:
                best_side = best_side_cfg
                best_inner = initial_inner
                best_outer_center = best_outer_center_cfg
                best_outer_angle = best_outer_angle_cfg

    # 4. Final Polish: Tighten the outer container around the fixed inner layout
    if best_inner is None:
        # Fallback to first config
        axial_coords = configs[0][:N]
        epsilon = 1e-4
        scale = 1.0 + epsilon
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    def objective_outer_polish_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_lse(best_inner, (cx, cy), angle_deg, temp=0.01)
        
    def objective_outer_polish_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try multiple local starts around the current best
    polish_starts = []
    for da in [0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
        polish_starts.append([best_outer_center[0], best_outer_center[1], best_outer_angle + da])
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            # First pass with smooth objective
            res_smooth = minimize(objective_outer_polish_smooth, ps, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 4000})
            
            # Second pass with exact objective starting from smooth result
            res_exact = minimize(objective_outer_polish_exact, res_smooth.x, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 4000})
            
            if res_exact.fun < best_polish_val:
                best_polish_val = res_exact.fun
                best_polish_params = res_exact.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    # Final validation
    if not check_all_disjoint(best_inner, eps=1e-4):
        # If validation fails, use a safe fallback
        axial_coords = configs[0][:N]
        epsilon = 1e-3
        scale = 1.0 + epsilon
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
