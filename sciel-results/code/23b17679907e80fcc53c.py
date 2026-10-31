"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

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
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    This differentiable approximation helps gradient-based optimizers.
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
    
    proj_arr = np.array(projections)
    max_proj = float(np.log(np.sum(np.exp(beta * proj_arr))) / beta)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
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
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def compute_force_directed_penalty(params, adjacency, k_rep=100.0, k_spring=10.0, min_dist=SQRT3 - 0.002):
    """
    Force-directed penalty: repulsive for non-adjacent, spring for adjacent.
    params: flat array [x0,y0,a0, x1,y1,a1, ..., ocx, ocy, oangle]
    adjacency: set of (i,j) pairs that should be close
    """
    n = N
    xs = np.array([params[i*3] for i in range(n)])
    ys = np.array([params[i*3+1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)
    
    penalty = 0.0
    
    # Repulsive force for all pairs (Lennard-Jones like)
    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]
    
    # Soft repulsion: penalize when distance < min_dist
    diff = np.maximum(0.0, min_dist - pair_dists)
    penalty += k_rep * np.sum(diff * diff)
    
    # Spring attraction for adjacent pairs
    for (i, j) in adjacency:
        if i < n and j < n:
            d = dists[i, j]
            target = SQRT3 * 0.99
            penalty += k_spring * (d - target) ** 2
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Multiple candidate configurations
    candidates = []
    
    # Candidate 1: Compact 3-4-3 hexagonal cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_1, "cluster343"))
    
    # Candidate 2: Cross 1-6-4 (opposite pairs in ring 2)
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_2, "cross164"))
    
    # Candidate 3: Compact block 3x3 + 2
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append((axial_3, "block3x3"))
    
    # Candidate 4: 4-4-3
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_4, "443"))
    
    # Candidate 5: 5-6 two rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append((axial_5, "56rows"))
    
    # Candidate 6: Diamond-like
    axial_6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_6, "diamond"))
    
    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0
    
    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        epsilon = 1e-5
        scale = 1.0 + epsilon
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
        
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # Phase 1: Optimize outer container with fixed inner layout
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        
        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)
        
        best_result = None
        best_val = float('inf')
        
        init_angles = list(range(0, 360, 30))
        
        for init_angle in init_angles:
            if time.time() > deadline - 25:
                break
            
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1500})
                
                res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2500})
                
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_result = res2.x
            except Exception:
                pass
        
        if best_result is not None:
            outer_center = (best_result[0], best_result[1])
            outer_angle = best_result[2]
            side_val = best_val
        else:
            outer_center = (cx_avg, cy_avg)
            outer_angle = 0.0
            side_val = compute_required_side(initial_inner, outer_center, outer_angle)
        
        # Phase 2: Joint optimization with simulated annealing + force-directed penalty
        # Build adjacency for this configuration
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 * 1.15:
                    adjacency.add((i, j))
        
        def objective_joint(params):
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
            penalty = compute_overlap_penalty(inner_list)
            fd_penalty = compute_force_directed_penalty(params, adjacency, k_rep=50.0, k_spring=2.0)
            
            return side + 100.0 * penalty + 0.1 * fd_penalty
        
        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])
        
        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val
        
        T = 0.8
        T_min = 0.003
        cooling = 0.9999
        
        max_iter = 8000
        for i in range(max_iter):
            if time.time() > deadline - 20:
                break
            
            neighbor = current_params.copy()
            
            if np.random.random() < 0.75:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, 0.15)
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 2.0)
                else:
                    delta = np.random.normal(0, 0.1)
                neighbor[dim_idx] += delta
            
            neighbor_val = objective_joint(neighbor)
            
            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
                current_params = neighbor
                current_val = neighbor_val
                
                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
            
            T *= cooling
            if T < T_min:
                break
        
        best_inner_joint = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_joint.append([x, y, a])
        
        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]
        
        # Phase 3: Final high-precision polish on outer container
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner_joint, (cx, cy), angle_deg)
        
        polish_starts = [
            [best_ocx, best_ocy, best_oangle],
            [best_ocx, best_ocy, best_oangle + 3],
            [best_ocx, best_ocy, best_oangle - 3],
            [best_ocx, best_ocy, best_oangle + 6],
            [best_ocx, best_ocy, best_oangle - 6],
            [best_ocx + 0.05, best_ocy, best_oangle],
            [best_ocx - 0.05, best_ocy, best_oangle],
            [best_ocx, best_ocy + 0.05, best_oangle],
            [best_ocx, best_ocy - 0.05, best_oangle],
        ]
        
        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]
        
        for ps in polish_starts:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass
        
        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val
        
        # Verify disjointness
        if check_all_disjoint(final_inner, eps=1e-5):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fall back to initial layout if overlapping
            fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
            if fallback_side < best_global_side:
                best_global_side = fallback_side
                best_global_inner = initial_inner
                best_global_center = (cx_avg, cy_avg)
                best_global_angle = 0.0
    
    if best_global_inner is None:
        # Fallback: use first candidate
        axial_coords = candidates[0][0][:N]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side(best_global_inner, best_global_center, best_global_angle)
    
    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
