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

def hexagon_min_distance(c1, a1_deg, c2, a2_deg):
    """
    Compute the minimum distance between two unit hexagons.
    Returns 0.0 if they overlap or touch.
    """
    v1 = get_hex_vertices(c1[0], c1[1], 1.0, math.radians(a1_deg))
    v2 = get_hex_vertices(c2[0], c2[1], 1.0, math.radians(a2_deg))
    
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
    
    min_dist = float('inf')
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Gap between the two projections
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        
        if gap < min_dist:
            min_dist = gap
    
    return min_dist

def compute_overlap_penalty_exact(inner, margin=0.001):
    """
    Compute a smooth penalty based on actual hexagon distances.
    Penalizes pairs where distance < margin.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = hexagon_min_distance(
                (inner[i][0], inner[i][1]), inner[i][2],
                (inner[j][0], inner[j][1]), inner[j][2]
            )
            if dist < margin:
                diff = margin - dist
                penalty += diff * diff
    
    return penalty

def compute_overlap_penalty_fast(inner, min_dist=SQRT3 - 0.005):
    """
    Fast center-distance based penalty for initial optimization.
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
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tiny spacing to ensure disjointness
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
    def objective_outer(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    best_result = None
    best_val = float('inf')
    
    # Try multiple diverse initial angles
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
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint Optimization with coordinate descent
    # Phase 4a: Fast center-distance penalty for bulk optimization
    def objective_joint_fast(params):
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
        penalty = compute_overlap_penalty_fast(inner_list)
        
        return side + 500.0 * penalty
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(initial_params, dtype=float)
    current_val = objective_joint_fast(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Coordinate descent: optimize each hexagon's 3 params, then outer 3 params
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    for iteration in range(5000):
        if time.time() > deadline - 20:
            break
        
        # Cycle through all hexagons
        improved = False
        for i in range(N):
            # Optimize hexagon i's position and angle
            base_idx = i * 3
            for _ in range(50):
                if time.time() > deadline - 20:
                    break
                    
                # Try small perturbations
                best_neighbor = None
                best_neighbor_val = current_val
                
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        for da in [-step_size, 0, step_size]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            neighbor = current_params.copy()
                            neighbor[base_idx] += dx
                            neighbor[base_idx+1] += dy
                            neighbor[base_idx+2] += da
                            nv = objective_joint_fast(neighbor)
                            if nv < best_neighbor_val - 1e-10:
                                best_neighbor_val = nv
                                best_neighbor = neighbor
                
                if best_neighbor is not None:
                    current_params = best_neighbor
                    current_val = best_neighbor_val
                    improved = True
                else:
                    break
            
            # Decay step size occasionally
            if iteration % 50 == 0 and iteration > 0:
                step_size *= 0.9
        
        # Optimize outer hexagon
        outer_base = N * 3
        for _ in range(20):
            if time.time() > deadline - 20:
                break
            best_neighbor = None
            best_neighbor_val = current_val
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    for da in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0 and da == 0:
                            continue
                        neighbor = current_params.copy()
                        neighbor[outer_base] += dx
                        neighbor[outer_base+1] += dy
                        neighbor[outer_base+2] += da
                        nv = objective_joint_fast(neighbor)
                        if nv < best_neighbor_val - 1e-10:
                            best_neighbor_val = nv
                            best_neighbor = neighbor
            
            if best_neighbor is not None:
                current_params = best_neighbor
                current_val = best_neighbor_val
                improved = True
            else:
                break
        
        if current_val < best_val_joint:
            best_params = current_params.copy()
            best_val_joint = current_val
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            break
    
    # Phase 4b: Refine with exact distance penalty
    if time.time() < deadline - 10:
        def objective_joint_exact(params):
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
            penalty = compute_overlap_penalty_exact(inner_list, margin=0.001)
            
            return side + 1000.0 * penalty
        
        # Start from the fast solution
        current_params = best_params.copy()
        current_val = objective_joint_exact(current_params)
        
        # Short coordinate descent with exact penalty
        step_size = 0.02
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                base_idx = i * 3
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    best_neighbor = None
                    best_neighbor_val = current_val
                    for dx in [-step_size, 0, step_size]:
                        for dy in [-step_size, 0, step_size]:
                            for da in [-step_size, 0, step_size]:
                                if dx == 0 and dy == 0 and da == 0:
                                    continue
                                neighbor = current_params.copy()
                                neighbor[base_idx] += dx
                                neighbor[base_idx+1] += dy
                                neighbor[base_idx+2] += da
                                nv = objective_joint_exact(neighbor)
                                if nv < best_neighbor_val - 1e-10:
                                    best_neighbor_val = nv
                                    best_neighbor = neighbor
                    if best_neighbor is not None:
                        current_params = best_neighbor
                        current_val = best_neighbor_val
                        improved = True
                    else:
                        break
                
                if iteration % 20 == 0 and iteration > 0:
                    step_size *= 0.95
            
            outer_base = N * 3
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                best_neighbor = None
                best_neighbor_val = current_val
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        for da in [-step_size, 0, step_size]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            neighbor = current_params.copy()
                            neighbor[outer_base] += dx
                            neighbor[outer_base+1] += dy
                            neighbor[outer_base+2] += da
                            nv = objective_joint_exact(neighbor)
                            if nv < best_neighbor_val - 1e-10:
                                best_neighbor_val = nv
                                best_neighbor = neighbor
                if best_neighbor is not None:
                    current_params = best_neighbor
                    current_val = best_neighbor_val
                    improved = True
                else:
                    break
            
            if current_val < best_val_joint:
                best_params = current_params.copy()
                best_val_joint = current_val
    
    # Reconstruct best solution
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
        # Fallback to rigid body result
        best_inner = initial_inner
        best_side = best_val

    # 5. Final High-Precision Polish: Tighten outer container
    if time.time() < deadline - 2:
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner, (cx, cy), angle_deg)
        
        polish_starts = [
            [best_outer_center[0], best_outer_center[1], best_outer_angle],
            [best_outer_center[0], best_outer_center[1], best_outer_angle + 0.5],
            [best_outer_center[0], best_outer_center[1], best_outer_angle - 0.5],
            [best_outer_center[0] + 0.1, best_outer_center[1], best_outer_angle],
            [best_outer_center[0] - 0.1, best_outer_center[1], best_outer_angle],
            [best_outer_center[0], best_outer_center[1] + 0.1, best_outer_angle],
            [best_outer_center[0], best_outer_center[1] - 0.1, best_outer_angle],
        ]
        
        best_polish_val = best_side
        best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
        
        for ps in polish_starts:
            if time.time() > deadline - 1:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 20000})
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

