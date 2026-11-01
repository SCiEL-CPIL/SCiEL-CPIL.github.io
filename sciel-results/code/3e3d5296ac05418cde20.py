"""Combined: Rigid-composite initialization + Edge-based contact optimization + Discrete angle snapping."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon as numpy array."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon."""
    base = np.radians(30)
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    normals = []
    for i in range(6):
        t = base + i * np.pi / 3.0
        nx = math.cos(t)
        ny = math.sin(t)
        rx = nx * cs - ny * sn
        ry = nx * sn + ny * cs
        normals.append((rx, ry))
    return normals

def support_function(inner, outer_center, outer_angle_deg):
    """
    Max projection of all inner hexagon vertices onto the 6 container normals.
    Returns the required side length.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array(container_normals(outer_angle_deg))
    
    # Precompute base angles for unit hexagon vertices
    base_thetas = np.arange(6) * np.pi / 3.0
    base_cos = np.cos(base_thetas)
    base_sin = np.sin(base_thetas)
    
    all_verts = []
    for (ix, iy, iang) in inner:
        a_rad = math.radians(iang)
        # Rotate the base vertices by a_rad
        thetas = a_rad + base_thetas
        vx = ix + np.cos(thetas)
        vy = iy + np.sin(thetas)
        all_verts.append(np.column_stack([vx, vy]))
    
    if not all_verts:
        return 0.0
        
    # Stack all vertices: shape (N*6, 2)
    V = np.vstack(all_verts)
    # Relative to center
    V_rel = V - np.array([ox, oy])
    
    # Project onto normals: shape (N*6, 6)
    # normals shape (6, 2)
    projs = V_rel @ normals.T
    max_proj = np.max(projs)
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    """Smooth log-sum-exp version of support function."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array(container_normals(outer_angle_deg))
    
    base_thetas = np.arange(6) * np.pi / 3.0
    
    all_verts = []
    for (ix, iy, iang) in inner:
        a_rad = math.radians(iang)
        thetas = a_rad + base_thetas
        vx = ix + np.cos(thetas)
        vy = iy + np.sin(thetas)
        all_verts.append(np.column_stack([vx, vy]))
        
    if not all_verts:
        return 0.0
        
    V = np.vstack(all_verts)
    V_rel = V - np.array([ox, oy])
    projs = V_rel @ normals.T
    P = projs.flatten()
    
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap for two convex polygons (numpy arrays)."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    for ax in axes:
        min1 = np.min(v1 @ ax)
        max1 = np.max(v1 @ ax)
        min2 = np.min(v2 @ ax)
        max2 = np.max(v2 @ ax)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT disjointness."""
    n = len(inner)
    polys = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def min_distance_between_hexes(poly1, poly2):
    """
    Compute minimum distance between two convex polygons.
    Returns 0 if they overlap or touch.
    """
    axes = []
    for poly in (poly1, poly2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    
    min_dist = float('inf')
    for ax in axes:
        min1 = np.min(poly1 @ ax)
        max1 = np.max(poly1 @ ax)
        min2 = np.min(poly2 @ ax)
        max2 = np.max(poly2 @ ax)
        
        if max1 < min2:
            d = min2 - max1
        elif max2 < min1:
            d = min1 - max2
        else:
            d = 0.0
            
        if d < min_dist:
            min_dist = d
            if min_dist == 0:
                break
    return min_dist

def contact_penalty_edge(inner, target_dist=0.0, max_dist=0.5):
    """
    Penalty for not being in contact (or overlapping).
    We want hexagons to be as close as possible to minimize the hull.
    Overlap is heavily penalized.
    """
    pen = 0.0
    n = len(inner)
    polys = [hex_verts(x, y, 1.0, math.radians(a)) for (x, y, a) in inner]
    for i in range(n):
        for j in range(i + 1, n):
            d = min_distance_between_hexes(polys[i], polys[j])
            if d < 0:
                # Overlap
                pen += 100.0 * d * d
            elif d > target_dist:
                # Not touching, penalize gap
                # Use a soft penalty that encourages contact
                if d < max_dist:
                    pen += 0.1 * (d - target_dist) ** 2
    return pen

def angle_snap_penalty(angles, snap_deg=30.0):
    """
    Encourage angles to be multiples of snap_deg.
    This helps in finding valid discrete topologies.
    """
    pen = 0.0
    for a in angles:
        a_rad = math.radians(a)
        # Distance to nearest multiple of 30 degrees
        # 30 deg = pi/6
        step = math.pi / 6.0
        nearest = round(a_rad / step) * step
        diff = a_rad - nearest
        # Quadratic penalty
        pen += 0.5 * diff * diff
    return pen

def build_axial(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coords."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_composite_pose(local_positions, comp_angle_deg, offset):
    """
    Apply rigid-body pose to local positions.
    """
    cx0 = sum(p[0] for p in local_positions) / len(local_positions)
    cy0 = sum(p[1] for p in local_positions) / len(local_positions)
    ca = math.cos(math.radians(comp_angle_deg))
    sa = math.sin(math.radians(comp_angle_deg))
    result = []
    for (lx, ly, la) in local_positions:
        rx = lx - cx0
        ry = ly - cy0
        nx = cx0 + ca * rx - sa * ry + offset[0]
        ny = cy0 + sa * rx + ca * ry + offset[1]
        na = la + comp_angle_deg
        result.append([nx, ny, na])
    return result

def optimize_rigid(local_positions, deadline, n_starts=8):
    """Multi-start optimization of rigid composite pose."""
    cx0 = sum(p[0] for p in local_positions) / len(local_positions)
    cy0 = sum(p[1] for p in local_positions) / len(local_positions)
    
    best_val = float('inf')
    best_params = None
    
    angle_candidates = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
    
    def objective_rigid_smooth(params):
        comp_angle = params[0]
        dx, dy = params[1], params[2]
        ox, oy = params[3], params[4]
        outer_angle = params[5]
        inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
        return support_function_smooth(inner, (ox, oy), outer_angle, beta=40.0)

    def objective_rigid_exact(params):
        comp_angle = params[0]
        dx, dy = params[1], params[2]
        ox, oy = params[3], params[4]
        outer_angle = params[5]
        inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
        return support_function(inner, (ox, oy), outer_angle)

    for k in range(n_starts):
        if time.time() > deadline - 30:
            break
        init = [
            angle_candidates[k % len(angle_candidates)],
            0.0, 0.0,
            cx0, cy0,
            angle_candidates[k % len(angle_candidates)]
        ]
        try:
            res_s = minimize(objective_rigid_smooth, init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 600})
            res_e = minimize(objective_rigid_exact, res_s.x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            if res_e.fun < best_val:
                best_val = res_e.fun
                best_params = res_e.x.copy()
        except Exception:
            pass
    
    return best_val, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    candidates = []

    # Candidate 1: Center + ring of 6 + 4 more (symmetric cross)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_1, "symmetric_cross"))

    # Candidate 2: Compact 3-4-3 hexagonal cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_2, "cluster343"))

    # Candidate 3: Dense hexagonal packing variant
    axial_3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    candidates.append((axial_3, "dense_variant"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 40:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        local_positions = build_axial(axial_coords)

        cx0 = sum(p[0] for p in local_positions) / len(local_positions)
        cy0 = sum(p[1] for p in local_positions) / len(local_positions)

        # Phase 1: Rigid composite optimization (Reduced budget)
        rigid_val, rigid_params = optimize_rigid(local_positions, deadline, n_starts=4)

        if rigid_params is not None:
            comp_angle = rigid_params[0]
            dx, dy = rigid_params[1], rigid_params[2]
            ox, oy = rigid_params[3], rigid_params[4]
            outer_angle = rigid_params[5]
            inner_rigid = rigid_composite_pose(local_positions, comp_angle, (dx, dy))
            current_side = rigid_val
            current_center = (ox, oy)
            current_angle = outer_angle
        else:
            inner_rigid = local_positions
            current_side = support_function(inner_rigid, (cx0, cy0), 0.0)
            current_center = (cx0, cy0)
            current_angle = 0.0

        # Phase 2: Joint optimization with Simulated Annealing (Reduced budget)
        
        def objective_joint_local(params):
            n = len(local_positions)
            inner = []
            angles = []
            for i in range(n):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner.append([x, y, a])
                angles.append(a)
            
            base = n * 3
            ox = params[base]
            oy = params[base+1]
            outer_angle = params[base+2]
            
            side = support_function(inner, (ox, oy), outer_angle)
            
            # Simplified contact penalty to save time
            pen_contact = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dist_sq = (inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2
                    # Approximate min dist for hexes is ~1.732 (sqrt3) for side 1
                    # If dist < 1.732, they might overlap. 
                    # This is a rough check. For strictness, we rely on final check_disjoint.
                    # We just penalize very close centers to encourage separation in SA.
                    if dist_sq < 2.5: # ~1.58^2
                        pen_contact += (2.5 - dist_sq) * 10.0
            
            # Angle snapping penalty
            pen_angle = 0.0
            for a in angles:
                a_rad = math.radians(a)
                step = math.pi / 6.0
                nearest = round(a_rad / step) * step
                diff = a_rad - nearest
                pen_angle += 0.5 * diff * diff
            
            return side + pen_contact + 0.1 * pen_angle

        initial_params = []
        for x, y, a in inner_rigid:
            initial_params.extend([x, y, a])
        initial_params.extend([current_center[0], current_center[1], current_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint_local(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val

        # Adaptive simulated annealing
        T = 0.5
        T_min = 0.001
        cooling = 0.9995 # Faster cooling

        max_iter = 5000 # Reduced from 15000
        no_improve_count = 0
        patience = 500  # Reduced patience

        for i in range(max_iter):
            if time.time() > deadline - 20:
                break

            neighbor = current_params.copy()

            if np.random.random() < 0.7:
                dim_idx = np.random.randint(0, N*3)
                if dim_idx % 3 == 2:
                    delta = np.random.normal(0, 0.2 * (1 + T))
                else:
                    delta = np.random.normal(0, 0.05 * (1 + T))
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 1.5 * (1 + T))
                else:
                    delta = np.random.normal(0, 0.05 * (1 + T))
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint_local(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > patience:
                    no_improve_count = 0
                    T = max(T * 0.7, T_min * 2)

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
            return support_function(best_inner_joint, (cx, cy), angle_deg)

        polish_starts = []
        base_angle = best_oangle
        for da in [-6, 0, 6]:
            polish_starts.append([best_ocx, best_ocy, base_angle + da])

        best_polish_val = support_function(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
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
        if check_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fall back to rigid layout if disjointness failed
            if rigid_params is not None:
                fallback_side = rigid_val
                fallback_inner = inner_rigid
                fallback_center = (ox, oy)
                fallback_angle = outer_angle
                if fallback_side < best_global_side:
                    best_global_side = fallback_side
                    best_global_inner = fallback_inner
                    best_global_center = fallback_center
                    best_global_angle = fallback_angle

    if best_global_inner is None:
        axial_coords = candidates[0][0][:N]
        centers = [(1.5 * q, SQRT3 * (r + q / 2.0)) for q, r in axial_coords]
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = support_function(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
