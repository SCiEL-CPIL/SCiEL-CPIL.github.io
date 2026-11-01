"""Rewritten hexagon packing with multi-start topologies and gradient-based optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_verts(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a 6x2 numpy array."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def container_normals(angle_deg):
    """Return 6 outward normals of the outer hexagon (unit vectors)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    base = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        bx, by = math.cos(theta), math.sin(theta)
        base.append((bx * ca - by * sa, bx * sa + by * ca))
    return np.array(base)

def support_function(inner, outer_center, outer_angle_deg, beta=None):
    """
    Compute max projection of all inner hexagon vertices onto container normals.
    If beta is given, return smooth log-sum-exp approximation.
    inner: list of (x, y, angle_deg)
    Returns: max_projection (or smooth max)
    """
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    
    all_projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_verts(ix, iy, 1.0, math.radians(iangle))
        rel = verts - np.array([ox, oy])
        projs = rel @ normals.T  # (6, 6)
        all_projs.append(projs.flatten())
    
    all_projs = np.concatenate(all_projs)
    
    if beta is None:
        return float(np.max(all_projs))
    else:
        m = np.max(all_projs)
        if m < -1000:
            return 0.0
        scaled = beta * (all_projs - m)
        # Numerically stable log-sum-exp
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        smooth_max = m + log_sum_exp / beta
        return float(smooth_max)

def side_from_projection(max_proj):
    """Convert max projection to outer hexagon side length."""
    return 2.0 * max_proj / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT check for two convex polygons (lists of (x,y))."""
    axes = []
    for verts in (v1, v2):
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            nx, ny = -ey, ex
            n = math.sqrt(nx * nx + ny * ny)
            if n > 1e-12:
                axes.append((nx / n, ny / n))
    
    for ax, ay in axes:
        min1 = min(p[0] * ax + p[1] * ay for p in v1)
        max1 = max(p[0] * ax + p[1] * ay for p in v1)
        min2 = min(p[0] * ax + p[1] * ay for p in v2)
        max2 = max(p[0] * ax + p[1] * ay for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner)
    verts_list = [get_hex_verts(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2])) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v1 = [(v[0], v[1]) for v in verts_list[i]]
            v2 = [(v[0], v[1]) for v in verts_list[j]]
            if sat_overlap(v1, v2, eps):
                return False
    return True

def contact_penalty(inner, min_dist=SQRT3 - 0.003):
    """Differentiable penalty for center distances below min_dist."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if d < min_dist:
                diff = min_dist - d
                penalty += diff * diff
    return penalty

def build_topologies():
    """Generate diverse initial topologies for N=11 unit hexagons."""
    topologies = []
    
    # Topology 1: Standard hexagonal cluster (center + ring + partial second ring)
    # Axial coordinates for compact hexagonal packing
    t1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
           (2, -1), (1, 1), (-2, 1), (-1, -1)]
    topologies.append(t1)
    
    # Topology 2: Linear chain with staggered arrangement
    t2 = []
    for i in range(11):
        q = i
        r = 0 if i % 2 == 0 else -1
        t2.append((q, r))
    topologies.append(t2)
    
    # Topology 3: 4-3-4 triangular arrangement
    t3 = []
    # Row 0: 4 hexagons
    for i in range(4):
        t3.append((i, 0))
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        t3.append((i + 1, -1))
    # Row 2: 4 hexagons
    for i in range(4):
        t3.append((i, -2))
    topologies.append(t3)
    
    # Topology 4: Flower with center + 6 + 4 outer
    t4 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
           (2, 0), (1, 2), (-2, 2), (-2, 0)]
    topologies.append(t4)
    
    # Topology 5: Two rows of 5+6
    t5 = []
    for i in range(5):
        t5.append((i, 0))
    for i in range(6):
        t5.append((i, -1))
    topologies.append(t5)
    
    # Topology 6: Compact with gaps (cross pattern)
    t6 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
           (2, -1), (2, 0), (0, 2), (-2, 1)]
    topologies.append(t6)
    
    # Topology 7: Diagonal band
    t7 = [(i, -i // 2) for i in range(11)]
    topologies.append(t7)
    
    # Topology 8: 3-5-3 pyramid
    t8 = []
    for i in range(3):
        t8.append((i, 0))
    for i in range(5):
        t8.append((i - 1, -1))
    for i in range(3):
        t8.append((i, -2))
    topologies.append(t8)
    
    return topologies

def axial_to_xy(axial_coords, scale=1.001):
    """Convert axial coordinates to Cartesian with slight expansion to avoid touching."""
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    return centers

def optimize_topology(topology, deadline, init_angles=None):
    """
    Optimize a given topology. Returns (inner, outer_center, side, outer_angle).
    """
    if init_angles is None:
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    centers = axial_to_xy(topology)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_oc = (0.0, 0.0)
    best_oa = 0.0
    
    # Phase 1: Optimize rigid body (global translation + rotation) with smooth objective
    def obj_rigid_smooth(params):
        tx, ty, ga = params
        # Apply global rotation and translation to inner hexagons
        ga_rad = math.radians(ga)
        cg, sg = math.cos(ga_rad), math.sin(ga_rad)
        rotated_inner = []
        for (x, y, a) in initial_inner:
            rx = cg * x - sg * y
            ry = sg * x + cg * y
            rotated_inner.append((rx + tx, ry + ty, a + ga))
        max_p = support_function(rotated_inner, (0.0, 0.0), ga, beta=60.0)
        return side_from_projection(max_p)
    
    def obj_rigid_exact(params):
        tx, ty, ga = params
        ga_rad = math.radians(ga)
        cg, sg = math.cos(ga_rad), math.sin(ga_rad)
        rotated_inner = []
        for (x, y, a) in initial_inner:
            rx = cg * x - sg * y
            ry = sg * x + cg * y
            rotated_inner.append((rx + tx, ry + ty, a + ga))
        max_p = support_function(rotated_inner, (0.0, 0.0), ga)
        return side_from_projection(max_p)
    
    best_rigid = None
    best_rigid_val = float('inf')
    
    for ia in init_angles:
        if time.time() > deadline - 60:
            break
        x0 = [0.0, 0.0, float(ia)]
        try:
            # Smooth L-BFGS-B
            res_s = minimize(obj_rigid_smooth, x0, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            # Exact Nelder-Mead polish
            res_e = minimize(obj_rigid_exact, res_s.x, method='Nelder-Mead',
                            options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 1500})
            if res_e.fun < best_rigid_val:
                best_rigid_val = res_e.fun
                best_rigid = res_e.x
        except Exception:
            try:
                res_e = minimize(obj_rigid_exact, x0, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
                if res_e.fun < best_rigid_val:
                    best_rigid_val = res_e.fun
                    best_rigid = res_e.x
            except Exception:
                pass
    
    if best_rigid is not None:
        tx, ty, ga = best_rigid
        ga_rad = math.radians(ga)
        cg, sg = math.cos(ga_rad), math.sin(ga_rad)
        standard_inner = []
        for (x, y, a) in initial_inner:
            rx = cg * x - sg * y
            ry = sg * x + cg * y
            standard_inner.append([rx + tx, ry + ty, a + ga])
        best_inner = standard_inner
        best_oc = (0.0, 0.0)
        best_oa = ga
        best_side = best_rigid_val
    else:
        # Fallback
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_inner = initial_inner
        best_oc = (cx_avg, cy_avg)
        best_oa = 0.0
        best_side = side_from_projection(support_function(initial_inner, best_oc, 0.0))
    
    # Phase 2: Joint optimization of inner positions + outer container
    # Parameters: x0,y0,a0, ..., x10,y10,a10, cx, cy, ca
    # Total: 3*N + 3 = 36 parameters
    
    def obj_joint_smooth(params):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append((x, y, a))
        cx = params[N * 3]
        cy = params[N * 3 + 1]
        ca = params[N * 3 + 2]
        
        max_p = support_function(inner_list, (cx, cy), ca, beta=50.0)
        side = side_from_projection(max_p)
        pen = contact_penalty(inner_list, min_dist=SQRT3 - 0.003)
        return side + 200.0 * pen
    
    def obj_joint_exact(params):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append((x, y, a))
        cx = params[N * 3]
        cy = params[N * 3 + 1]
        ca = params[N * 3 + 2]
        
        max_p = support_function(inner_list, (cx, cy), ca)
        side = side_from_projection(max_p)
        pen = contact_penalty(inner_list, min_dist=SQRT3 - 0.003)
        return side + 200.0 * pen
    
    # Initialize joint params from Phase 1 result
    joint_params = []
    for (x, y, a) in best_inner:
        joint_params.extend([x, y, a])
    joint_params.extend([best_oc[0], best_oc[1], best_oa])
    
    current_params = np.array(joint_params, dtype=float)
    current_val = obj_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val = current_val
    
    # Simulated annealing with adaptive step size
    T = 0.2
    T_min = 0.0005
    cooling = 0.9997
    max_iter = 20000
    stagnant = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        neighbor = current_params.copy()
        step_scale = 0.15 * (T / T_min) ** 0.25
        
        r = np.random.random()
        if r < 0.6:
            # Move one hexagon
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim < 2:
                delta = np.random.normal(0, 0.08 * step_scale)
            else:
                delta = np.random.normal(0, 1.0 * step_scale)
            neighbor[idx * 3 + dim] += delta
        elif r < 0.8:
            # Move two hexagons together (rigid pair)
            i1, i2 = np.random.permutation(N)[:2]
            dx = np.random.normal(0, 0.06 * step_scale)
            dy = np.random.normal(0, 0.06 * step_scale)
            neighbor[i1 * 3] += dx
            neighbor[i1 * 3 + 1] += dy
            neighbor[i2 * 3] += dx
            neighbor[i2 * 3 + 1] += dy
        elif r < 0.9:
            # Move outer container
            dim = np.random.randint(N * 3, N * 3 + 3)
            if dim == N * 3 + 2:
                delta = np.random.normal(0, 1.0 * step_scale)
            else:
                delta = np.random.normal(0, 0.06 * step_scale)
            neighbor[dim] += delta
        else:
            # Global rotation of all inner hexagons
            da = np.random.normal(0, 0.5 * step_scale)
            for i in range(N):
                neighbor[i * 3 + 2] += da
            neighbor[N * 3 + 2] += da
        
        neighbor_val = obj_joint_smooth(neighbor)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best_params = current_params.copy()
                best_val = current_val
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1
        
        T *= cooling
        if T < T_min:
            break
        
        # Occasional L-BFGS-B refinement
        if stagnant > 500 and it % 2000 == 1000:
            try:
                res = minimize(obj_joint_smooth, current_params, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6})
                if res.fun < best_val:
                    best_params = res.x
                    best_val = res.fun
                    current_params = res.x
                    current_val = res.fun
                    stagnant = 0
            except Exception:
                pass
    
    # Extract best joint solution
    best_inner_joint = []
    for i in range(N):
        best_inner_joint.append([best_params[i * 3], best_params[i * 3 + 1], best_params[i * 3 + 2]])
    best_oc_j = (best_params[N * 3], best_params[N * 3 + 1])
    best_ao_j = best_params[N * 3 + 2]
    
    # Phase 3: Final polish - optimize outer container with fixed inner positions
    def obj_outer_smooth(params):
        cx, cy, ca = params
        max_p = support_function(best_inner_joint, (cx, cy), ca, beta=80.0)
        return side_from_projection(max_p)
    
    def obj_outer_exact(params):
        cx, cy, ca = params
        max_p = support_function(best_inner_joint, (cx, cy), ca)
        return side_from_projection(max_p)
    
    best_outer_val = float('inf')
    best_outer_params = None
    
    polish_starts = [
        [best_oc_j[0], best_oc_j[1], best_ao_j],
        [best_oc_j[0], best_oc_j[1], best_ao_j + 5],
        [best_oc_j[0], best_oc_j[1], best_ao_j - 5],
        [best_oc_j[0] + 0.05, best_oc_j[1], best_ao_j],
        [best_oc_j[0] - 0.05, best_oc_j[1], best_ao_j],
        [best_oc_j[0], best_oc_j[1] + 0.05, best_ao_j],
        [best_oc_j[0], best_oc_j[1] - 0.05, best_ao_j],
        [best_oc_j[0] + 0.03, best_oc_j[1] + 0.03, best_ao_j],
        [best_oc_j[0] - 0.03, best_oc_j[1] - 0.03, best_ao_j],
    ]
    
    for ps in polish_starts:
        if time.time() > deadline - 5:
            break
        try:
            res_s = minimize(obj_outer_smooth, ps, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            res_e = minimize(obj_outer_exact, res_s.x, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 5000})
            if res_e.fun < best_outer_val:
                best_outer_val = res_e.fun
                best_outer_params = res_e.x
        except Exception:
            try:
                res_e = minimize(obj_outer_exact, ps, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 4000})
                if res_e.fun < best_outer_val:
                    best_outer_val = res_e.fun
                    best_outer_params = res_e.x
            except Exception:
                pass
    
    if best_outer_params is not None:
        best_oc = (best_outer_params[0], best_outer_params[1])
        best_oa = best_outer_params[2]
        best_side = best_outer_val
        best_inner = best_inner_joint
    
    # Final disjointness check
    if not check_disjoint(best_inner, eps=1e-5):
        # Try to fix by slightly expanding
        for expand in [1.001, 1.002, 1.005, 1.01]:
            adjusted = []
            cx_avg = sum(p[0] for p in best_inner) / N
            cy_avg = sum(p[1] for p in best_inner) / N
            for (x, y, a) in best_inner:
                dx = x - cx_avg
                dy = y - cy_avg
                adjusted.append([cx_avg + expand * dx, cy_avg + expand * dy, a])
            if check_disjoint(adjusted, eps=1e-5):
                best_inner = adjusted
                best_side = side_from_projection(support_function(best_inner, best_oc, best_oa))
                break
    
    return best_inner, best_oc, best_side, best_ao_j if best_outer_params is None else best_oa

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    topologies = build_topologies()
    
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_oc = (0.0, 0.0)
    best_overall_oa = 0.0
    
    # Allocate time per topology
    n_tops = len(topologies)
    time_per_top = min(20.0, (deadline - start_time - 10.0) / n_tops)
    
    for ti, top in enumerate(topologies):
        if time.time() > deadline - 15:
            break
        
        top_deadline = min(deadline - 15.0, start_time + time_per_top * (ti + 1))
        
        try:
            inner, oc, side, oa = optimize_topology(top, top_deadline)
            
            # Verify and compute exact side
            if inner is not None and len(inner) == N:
                exact_side = side_from_projection(support_function(inner, oc, oa))
                if check_disjoint(inner, eps=1e-5):
                    if exact_side < best_overall_side:
                        best_overall_side = exact_side
                        best_overall_inner = inner
                        best_overall_oc = oc
                        best_overall_oa = oa
        except Exception:
            continue
    
    # Final global polish on the best solution
    if best_overall_inner is not None and time.time() < deadline - 10:
        def obj_final_smooth(params):
            inner_list = []
            for i in range(N):
                inner_list.append((params[i * 3], params[i * 3 + 1], params[i * 3 + 2]))
            cx = params[N * 3]
            cy = params[N * 3 + 1]
            ca = params[N * 3 + 2]
            max_p = support_function(inner_list, (cx, cy), ca, beta=100.0)
            side = side_from_projection(max_p)
            pen = contact_penalty(inner_list, min_dist=SQRT3 - 0.002)
            return side + 100.0 * pen
        
        final_params = []
        for (x, y, a) in best_overall_inner:
            final_params.extend([x, y, a])
        final_params.extend([best_overall_oc[0], best_overall_oc[1], best_overall_oa])
        
        try:
            res = minimize(obj_final_smooth, np.array(final_params), method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9})
            final_inner = []
            for i in range(N):
                final_inner.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            final_oc = (res.x[N * 3], res.x[N * 3 + 1])
            final_ao = res.x[N * 3 + 2]
            final_side = side_from_projection(support_function(final_inner, final_oc, final_ao))
            
            if check_disjoint(final_inner, eps=1e-5) and final_side < best_overall_side:
                best_overall_inner = final_inner
                best_overall_oc = final_oc
                best_overall_oa = final_ao
                best_overall_side = final_side
        except Exception:
            pass
    
    # Ensure we have a valid result
    if best_overall_inner is None:
        # Absolute fallback: simple hexagonal cluster
        centers = axial_to_xy([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                               (2, -1), (1, 1), (-2, 1), (-1, -1)])
        best_overall_inner = [[x, y, 0.0] for x, y in centers]
        best_overall_oc = (0.0, 0.0)
        best_overall_ao = 0.0
        best_overall_side = side_from_projection(support_function(best_overall_inner, best_overall_oc, best_overall_ao))
    
    return best_overall_inner, best_overall_oc, best_overall_side, best_overall_oa

# EVOLVE-BLOCK-END
