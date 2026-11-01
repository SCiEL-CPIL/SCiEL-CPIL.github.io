"""Combined: rigid-composite support-function objective + contact-graph penalty + multi-start."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, side, angle_rad):
    """6 vertices of a regular hexagon."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def hex_verts_local(side=1.0, angle_rad=0.0):
    """6 vertices of unit hexagon at origin."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([side * np.cos(thetas), side * np.sin(thetas)])

def container_normals(outer_angle_deg):
    """6 outward normals of the container hexagon."""
    base = np.radians(30)
    a = math.radians(outer_angle_deg)
    cs, sn = math.cos(a), math.sin(a)
    normals = []
    for i in range(6):
        t = base + i * math.pi / 3.0
        nx = math.cos(t)
        ny = math.sin(t)
        # Rotate normal by outer angle
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
    normals = container_normals(outer_angle_deg)
    max_proj = 0.0
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                p = dx * nx + dy * ny
                if p > max_proj:
                    max_proj = p
    return 2.0 * max_proj / SQRT3

def support_function_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    """Smooth log-sum-exp version."""
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = container_normals(outer_angle_deg)
    projs = []
    for (ix, iy, iang) in inner:
        verts = hex_verts(ix, iy, 1.0, math.radians(iang))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            for (nx, ny) in normals:
                projs.append(dx * nx + dy * ny)
    if not projs:
        return 0.0
    P = np.array(projs)
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

def contact_penalty(inner, min_dist=SQRT3 - 0.003):
    """Quadratic penalty for center-distance violations."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_axial(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coords."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_composite_pose(local_positions, comp_angle_deg, offset, outer_center, outer_angle_deg):
    """
    Apply rigid-body pose to local positions.
    local_positions: list of (x, y, angle_deg) in composite frame
    comp_angle_deg: rotation of composite about its centroid
    offset: (dx, dy) translation
    Returns absolute positions.
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

def objective_rigid_smooth(params, local_positions, beta=50.0):
    """
    params: [comp_angle, dx, dy, ox, oy, outer_angle]
    """
    comp_angle = params[0]
    dx, dy = params[1], params[2]
    ox, oy = params[3], params[4]
    outer_angle = params[5]
    inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy), (ox, oy), outer_angle)
    return support_function_smooth(inner, (ox, oy), outer_angle, beta=beta)

def objective_rigid_exact(params, local_positions):
    comp_angle = params[0]
    dx, dy = params[1], params[2]
    ox, oy = params[3], params[4]
    outer_angle = params[5]
    inner = rigid_composite_pose(local_positions, comp_angle, (dx, dy), (ox, oy), outer_angle)
    return support_function(inner, (ox, oy), outer_angle)

def optimize_rigid(local_positions, deadline, n_starts=8):
    """Multi-start optimization of rigid composite pose."""
    cx0 = sum(p[0] for p in local_positions) / len(local_positions)
    cy0 = sum(p[1] for p in local_positions) / len(local_positions)
    
    best_val = float('inf')
    best_params = None
    
    # Initial angle candidates
    angle_candidates = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
    
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
            res_s = minimize(lambda p: objective_rigid_smooth(p, local_positions, beta=40.0),
                           init, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 600})
            res_e = minimize(lambda p: objective_rigid_exact(p, local_positions),
                           res_s.x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
            if res_e.fun < best_val:
                best_val = res_e.fun
                best_params = res_e.x.copy()
        except Exception:
            pass
    
    return best_val, best_params

def objective_joint(params, local_positions, min_dist=SQRT3 - 0.003):
    """
    params: [x0,y0,a0, x1,y1,a1, ..., xN,yN,aN, comp_angle, dx, dy, ox, oy, outer_angle]
    """
    n = len(local_positions)
    inner = []
    for i in range(n):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        inner.append([x, y, a])
    
    base = n * 3
    comp_angle = params[base]
    dx, dy = params[base+1], params[base+2]
    ox, oy = params[base+3], params[base+4]
    outer_angle = params[base+5]
    
    # Apply rigid pose
    cx0 = sum(p[0] for p in inner) / n
    cy0 = sum(p[1] for p in inner) / n
    ca = math.cos(math.radians(comp_angle))
    sa = math.sin(math.radians(comp_angle))
    
    posed = []
    for (lx, ly, la) in inner:
        rx = lx - cx0
        ry = ly - cy0
        nx = cx0 + ca * rx - sa * ry + dx
        ny = cy0 + sa * rx + ca * ry + dy
        na = la + comp_angle
        posed.append([nx, ny, na])
    
    side = support_function(posed, (ox, oy), outer_angle)
    pen = contact_penalty(posed, min_dist=min_dist)
    return side + 500.0 * pen

def anneal_joint(params0, local_positions, deadline, max_iter=8000):
    """Simulated annealing on joint parameters."""
    n = len(local_positions)
    current = np.array(params0, dtype=float)
    current_val = objective_joint(current, local_positions)
    best = current.copy()
    best_val = current_val
    
    T = 0.2
    T_min = 0.0005
    cooling = 0.9997
    
    for i in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        neighbor = current.copy()
        step = 0.15 * (T / T_min) ** 0.25
        
        r = np.random.random()
        if r < 0.6:
            idx = np.random.randint(0, n)
            dim = np.random.randint(0, 3)
            scale = 0.08 if dim < 2 else 1.2
            neighbor[idx*3 + dim] += np.random.normal(0, scale * step)
        elif r < 0.8:
            idx1 = np.random.randint(0, n)
            idx2 = np.random.randint(0, n)
            if idx1 != idx2:
                ddx = np.random.normal(0, 0.06 * step)
                ddy = np.random.normal(0, 0.06 * step)
                neighbor[idx1*3] += ddx
                neighbor[idx1*3+1] += ddy
                neighbor[idx2*3] += ddx
                neighbor[idx2*3+1] += ddy
        else:
            base = n * 3
            dim = np.random.randint(base, base + 6)
            scale = 1.2 if dim in (base, base+5) else 0.06
            neighbor[dim] += np.random.normal(0, scale * step)
        
        neighbor_val = objective_joint(neighbor, local_positions)
        
        if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
            current = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
        
        T *= cooling
        if T < T_min:
            break
    
    return best, best_val

def polish_outer(inner, center0, angle0, deadline):
    """Final polish of outer container."""
    def obj(p):
        return support_function(inner, (p[0], p[1]), p[2])
    
    best_val = support_function(inner, center0, angle0)
    best_p = np.array([center0[0], center0[1], angle0], dtype=float)
    
    starts = [
        best_p.copy(),
        best_p + np.array([0.03, 0, 1.0]),
        best_p + np.array([-0.03, 0, -1.0]),
        best_p + np.array([0, 0.03, 2.0]),
        best_p + np.array([0, -0.03, -2.0]),
        best_p + np.array([0.02, 0.02, 0.5]),
        best_p + np.array([-0.02, -0.02, -0.5]),
    ]
    
    for s in starts:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(obj, s, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 6000})
            if res.fun < best_val:
                best_val = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    
    return best_p, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Multiple candidate cluster shapes (axial coordinates)
    # Shape 1: center + 6 neighbors + 4 more (classic 11-hex cluster)
    shape1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (1,1), (-2,1), (-1,-1)]
    # Shape 2: more elongated
    shape2 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (-1,0), (-1,1), (0,-1), (1,-1)]
    # Shape 3: compact with different arrangement
    shape3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (-2,0), (-2,1)]
    
    shapes = [shape1, shape2, shape3]
    
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_center = (0.0, 0.0)
    best_overall_angle = 0.0
    
    for shape in shapes:
        if time.time() > deadline - 40:
            break
        if len(shape) > N:
            shape = shape[:N]
        
        local = build_axial(shape)
        
        # Phase 1: Rigid composite optimization
        rigid_val, rigid_params = optimize_rigid(local, deadline, n_starts=6)
        
        if rigid_params is not None:
            comp_angle, dx, dy, ox, oy, outer_angle = rigid_params
            inner_rigid = rigid_composite_pose(local, comp_angle, (dx, dy), (ox, oy), outer_angle)
            
            if check_disjoint(inner_rigid, eps=1e-5):
                side = support_function(inner_rigid, (ox, oy), outer_angle)
                if side < best_overall_side:
                    best_overall_side = side
                    best_overall_inner = inner_rigid
                    best_overall_center = (ox, oy)
                    best_overall_angle = outer_angle
        
        # Phase 2: Joint optimization with annealing
        if time.time() > deadline - 25:
            continue
        
        cx0 = sum(p[0] for p in local) / len(local)
        cy0 = sum(p[1] for p in local) / len(local)
        
        # Initialize joint params from rigid solution or identity
        if rigid_params is not None:
            init_comp_angle, init_dx, init_dy, init_ox, init_oy, init_outer = rigid_params
        else:
            init_comp_angle, init_dx, init_dy = 0.0, 0.0, 0.0
            init_ox, init_oy, init_outer = cx0, cy0, 0.0
        
        joint_params0 = []
        for (x, y, a) in local:
            joint_params0.extend([x, y, a])
        joint_params0.extend([init_comp_angle, init_dx, init_dy, init_ox, init_oy, init_outer])
        
        annealed_params, annealed_val = anneal_joint(joint_params0, local, deadline, max_iter=5000)
        
        # Extract inner from annealed params
        n = len(local)
        inner_joint = []
        for i in range(n):
            x = annealed_params[i*3]
            y = annealed_params[i*3+1]
            a = annealed_params[i*3+2]
            inner_joint.append([x, y, a])
        
        base = n * 3
        j_comp_angle = annealed_params[base]
        j_dx, j_dy = annealed_params[base+1], annealed_params[base+2]
        j_ox, j_oy = annealed_params[base+3], annealed_params[base+4]
        j_outer = annealed_params[base+5]
        
        # Apply rigid pose to get absolute positions
        cx_j = sum(p[0] for p in inner_joint) / n
        cy_j = sum(p[1] for p in inner_joint) / n
        ca = math.cos(math.radians(j_comp_angle))
        sa = math.sin(math.radians(j_comp_angle))
        
        posed = []
        for (lx, ly, la) in inner_joint:
            rx = lx - cx_j
            ry = ly - cy_j
            nx = cx_j + ca * rx - sa * ry + j_dx
            ny = cy_j + sa * rx + ca * ry + j_dy
            na = la + j_comp_angle
            posed.append([nx, ny, na])
        
        if check_disjoint(posed, eps=1e-5):
            side = support_function(posed, (j_ox, j_oy), j_outer)
            if side < best_overall_side:
                best_overall_side = side
                best_overall_inner = posed
                best_overall_center = (j_ox, j_oy)
                best_overall_angle = j_outer
    
    # Phase 3: Final polish
    if best_overall_inner is not None:
        if time.time() < deadline - 5:
            polished, polished_val = polish_outer(
                best_overall_inner, best_overall_center, best_overall_angle, deadline)
            best_overall_center = (polished[0], polished[1])
            best_overall_angle = polished[2]
            best_overall_side = polished_val
    
    # Fallback: simple lattice if nothing worked
    if best_overall_inner is None:
        local = build_axial(shape1)
        cx0 = sum(p[0] for p in local) / len(local)
        cy0 = sum(p[1] for p in local) / len(local)
        best_overall_inner = local
        best_overall_center = (cx0, cy0)
        best_overall_angle = 0.0
        best_overall_side = support_function(local, best_overall_center, best_overall_angle)
    
    return best_overall_inner, best_overall_center, best_overall_side, best_overall_angle

# EVOLVE-BLOCK-END
