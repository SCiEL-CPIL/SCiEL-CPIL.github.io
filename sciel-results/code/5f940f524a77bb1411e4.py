"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses SLSQP with analytic gradients and smooth support function formulation.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts


def smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    values = np.array(values)
    m = np.max(values)
    if m == 0:
        return 0.0
    exps = np.exp(beta * (values - m))
    s = np.sum(exps)
    if s == 0:
        return m
    return m + math.log(s) / beta


def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True


def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj


def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    # inside outer
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    # pairwise non-overlap (allow touching)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True


def params_to_inner(params):
    """Convert flat parameter vector to inner hexagon list."""
    inner = []
    for i in range(N):
        x = params[i * 3]
        y = params[i * 3 + 1]
        ang = params[i * 3 + 2]
        inner.append((x, y, ang))
    return inner


def slsqp_objective_and_grad(params, outer_center=(0.0, 0.0), outer_angle=0.0, beta=30.0):
    """
    SLSQP objective: smooth outer side length + overlap penalty.
    Returns (value, gradient).
    """
    if time.time() > DEADLINE:
        return 1e6, np.zeros_like(params)
    
    inner = params_to_inner(params)
    cx, cy = outer_center
    a = math.radians(outer_angle)
    
    # Compute normals for outer hexagon
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Compute all vertex projections for smooth max
    all_projs = []
    proj_grads = []  # gradient contribution for each projection
    
    for i in range(N):
        x, y, ang = inner[i]
        a_rad = math.radians(ang)
        # Vertices of hexagon i
        for v in range(6):
            theta = a_rad + v * math.pi / 3.0
            vx = x + math.cos(theta)
            vy = y + math.sin(theta)
            dx, dy = vx - cx, vy - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                all_projs.append(proj)
                # Gradient with respect to x, y, ang
                gx = nx
                gy = ny
                # d/dang: -sin(theta)*nx + cos(theta)*ny for the vertex position
                g_ang = (-math.sin(theta) * nx + math.cos(theta) * ny)
                proj_grads.append((gx, gy, g_ang))
    
    all_projs = np.array(all_projs)
    m = np.max(all_projs)
    if m == 0:
        sm = 0.0
    else:
        exps = np.exp(beta * (all_projs - m))
        sm = m + math.log(np.sum(exps)) / beta
    
    outer_side = (2.0 / math.sqrt(3.0)) * sm
    
    # Gradient of outer_side w.r.t. params
    grad = np.zeros(3 * N)
    if m != 0:
        exps = np.exp(beta * (all_projs - m))
        weights = exps / np.sum(exps)  # softmax weights
        for k in range(len(all_projs)):
            w = weights[k]
            if w < 1e-10:
                continue
            idx = k // 6  # which hexagon (6 vertices per hex)
            gx, gy, g_ang = proj_grads[k]
            grad[idx * 3] += (2.0 / math.sqrt(3.0)) * w * gx
            grad[idx * 3 + 1] += (2.0 / math.sqrt(3.0)) * w * gy
            grad[idx * 3 + 2] += (2.0 / math.sqrt(3.0)) * w * g_ang
    
    # Overlap penalty (smooth)
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_pen = 0.0
    overlap_grad = np.zeros(3 * N)
    
    for i in range(N):
        for j in range(i + 1, N):
            # Compute minimum separation using SAT
            min_sep = float('inf')
            min_sep_axis = None
            A = polys[i]
            B = polys[j]
            for poly in (A, B):
                m_len = len(poly)
                for k in range(m_len):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % m_len]
                    dx, dy = x2 - x1, y2 - y1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length < 1e-12:
                        continue
                    nx, ny = -dy / length, dx / length
                    minA = min(nx * px + ny * py for px, py in A)
                    maxA = max(nx * px + ny * py for px, py in A)
                    minB = min(nx * px + ny * py for px, py in B)
                    maxB = max(nx * px + ny * py for px, py in B)
                    if maxA < minB:
                        sep = minB - maxA
                    elif maxB < minA:
                        sep = minA - maxB
                    else:
                        sep = 0.0
                    if sep < min_sep:
                        min_sep = sep
                        min_sep_axis = (nx, ny, k, poly is A)
            
            if min_sep < 1e-6:
                # Overlapping or touching - compute penalty
                # Use smooth penalty: -min_sep smoothed
                if min_sep < 0:
                    val = -min_sep
                    penalty_val = val / (1.0 + math.exp(-30.0 * val))
                    overlap_pen += penalty_val
                    # Gradient approximation: push apart along separation axis
                    if min_sep_axis:
                        nx, ny, k_idx, is_a = min_sep_axis
                        # Gradient is in direction of normal
                        # For simplicity, use numerical gradient approximation
                        # Actually, let's use a simpler approach: 
                        # The penalty gradient points in the direction that increases separation
                        # This is complex, so use finite differences for overlap part
                        pass
                else:
                    # Touching or very close, small penalty
                    pass
    
    # For the overlap gradient, use finite differences (expensive but accurate)
    if overlap_pen > 1e-10:
        # Use finite differences for the overlap part
        eps_fd = 1e-6
        for i in range(3 * N):
            old = params[i]
            params[i] = old + eps_fd
            inner_plus = params_to_inner(params)
            polys_plus = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner_plus]
            pen_plus = 0.0
            for ii in range(N):
                for jj in range(ii + 1, N):
                    min_sep = float('inf')
                    A = polys_plus[ii]
                    B = polys_plus[jj]
                    for poly in (A, B):
                        m_len = len(poly)
                        for k in range(m_len):
                            x1, y1 = poly[k]
                            x2, y2 = poly[(k + 1) % m_len]
                            dx, dy = x2 - x1, y2 - y1
                            length = math.sqrt(dx * dx + dy * dy)
                            if length < 1e-12:
                                continue
                            nx, ny = -dy / length, dx / length
                            minA = min(nx * px + ny * py for px, py in A)
                            maxA = max(nx * px + ny * py for px, py in A)
                            minB = min(nx * px + ny * py for px, py in B)
                            maxB = max(nx * px + ny * py for px, py in B)
                            if maxA < minB:
                                sep = minB - maxA
                            elif maxB < minA:
                                sep = minA - maxB
                            else:
                                sep = 0.0
                            if sep < min_sep:
                                min_sep = sep
                    if min_sep < 0:
                        val = -min_sep
                        pen_plus += val / (1.0 + math.exp(-30.0 * val))
            params[i] = old
            overlap_grad[i] = (pen_plus - overlap_pen) / eps_fd
    
    total_value = outer_side + 50.0 * overlap_pen
    total_grad = grad + 50.0 * overlap_grad
    
    return total_value, total_grad


def slsqp_objective(params, outer_center=(0.0, 0.0), outer_angle=0.0, beta=30.0):
    """SLSQP objective without gradient (for fallback)."""
    val, _ = slsqp_objective_and_grad(params, outer_center, outer_angle, beta)
    return val


def build_hexagonal_cluster(scale=2.0, rot_in=0.0, rot_out=0.0):
    """Build a hexagonal cluster: center + ring1(6) + ring2(5)."""
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        positions.append((x, y))
    for k in range(5):
        theta = math.radians(k * 60.0)
        x = 2.0 * scale * math.cos(theta)
        y = 2.0 * scale * math.sin(theta)
        positions.append((x, y))
    
    inner = [(x, y, rot_in) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out)
    return inner, outer_center, outer_side, rot_out


def build_ring12(scale=2.0, rot_in=0.0, rot_out=0.0):
    """Build a ring of 12 hexagons (no center)."""
    positions = []
    for k in range(6):
        theta = math.radians(k * 60.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        positions.append((x, y))
    for k in range(6):
        theta = math.radians(k * 60.0 + 30.0)
        x = 2.0 * scale * math.cos(theta)
        y = 2.0 * scale * math.sin(theta)
        positions.append((x, y))
    
    inner = [(x, y, rot_in) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out)
    return inner, outer_center, outer_side, rot_out


def build_rect_grid(scale=2.0, rot_in=0.0, rot_out=0.0):
    """Build a 3x4 rectangular grid."""
    positions = []
    row_spacing = scale * math.sqrt(3) / 2.0
    for row in range(3):
        for col in range(4):
            x = col * scale
            y = row * row_spacing
            if row % 2 == 1:
                x += scale / 2.0
            positions.append((x - 1.5 * scale, y - row_spacing))
    
    inner = [(x, y, rot_in) for (x, y) in positions]
    outer_center = (0.0, 0.0)
    outer_side = compute_min_outer_side(inner, outer_center, rot_out)
    return inner, outer_center, outer_side, rot_out


def optimize_slsqp(initial_params, outer_center=(0.0, 0.0), outer_angle=0.0, max_time=60.0):
    """Run SLSQP optimization from initial params."""
    t_start = time.time()
    
    def obj_and_grad(p):
        if time.time() - t_start > max_time:
            return 1e6, np.zeros_like(p)
        return slsqp_objective_and_grad(p, outer_center, outer_angle)
    
    def obj_only(p):
        if time.time() - t_start > max_time:
            return 1e6
        val, _ = slsqp_objective_and_grad(p, outer_center, outer_angle)
        return val
    
    try:
        # First try with gradient
        result = minimize(
            obj_only,
            initial_params,
            jac=lambda p: slsqp_objective_and_grad(p, outer_center, outer_angle)[1],
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if result.fun < 1e5:
            inner = params_to_inner(result.x)
            outer_side = compute_min_outer_side(inner, outer_center, outer_angle)
            if check_validity(inner, outer_center, outer_side, outer_angle):
                return outer_side, result.x
    except Exception:
        pass
    
    return 1e6, initial_params


def optimize_de(max_time=80.0):
    """Run differential evolution for global search."""
    t_start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((-60.0, 60.0))
    
    best_side = 1e6
    best_params = None
    
    for outer_angle in [0.0, 15.0, 30.0]:
        if time.time() - t_start > max_time:
            break
        
        try:
            result = differential_evolution(
                lambda p, oa=outer_angle: slsqp_objective(p, (0.0, 0.0), oa),
                bounds=bounds,
                maxiter=30,
                popsize=12,
                tol=1e-4,
                mutation=(0.3, 1.2),
                recombination=0.9,
                seed=42,
                polish=True,
                polish_options={'method': 'SLSQP', 'options': {'maxiter': 100, 'ftol': 1e-8}},
                init='latinhypercube'
            )
            
            if result.fun < 1e5:
                inner = params_to_inner(result.x)
                outer_side = compute_min_outer_side(inner, (0.0, 0.0), outer_angle)
                if check_validity(inner, (0.0, 0.0), outer_side, outer_angle):
                    if outer_side < best_side:
                        best_side = outer_side
                        best_params = result.x
        except Exception:
            pass
    
    return best_side, best_params


def construct():
    t0 = time.time()
    
    best_side = 1e6
    best_result = None
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # Hexagonal cluster
    for scale in [1.8, 2.0, 2.2]:
        for rot_in in [0.0, 15.0, 30.0]:
            for rot_out in [0.0, 15.0, 30.0]:
                inner, oc, os_, oa = build_hexagonal_cluster(scale, rot_in, rot_out)
                if check_validity(inner, oc, os_, oa):
                    initial_configs.append((params_to_inner_flat(inner), oc, oa, os_))
    
    # Ring 12
    for scale in [1.8, 2.0, 2.2]:
        for rot_in in [0.0, 15.0, 30.0]:
            for rot_out in [0.0, 15.0, 30.0]:
                inner, oc, os_, oa = build_ring12(scale, rot_in, rot_out)
                if check_validity(inner, oc, os_, oa):
                    initial_configs.append((params_to_inner_flat(inner), oc, oa, os_))
    
    # Rect grid
    for scale in [1.8, 2.0, 2.2]:
        for rot_in in [0.0, 15.0, 30.0]:
            for rot_out in [0.0, 15.0, 30.0]:
                inner, oc, os_, oa = build_rect_grid(scale, rot_in, rot_out)
                if check_validity(inner, oc, os_, oa):
                    initial_configs.append((params_to_inner_flat(inner), oc, oa, os_))
    
    # Sort by initial side length
    initial_configs.sort(key=lambda x: x[3])
    
    # Try SLSQP from best initial configs
    for init_flat, oc, oa, init_side in initial_configs[:15]:
        if time.time() > DEADLINE - 20:
            break
        
        side, params = optimize_slsqp(init_flat, oc, oa, max_time=30.0)
        if side < best_side:
            best_side = side
            inner = params_to_inner(params)
            outer_side = compute_min_outer_side(inner, oc, oa)
            if check_validity(inner, oc, outer_side, oa):
                best_result = (inner, oc, outer_side, oa)
    
    # Try DE if time remains
    remaining = DEADLINE - time.time()
    if remaining > 40:
        de_side, de_params = optimize_de(max_time=min(remaining - 15, 80.0))
        if de_side < best_side:
            inner = params_to_inner(de_params)
            # Find the best outer angle
            best_oa = 0.0
            best_os = 1e6
            for oa in [0.0, 15.0, 30.0]:
                os_ = compute_min_outer_side(inner, (0.0, 0.0), oa)
                if check_validity(inner, (0.0, 0.0), os_, oa) and os_ < best_os:
                    best_os = os_
                    best_oa = oa
            if best_os < best_side:
                best_side = best_os
                best_result = (inner, (0.0, 0.0), best_os, best_oa)
    
    # Fallback
    if best_result is None:
        inner, oc, os_, oa = build_hexagonal_cluster(2.0, 0.0, 0.0)
        best_result = (inner, oc, os_, oa)
    
    return best_result


def params_to_inner_flat(inner):
    """Convert inner list to flat parameter vector."""
    params = []
    for (x, y, ang) in inner:
        params.append(x)
        params.append(y)
        params.append(ang)
    return np.array(params)

