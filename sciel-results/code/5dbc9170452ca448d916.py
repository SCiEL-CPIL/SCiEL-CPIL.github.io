import math
import time
import itertools
from scipy.optimize import minimize
import numpy as np

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    oangles = [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0,
               10.0, -10.0, 20.0, -20.0, 50.0, -50.0, 80.0, -80.0, 5.0, -5.0]
    
    for combo in all_combos:
        if time.time() > deadline - 40.0:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        for oang in oangles:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang

    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Pre-compute vertices of inner hexagons
        inner_verts = []
        for h in inner_hexes:
            inner_verts.extend(_hex_vertices(h[0], h[1], h[2], h[3]))
        
        # Convert to numpy array for speed
        inner_verts_np = np.array(inner_verts)
        
        def soft_max_outer_side(cx, cy, angle_deg, beta=50.0):
            """Smooth approximation of _min_outer_side using log-sum-exp."""
            a = -math.radians(angle_deg)
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            
            # Rotate inner vertices relative to center
            dx = inner_verts_np[:, 0] - cx
            dy = inner_verts_np[:, 1] - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # Project onto 6 normals
            projs = []
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                p = lx * nx + ly * ny
                projs.append(p)
            
            projs = np.array(projs).flatten()
            
            m = np.max(projs)
            exp_sum = np.sum(np.exp(projs - m))
            soft_max = m + math.log(exp_sum) / beta
            
            return soft_max * 2.0 / sqrt3
        
        def gradient_soft_max(cx, cy, angle_deg, beta=50.0):
            """Analytic gradient of soft_max_outer_side."""
            a = -math.radians(angle_deg)
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            
            dx = inner_verts_np[:, 0] - cx
            dy = inner_verts_np[:, 1] - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # Weights for each projection
            # There are 6 normals, each projecting all vertices
            # Total 6 * N_verts values
            
            weights_sum = 0.0
            grad_cx = 0.0
            grad_cy = 0.0
            grad_ang = 0.0
            
            # To avoid numerical instability in exp, we subtract max
            # But we need to compute weights w_ik = exp(beta * u_ik) / Sum(exp(beta * u_jl))
            
            # First pass: find max and sum of exps
            max_proj = -np.inf
            exp_vals = []
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                p = lx * nx + ly * ny
                exp_vals.append(p)
                if np.max(p) > max_proj:
                    max_proj = np.max(p)
            
            # Second pass: compute weights
            total_exp_sum = 0.0
            exp_mats = []
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                p = lx * nx + ly * ny
                e = np.exp(p - max_proj)
                exp_mats.append(e)
                total_exp_sum += np.sum(e)
            
            # Normalize weights
            # w_ik = e_ik / total_exp_sum
            
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                nx = math.cos(theta)
                ny = math.sin(theta)
                e = exp_mats[k]
                w = e / total_exp_sum
                
                # Gradient wrt cx, cy
                # u_ik = n_k . (v_i - c)
                # du_ik/dc = -n_k
                # grad_cx += sum(w_ik * (-nx))
                # grad_cy += sum(w_ik * (-ny))
                
                sum_w = np.sum(w)
                grad_cx -= sum_w * nx
                grad_cy -= sum_w * ny
                
                # Gradient wrt angle
                # du_ik/dang = n_k_perp . (v_i - c)
                # n_k_perp = (-ny, nx)
                # (v_i - c) in local coords is (lx, ly)
                # dot = -ny * lx + nx * ly
                dot_val = -ny * lx + nx * ly
                grad_ang += np.sum(w * dot_val)
            
            # Chain rule: soft_max = (1/beta) log(sum exp(beta u))
            # d(soft_max)/dx = (1/beta) * (sum exp(beta u) du/dx) / sum exp(beta u)
            # = sum (exp(beta u) / sum exp(beta u)) * du/dx
            # = sum w * du/dx
            # Note: The factor 1/beta cancels out in the gradient of the log-sum-exp form?
            # Let S = (1/beta) log(sum exp(beta u))
            # dS/du = (1/beta) * (beta sum exp(beta u) du) / sum exp(beta u) = sum w du
            # So the gradient is just sum w * du/dx.
            
            # Scale by 2/sqrt(3)
            factor = 2.0 / sqrt3
            return factor * grad_cx, factor * grad_cy, factor * grad_ang

        # Use scipy minimize
        x0 = np.array([best_center[0], best_center[1], best_angle])
        
        def objective(x):
            return soft_max_outer_side(x[0], x[1], x[2], beta=50.0)
            
        def jac(x):
            return gradient_soft_max(x[0], x[1], x[2], beta=50.0)

        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success and res.fun < best_side:
                # Verify with exact calculation
                cx_opt, cy_opt, ang_opt = res.x
                exact_side = _min_outer_side(inner_hexes, cx_opt, cy_opt, ang_opt)
                if exact_side < best_side:
                    best_side = exact_side
                    best_center = (cx_opt, cy_opt)
                    best_angle = ang_opt
        except Exception:
            pass

    if best_positions is None:
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
