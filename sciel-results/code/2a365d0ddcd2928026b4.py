import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hex_edge_normals(rot_deg):
    base = np.radians(np.array([30, 90, 150, 210, 270, 330])) + np.radians(rot_deg)
    return np.column_stack((np.cos(base), np.sin(base)))

def _sat_overlap_and_grad(v1, v2, n1, n2):
    """Returns (overlap, grad_wrt_v1, grad_wrt_v2) using the critical axis."""
    min_overlap = float('inf')
    crit_axis = None
    crit_from = 0  # 0 = from n1, 1 = from n2
    
    for k in range(6):
        n = n1[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap, None, None
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            crit_axis = n
            crit_from = 0
    
    for k in range(6):
        n = n2[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap, None, None
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            crit_axis = n
            crit_from = 1
    
    # Gradient: overlap = min(max1, max2) - max(min1, min2) along crit_axis
    # d(overlap)/d(v1) = -sign(max1 < max2) * crit_axis + sign(min1 > min2) * crit_axis
    # More precisely: if max1 < max2: d(max1)/d(v1) = crit_axis (for vertex achieving max1)
    # We use a simplified gradient: -crit_axis for the "pushing" direction
    # For L-BFGS-B, we need grad wrt all 12 vertices of each hex
    # The overlap changes by: if we move v1 by dv, proj1 shifts by dv@n
    # So d(overlap)/dv1 = -n if v1's max is the binding one, +n if v1's min is binding
    
    n = crit_axis
    proj1 = v1 @ n
    proj2 = v2 @ n
    min1, max1 = np.min(proj1), np.max(proj1)
    min2, max2 = np.min(proj2), np.max(proj2)
    
    # Which bound is tight?
    max1_tight = (max1 < max2)  # v1's max is the upper bound
    min1_tight = (min1 > min2)  # v1's min is the lower bound
    
    # Gradient wrt v1: 
    # If max1_tight: moving v1 in +n direction increases max1, decreasing overlap => grad = -n
    # If min1_tight: moving v1 in -n direction decreases min1, decreasing overlap => grad = -n
    # Actually: overlap = min(max1,max2) - max(min1,min2)
    # d(overlap)/d(v1) = d(min(max1,max2))/d(v1) - d(max(min1,min2))/d(v1)
    # = (n if max1<max2 else 0) - (n if min1>min2 else 0)
    # Wait, that's not right. Let me think again.
    # proj1_i = v1_i @ n, so d(proj1_i)/d(v1_i) = n
    # max1 = max_i proj1_i, so d(max1)/d(v1_i) = n if i is argmax, else 0
    # min1 = min_i proj1_i, so d(min1)/d(v1_i) = n if i is argmin, else 0
    # overlap = min(max1,max2) - max(min1,min2)
    # d(overlap)/d(v1_i) = [n if (max1<max2 and i==argmax1) else 0] - [n if (min1>min2 and i==argmin1) else 0]
    
    grad_v1 = np.zeros_like(v1)
    grad_v2 = np.zeros_like(v2)
    
    if max1 < max2:
        i_max1 = np.argmax(proj1)
        grad_v1[i_max1] += n
    if min1 > min2:
        i_min1 = np.argmin(proj1)
        grad_v1[i_min1] -= n
    
    if max1 < max2:
        i_max2 = np.argmax(proj2)
        grad_v2[i_max2] -= n
    if min1 > min2:
        i_min2 = np.argmin(proj2)
        grad_v2[i_min2] += n
    
    return -min_overlap, grad_v1, grad_v2

def _outer_hex_side_fast(points, outer_rot_deg):
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def _check_disjoint(inner, eps=1e-9):
    n = len(inner)
    verts = []
    norms = []
    for i in range(n):
        x, y, rot = inner[i]
        v = _hex_vertices(x, y, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts.append(v)
        norms.append(nm)
    for i in range(n):
        for j in range(i + 1, n):
            d, _, _ = _sat_overlap_and_grad(verts[i], verts[j], norms[i], norms[j])
            if d < -eps:
                return False
    return True

def _make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
    pts = []
    sqrt3 = np.sqrt(3)
    spacing_x = 2.0 + gap
    spacing_y = sqrt3 + gap * 0.5
    for r in range(rows):
        y = r * spacing_y
        for c in range(cols):
            x = c * spacing_x
            if r % 2 == 1:
                x += spacing_x / 2.0
            pts.append([x, y, rot_offset])
    pts = np.array(pts)
    if len(pts) > 12:
        pts = pts[:12]
    elif len(pts) < 12:
        while len(pts) < 12:
            pts.append([0, 0, rot_offset])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_ring(gap=0.0):
    pts = []
    d1 = np.sqrt(3) + gap
    d2 = 2 * np.sqrt(3) + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _objective_and_grad(p, t0, deadline):
    """Returns (objective, gradient) for L-BFGS-B."""
    if time.time() - t0 > deadline:
        return 1e6, None
    
    inner = p[:36].reshape(12, 3)
    ox = p[36]
    oy = p[37]
    oang = p[38]
    
    verts_list = []
    norms_list = []
    for i in range(12):
        cx, cy, rot = inner[i]
        v = _hex_vertices(cx, cy, 1.0, rot)
        nm = _hex_edge_normals(rot)
        verts_list.append(v)
        norms_list.append(nm)
    
    penalty = 0.0
    grad = np.zeros_like(p)
    
    for i in range(12):
        for j in range(i + 1, 12):
            d, gv1, gv2 = _sat_overlap_and_grad(verts_list[i], verts_list[j], norms_list[i], norms_list[j])
            if d < 0:
                overlap_sq = (-d) ** 2
                penalty += overlap_sq * 50.0
                # Gradient of penalty wrt positions
                # d(overlap_sq)/d(pos) = 2*(-d) * d(overlap)/d(pos) = 2*(-d) * (-grad_overlap)
                # = 2*d * grad_overlap (since d<0, -d>0)
                # Actually: penalty = 50 * (-d)^2, d(overlap)/d(v) = gv
                # d(penalty)/d(v) = 50 * 2*(-d) * d(-d)/d(v) = 50 * 2*(-d) * (-gv) = 50 * 2 * d * gv
                if gv1 is not None:
                    # Gradient wrt cx, cy of hex i
                    # v = _hex_vertices(cx, cy, 1, rot)
                    # dv/dc = [[cos(a_k), sin(a_k)] for k] where a_k are vertex angles
                    # d(penalty)/dcx = sum_k gv1[k] * cos(a_k)
                    # d(penalty)/dcy = sum_k gv1[k] * sin(a_k)
                    rot_i = np.radians(inner[i, 2])
                    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + rot_i
                    cos_a = np.cos(angles)
                    sin_a = np.sin(angles)
                    grad[i*3] += 50.0 * 2.0 * d * np.dot(gv1[:, 0], cos_a)
                    grad[i*3+1] += 50.0 * 2.0 * d * np.dot(gv1[:, 1], sin_a)
                    # Gradient wrt rotation: more complex, skip for now or approximate
                    # d(v)/d(rot) = -side * [[sin(a_k), cos(a_k)]]
                    # d(penalty)/d(rot) = sum_k gv1[k] * (-sin(a_k)) for x, (-cos(a_k)) for y... 
                    # Actually dv_k/drot = side * [-sin(a_k), cos(a_k)]
                    # d(penalty)/drot = sum_k (gv1[k,0] * (-sin(a_k)) + gv1[k,1] * cos(a_k))
                    grad[i*3+2] += 50.0 * 2.0 * d * np.sum(gv1[:, 0] * (-np.sin(angles)) + gv1[:, 1] * np.cos(angles))
                
                if gv2 is not None:
                    rot_j = np.radians(inner[j, 2])
                    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + rot_j
                    cos_a = np.cos(angles)
                    sin_a = np.sin(angles)
                    grad[j*3] += 50.0 * 2.0 * d * np.dot(gv2[:, 0], cos_a)
                    grad[j*3+1] += 50.0 * 2.0 * d * np.dot(gv2[:, 1], sin_a)
                    grad[j*3+2] += 50.0 * 2.0 * d * np.sum(gv2[:, 0] * (-np.sin(angles)) + gv2[:, 1] * np.cos(angles))
    
    all_pts = np.vstack(verts_list)
    all_pts[:, 0] -= ox
    all_pts[:, 1] -= oy
    
    L = _outer_hex_side_fast(all_pts, oang)
    
    # Gradient of L wrt ox, oy, oang
    # L = (2/sqrt(3)) * max_k |proj_k| where proj_k = all_pts @ n_k
    # dL/d(ox) = (2/sqrt(3)) * sign(max_k proj_k) * n_k[0] for the critical k
    # This is complex; use finite differences for outer params
    eps = 1e-6
    L_ox = _outer_hex_side_fast(all_pts - np.array([eps, 0]), oang)
    L_oy = _outer_hex_side_fast(all_pts - np.array([0, eps]), oang)
    L_oang = _outer_hex_side_fast(all_pts, oang + eps)
    grad[36] = (L_ox - L) / eps
    grad[37] = (L_oy - L) / eps
    grad[38] = (L_oang - L) / eps
    
    return L + penalty, grad

def _objective_only(p, t0, deadline):
    val, _ = _objective_and_grad(p, t0, deadline)
    return val

def _polish_lbfgs(p, t0, deadline, maxiter=100):
    """Polish using L-BFGS-B."""
    try:
        res = minimize(
            _objective_only,
            p,
            args=(t0, deadline),
            method='L-BFGS-B',
            jac=lambda x, t0, deadline: _objective_and_grad(x, t0, deadline)[1],
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < _objective_only(p, t0, deadline):
            return res.x
    except Exception:
        pass
    return p

def _simulated_annealing(p, t0, deadline, T_init=2.0, T_final=0.01, n_iter=500):
    """Simulated annealing for global search."""
    current = p.copy()
    current_val = _objective_only(current, t0, deadline)
    best = current.copy()
    best_val = current_val
    
    n_steps = n_iter
    for it in range(n_steps):
        if time.time() - t0 > deadline:
            break
        
        T = T_init * (T_final / T_init) ** (it / n_steps)
        
        # Generate perturbation
        perturbation = np.random.randn(len(current)) * 0.3
        # Scale position perturbations
        perturbation[:36] *= 0.5
        perturbation[36:39] *= 0.1
        
        new_p = current + perturbation
        
        # Wrap angles to [0, 360)
        new_p[2::3] = new_p[2::3] % 360
        
        new_val = _objective_only(new_p, t0, deadline)
        
        if new_val < current_val or np.random.rand() < np.exp(-(new_val - current_val) / T):
            current = new_p
            current_val = new_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
    
    return best

def _make_packed_cluster():
    """Create a tightly packed hexagonal cluster."""
    pts = []
    sqrt3 = np.sqrt(3)
    # Center hex
    pts.append([0, 0, 0.0])
    # 6 neighbors at distance sqrt(3)
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
    # 5 more at distance 2*sqrt(3) at 30-degree offsets
    for k in range(5):
        ang = np.radians(60 * k + 30)
        pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def _make_optimal_structure():
    """Create a structure based on known optimal packings."""
    # For 12 hexagons, a good arrangement is a 3-4-3-2 pattern or similar
    sqrt3 = np.sqrt(3)
    pts = []
    # Row 1: 3 hexes
    for i in range(3):
        pts.append([(i - 1) * 2, 0, 0.0])
    # Row 2: 4 hexes (offset)
    for i in range(4):
        pts.append([(i - 1.5) * 2, sqrt3, 0.0])
    # Row 3: 3 hexes
    for i in range(3):
        pts.append([(i - 1) * 2, 2 * sqrt3, 0.0])
    # Row 4: 2 hexes (offset)
    for i in range(2):
        pts.append([(i - 0.5) * 2, 3 * sqrt3, 0.0])
    pts = np.array(pts)
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    return pts

def construct():
    t0 = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - t0)
    
    np.random.seed(42)
    
    # Generate diverse initial configurations
    configs = [
        _make_hex_grid(4, 3, gap=0.1),
        _make_hex_grid(3, 4, gap=0.1),
        _make_hex_grid(2, 6, gap=0.1),
        _make_hex_grid(6, 2, gap=0.1),
        _make_ring(gap=0.1),
        _make_hex_grid(4, 3, rot_offset=30, gap=0.1),
        _make_hex_grid(3, 4, rot_offset=30, gap=0.1),
        _make_packed_cluster(),
        _make_optimal_structure(),
    ]
    
    # Add random configurations
    for _ in range(10):
        pts = np.random.randn(12, 3) * 2.0
        pts[:, 2] = np.random.uniform(0, 60, 12)
        pts[:, 0] -= np.mean(pts[:, 0])
        pts[:, 1] -= np.mean(pts[:, 1])
        configs.append(pts)
    
    best_L = float('inf')
    best_params = None
    
    for cfg in configs:
        if time_left() < 15:
            break
        
        for oang in [0.0, 30.0]:
            if time_left() < 5:
                break
            
            init_params = np.concatenate((cfg.flatten(), [0.0, 0.0, oang]))
            
            # Stage 1: Simulated Annealing for global search
            sa_params = _simulated_annealing(
                init_params, t0, DEADLINE - 5,
                T_init=1.0, T_final=0.01, n_iter=200
            )
            
            # Stage 2: L-BFGS-B polish
            polished = _polish_lbfgs(sa_params, t0, DEADLINE - 3, maxiter=80)
            
            # Stage 3: Fine SA
            fine_sa = _simulated_annealing(
                polished, t0, DEADLINE - 2,
                T_init=0.3, T_final=0.005, n_iter=100
            )
            
            # Stage 4: Final L-BFGS-B polish
            final = _polish_lbfgs(fine_sa, t0, DEADLINE - 1, maxiter=50)
            
            val = _objective_only(final, t0, DEADLINE)
            
            if val < best_L:
                inner = final[:36].reshape(12, 3)
                if _check_disjoint(inner):
                    ox = final[36]
                    oy = final[37]
                    oang = final[38]
                    all_pts = np.vstack([_hex_vertices(inner[i,0], inner[i,1], 1.0, inner[i,2]) for i in range(12)])
                    all_pts[:, 0] -= ox
                    all_pts[:, 1] -= oy
                    L = _outer_hex_side_fast(all_pts, oang)
                    
                    if L < best_L:
                        best_L = L
                        best_params = final.copy()
    
    if best_params is None:
        # Fallback: use the best initial config
        cfg = _make_packed_cluster()
        best_params = np.concatenate((cfg.flatten(), [0.0, 0.0, 0.0]))
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    inner = []
    for i in range(12):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        r = best_params[i * 3 + 2] % 360
        inner.append([float(x), float(y), float(r)])
    
    outer_center = (float(best_params[36]), float(best_params[37]))
    outer_angle = float(best_params[38]) % 360
    
    if not _check_disjoint(inner):
        # Fallback to a known valid configuration
        cfg = _make_hex_grid(4, 3, gap=0.5)
        inner = [[float(p[0]), float(p[1]), float(p[2])] for p in cfg]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
        outer_side = _outer_hex_side_fast(all_pts, outer_angle)
        return inner, outer_center, outer_side, outer_angle
    
    all_pts = np.vstack([_hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner])
    all_pts[:, 0] -= outer_center[0]
    all_pts[:, 1] -= outer_center[1]
    outer_side = _outer_hex_side_fast(all_pts, outer_angle)
    outer_side *= 1.0001
    
    return inner, outer_center, outer_side, outer_angle
