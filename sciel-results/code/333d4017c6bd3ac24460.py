import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def _soft_max(x, beta=50.0):
    """Smooth max via log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _soft_min(x, beta=50.0):
    """Smooth min via negative log-sum-exp."""
    return -_soft_max(-x, beta)

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]
    
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)
    
    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        for i in range(6):
            nx, ny = en1x[i], en1y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        for i in range(6):
            nx, ny = en2x[i], en2y[i]
            p1 = v1x * nx + v1y * ny
            p2 = v2x * nx + v2y * ny
            if p1.max() < p2.min() or p2.max() < p1.min():
                return True
        return False
    
    def compute_min_outer_side(positions, angles_deg, fine_tune=True):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        max_sides = np.max(side_vals, axis=1)
        
        best_idx = np.argmin(max_sides)
        best_side = float(max_sides[best_idx])
        best_phi = float(best_idx)
        
        if fine_tune:
            try:
                lo = max(0.0, best_phi - 1.0)
                hi = min(60.0, best_phi + 1.0)
                res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                      options={'xatol': 1e-10})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = res.x
            except Exception:
                pass
            
        return best_side, best_phi, center[0], center[1]
    
    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True
    
    def make_grid_2x3(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_grid_3x2(scale=1.0, rotation_deg=0.0, sx=1.0, sy=1.0, shift_x=0.0, shift_y=0.0):
        a1 = np.array([sqrt3, 0.0]) * scale
        a2 = np.array([sqrt3/2.0, 1.5]) * scale
        a1 = a1 * np.array([sx, 1.0])
        a2 = a2 * np.array([sx, sy])
        
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(3):
            for i in range(2):
                pos = i * a1 + j * a2
                positions[idx] = pos + np.array([shift_x, shift_y])
                angles[idx] = rotation_deg
                idx += 1
        return positions, angles
    
    def make_hex_ring(center_pos, radius, rotation_deg=0.0):
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            theta = 2 * np.pi * i / N + np.radians(rotation_deg)
            positions[i, 0] = center_pos[0] + radius * math.cos(theta)
            positions[i, 1] = center_pos[1] + radius * math.sin(theta)
            angles[i] = rotation_deg
        return positions, angles
    
    def make_two_clusters():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        idx = 0
        for j in range(2):
            for i in range(3):
                positions[idx] = i * a1 + j * a2
                angles[idx] = 0.0
                idx += 1
        offset = 3 * a1 + 0.5 * a2
        for j in range(2):
            for i in range(3):
                positions[idx] = (i) * a1 + j * a2 + offset
                angles[idx] = 0.0
                idx += 1
        return positions, angles
    
    # Generate initial configurations
    initial_configs = []
    
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_2x3(scale=s))
    
    for s in [0.95, 0.98, 1.0, 1.02, 1.05]:
        initial_configs.append(make_grid_3x2(scale=s))
    
    for rot in [5, 10, 15, 20, 30]:
        initial_configs.append(make_grid_2x3(scale=1.0, rotation_deg=rot))
        initial_configs.append(make_grid_3x2(scale=1.0, rotation_deg=rot))
    
    for sx in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for sy in [0.95, 0.98, 1.0, 1.02, 1.05]:
            initial_configs.append(make_grid_2x3(scale=1.0, sx=sx, sy=sy))
    
    for r in [2.0, 2.5, 3.0, 3.5]:
        initial_configs.append(make_hex_ring([0, 0], r))
    
    initial_configs.append(make_two_clusters())
    
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    for scale in [0.9, 0.95, 1.0]:
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            positions[idx] = (i * a1 + j * a2) * scale
            angles[idx] = 0.0
        initial_configs.append((positions, angles))
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate all initial configs
    for pos, ang in initial_configs:
        if time.time() - t0 > DEADLINE * 0.25:
            break
        try:
            side, phi, cx, cy = compute_min_outer_side(pos, ang, fine_tune=False)
            if is_valid_configuration(pos, ang) and side < best_side:
                best_side = side
                best_positions = pos.copy()
                best_angles = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
        except Exception:
            continue
    
    if best_positions is None:
        best_positions, best_angles = make_grid_2x3(1.0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # SA phase
    sa_time_budget = DEADLINE * 0.45
    sa_deadline = t0 + sa_time_budget
    
    current_positions = best_positions.copy()
    current_angles = best_angles.copy()
    current_side, current_phi, current_cx, current_cy = compute_min_outer_side(current_positions, current_angles)
    
    T = 0.3
    T_min = 0.0001
    cooling = 0.99995
    step_size = 0.15
    step_decay = 0.99999
    
    while time.time() < sa_deadline:
        idx = random.randint(0, N - 1)
        new_positions = current_positions.copy()
        new_angles = current_angles.copy()
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 3.0)
        
        new_positions[idx, 0] += dx
        new_positions[idx, 1] += dy
        new_angles[idx] += da
        
        angles_rad_new = np.radians(new_angles)
        valid = True
        h_moved = (new_positions[idx, 0], new_positions[idx, 1], angles_rad_new[idx])
        for j in range(N):
            if j == idx:
                continue
            h_other = (new_positions[j, 0], new_positions[j, 1], angles_rad_new[j])
            if not hexagons_disjoint_fast(h_moved, h_other):
                valid = False
                break
        
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            step_size *= step_decay
            if step_size < 0.003: step_size = 0.003
            continue
        
        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_positions, new_angles, fine_tune=False)
        
        if new_side < current_side:
            current_positions = new_positions
            current_angles = new_angles
            current_side = new_side
            current_phi = new_phi
            current_cx = new_cx
            current_cy = new_cy
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_positions = new_positions
                current_angles = new_angles
                current_side = new_side
                current_phi = new_phi
                current_cx = new_cx
                current_cy = new_cy
                if new_side < best_side:
                    best_side = new_side
                    best_positions = new_positions.copy()
                    best_angles = new_angles.copy()
                    best_phi = new_phi
                    best_cx = new_cx
                    best_cy = new_cy
        
        T *= cooling
        if T < T_min: T = T_min
        step_size *= step_decay
        if step_size < 0.003: step_size = 0.003
    
    # Gradient-based optimization with improved smooth surrogate
    remaining = DEADLINE - (time.time() - t0)
    
    if remaining > 15.0:
        grad_deadline = t0 + remaining * 0.6
        
        def smooth_outer_side_surrogate(positions, angles_deg, beta=60.0):
            """Smooth log-sum-exp surrogate for the outer hexagon side length."""
            angles_rad = np.radians(angles_deg)
            all_verts = get_all_vertices(positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center

            # Sample phi values
            phi_vals = np.radians(np.arange(0, 60, 2))
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs

            # Soft min over phi candidates
            return _soft_min(side_vals.max(axis=1), beta=beta)

        def smooth_outer_side_and_grad(positions, angles_deg, beta=60.0):
            """Smooth surrogate with analytic gradient via LSE weights."""
            angles_rad = np.radians(angles_deg)
            c = np.cos(angles_rad)
            s = np.sin(angles_rad)
            rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
            ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
            verts_x = rx + positions[:, 0:1]
            verts_y = ry + positions[:, 1:2]
            verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            n_v = 2 * N  # 72 vertices

            phi_vals = np.radians(np.arange(0, 60, 2))
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            max_sides = side_vals.max(axis=1)

            # Soft-min over phi: f = -LSE(-max_sides, beta)
            m = np.max(-max_sides)
            exps = np.exp(beta * (-max_sides - m))
            w_phi = exps / np.sum(exps)
            f_val = -m - np.log(np.sum(exps)) / beta

            # Gradient w.r.t. max_sides: df/d(max_sides) = -beta * w_phi * sign(max_sides - max_sides)
            # Actually: f = -LSE(-x) where x = max_sides
            # LSE(-x) = m + log(sum(exp(beta*(-x_i - m))))
            # dLSE/dx_i = -beta * exp(beta*(-x_i - m)) / sum(...) = -beta * w_i
            # df/dx_i = -dLSE/dx_i = beta * w_phi[i]
            grad_max_sides = beta * w_phi

            # For each active phi, find which edge gives max
            edge_idx = np.argmax(side_vals, axis=1)
            # grad w.r.t. side_vals for each phi
            grad_side_vals = np.zeros_like(side_vals)
            for i in range(len(phi_vals)):
                grad_side_vals[i, edge_idx[i]] = grad_max_sides[i]

            # side_vals[i, j] = (2/sqrt3) * max_projs[i, j]
            # grad w.r.t. max_projs: same as grad_side_vals * sqrt3/2
            grad_max_projs = grad_side_vals * (sqrt3 / 2.0)

            # max_projs[i, j] = max over vertices of abs_proj[i, j, v]
            # Find argmax vertex for each (i, j)
            vert_idx = np.argmax(abs_proj, axis=0)
            # grad w.r.t. abs_proj: scatter
            grad_abs_proj = np.zeros_like(abs_proj)
            for i in range(len(phi_vals)):
                for j in range(6):
                    v = vert_idx[i, j]
                    grad_abs_proj[i, j, v] = grad_max_projs[i, j]

            # abs_proj[i, j, v] = |proj[i, j, v]|
            # d|proj|/dproj = sign(proj)
            sign_proj = np.sign(proj)
            # Handle zeros
            sign_proj[proj == 0] = 1.0
            grad_proj = grad_abs_proj * sign_proj

            # proj[i, j, v] = offsets[v, 0]*nx[i,j] + offsets[v, 1]*ny[i,j]
            # offsets[v] = verts_flat[v] - center
            # dproj/dverts_flat[v] = (nx[i,j]*sign, ny[i,j]*sign) - (1/n_v)*(sum over all)
            # But center = mean(verts_flat), so dcenter/dverts_flat[v] = (1/n_v)*(1,1)
            # dproj/dverts_flat[v] = (nx*sign - nx*sign/n_v, ny*sign - ny*sign/n_v) for the specific v
            # Actually more carefully:
            # proj[v] = (verts_flat[v] - center) · n
            # dproj[v]/dverts_flat[u] = n if u==v, -n/n_v otherwise
            # So: dproj/dverts_flat = sign * (n - mean(n_contrib))
            # Let's compute gradient w.r.t. verts_flat directly
            # grad_verts = zeros (n_v, 2)
            # For each (i, j, v): contrib = grad_proj[i,j,v] * (nx[i,j], ny[i,j])
            #   grad_verts[v] += contrib
            #   grad_verts[all] -= contrib / n_v
            grad_verts = np.zeros((n_v, 2))
            for i in range(len(phi_vals)):
                for j in range(6):
                    g = grad_proj[i, j]
                    if np.all(g == 0):
                        continue
                    nxj = nx[i, j]
                    nyj = ny[i, j]
                    # g is (n_v,)
                    contrib = g[:, None] * np.array([nxj, nyj])
                    grad_verts += contrib
                    grad_verts -= contrib.mean(axis=0, keepdims=True)

            # Now map grad_verts back to positions and angles
            # verts_flat is reshape of stacked (verts_x, verts_y)
            # verts_x[n, k] = _ux[k]*c[n] - _uy[k]*s[n] + positions[n,0]
            # verts_y[n, k] = _ux[k]*s[n] + _uy[k]*c[n] + positions[n,1]
            # So dverts_x[n,k]/dpositions[n,0] = 1
            #    dverts_y[n,k]/dpositions[n,1] = 1
            #    dverts_x[n,k]/dangles[n] = -_ux[k]*s[n] - _uy[k]*c[n]
            #    dverts_y[n,k]/dangles[n] = _ux[k]*c[n] - _uy[k]*s[n]
            # (angles in radians; we'll convert)
            grad_pos = np.zeros((N, 2))
            grad_ang_rad = np.zeros(N)
            for n in range(N):
                for k in range(6):
                    vx_idx = n * 6 + k
                    vy_idx = n * 6 + 6 + k
                    # Wait, the stacking is: np.stack((verts_x, verts_y), axis=-1) gives (N, 6, 2)
                    # Then reshape(-1, 2) gives (72, 2) where index n*6+k is (verts_x[n,k], verts_y[n,k])
                    gx = grad_verts[n * 6 + k, 0]
                    gy = grad_verts[n * 6 + k, 1]
                    grad_pos[n, 0] += gx
                    grad_pos[n, 1] += gy
                    # d/dangle (in radians)
                    ddx_da = -_ux[k] * s[n] - _uy[k] * c[n]
                    ddy_da = _ux[k] * c[n] - _uy[k] * s[n]
                    grad_ang_rad[n] += gx * ddx_da + gy * ddy_da

            # Convert angle gradient to degrees
            grad_ang_deg = grad_ang_rad * (180.0 / np.pi)

            grad = np.concatenate([grad_pos.flatten(), grad_ang_deg])
            return f_val, grad
        
        def smooth_outer_side_surrogate2(positions, angles_deg, beta=40.0):
            """Alternative surrogate using direct soft-max over all (vertex, phi, edge) combos."""
            angles_rad = np.radians(angles_deg)
            all_verts = get_all_vertices(positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            
            # Fewer phi values for speed
            phi_vals = np.radians(np.arange(0, 60, 5))
            normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
            abs_proj = np.abs(proj)
            
            # Soft max over vertices for each (phi, edge)
            max_v = abs_proj.max(axis=0, keepdims=True)
            logsumexp_v = max_v.squeeze(0) + np.log(np.sum(np.exp(beta * (abs_proj - max_v)), axis=0)) / beta
            
            # Soft max over edges for each phi
            max_e = logsumexp_v.max(axis=1, keepdims=True)
            logsumexp_e = max_e.squeeze(1) + np.log(np.sum(np.exp(beta * (logsumexp_v - max_e)), axis=1)) / beta
            
            # Soft max over phi
            max_p = logsumexp_e.max(axis=0, keepdims=True)
            logsumexp_phi = max_p.squeeze(0) + np.log(np.sum(np.exp(beta * (logsumexp_e - max_p)), axis=0)) / beta
            
            return (2.0 / sqrt3) * logsumexp_phi
        
        # Try gradient-based optimization with analytic gradient
        x0 = np.concatenate([best_positions.flatten(), best_angles])

        def obj_and_grad(x):
            pos = x[:2*N].reshape(N, 2)
            ang = x[2*N:]
            return smooth_outer_side_and_grad(pos, ang, beta=60.0)

        try:
            res = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.success or res.fun < obj_and_grad(x0)[0]:
                new_pos = res.x[:2*N].reshape(N, 2)
                new_ang = res.x[2*N:]
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos.copy()
                        best_angles = new_ang.copy()
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
                        x0 = res.x.copy()
        except Exception:
            pass

        # Also try the non-gradient surrogate as backup
        if time.time() < grad_deadline:
            try:
                res = minimize(
                    lambda x: smooth_outer_side_surrogate(x[:2*N].reshape(N, 2), x[2*N:]),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30}
                )
                if res.success:
                    new_pos = res.x[:2*N].reshape(N, 2)
                    new_ang = res.x[2*N:]
                    if is_valid_configuration(new_pos, new_ang):
                        new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                        if new_side < best_side:
                            best_side = new_side
                            best_positions = new_pos.copy()
                            best_angles = new_ang.copy()
                            best_phi = new_phi
                            best_cx = new_cx
                            best_cy = new_cy
                            x0 = res.x.copy()
            except Exception:
                pass

        # Try from perturbed starts with analytic gradient
        if time.time() < grad_deadline:
            n_restarts = 10
            for restart in range(n_restarts):
                if time.time() > grad_deadline:
                    break
                x0_pert = x0.copy()
                x0_pert[:2*N] += random.gauss(0, 0.06, (2*N,))
                x0_pert[2*N:] += random.gauss(0, 1.2, (N,))
                try:
                    res = minimize(
                        obj_and_grad,
                        x0_pert,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
                    )
                    if res.success:
                        new_pos = res.x[:2*N].reshape(N, 2)
                        new_ang = res.x[2*N:]
                        if is_valid_configuration(new_pos, new_ang):
                            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = new_pos.copy()
                                best_angles = new_ang.copy()
                                best_phi = new_phi
                                best_cx = new_cx
                                best_cy = new_cy
                except Exception:
                    pass
    
    # Coordinate descent polish with adaptive step
    remaining = DEADLINE - (time.time() - t0)

    if remaining > 5.0:
        cd_deadline = t0 + remaining * 0.6
        step = 0.05
        min_step = 0.00005
        max_fail = 0
        max_fail_limit = 30

        while time.time() < cd_deadline and step > min_step:
            improved = False
            max_fail = 0
            for idx in range(N):
                for dim in range(3):
                    found_improve = False
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()

                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step

                        angles_rad_new = np.radians(new_ang)
                        valid = True
                        h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                        for j in range(N):
                            if j == idx: continue
                            h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break

                        if not valid:
                            max_fail += 1
                            continue

                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang, fine_tune=False)

                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            found_improve = True
                            break
                        else:
                            max_fail += 1
                    if found_improve:
                        break
                if improved:
                    break
            if not improved:
                step *= 0.5
                if max_fail > max_fail_limit:
                    step *= 0.5
            else:
                step = max(step * 0.85, min_step)

        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    # Final fine-tune of outer hexagon angle
    remaining2 = DEADLINE - (time.time() - t0)
    if remaining2 > 3.0:
        def eval_final_phi(phi_deg):
            phi = np.radians(phi_deg)
            angles_rad = np.radians(best_angles)
            all_verts = get_all_vertices(best_positions, angles_rad)
            verts_flat = all_verts.reshape(-1, 2)
            center = np.mean(verts_flat, axis=0)
            offsets = verts_flat - center
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)

        try:
            res = minimize_scalar(eval_final_phi, bounds=(0, 60), method='bounded',
                                  options={'xatol': 1e-12})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
                angles_rad = np.radians(best_angles)
                all_verts = get_all_vertices(best_positions, angles_rad)
                verts_flat = all_verts.reshape(-1, 2)
                center = np.mean(verts_flat, axis=0)
                best_cx = center[0]
                best_cy = center[1]
        except Exception:
            pass

        # Also try fine-tuning individual hexagon angles in a final pass
        if remaining2 > 6.0:
            for idx in range(N):
                if time.time() > t0 + DEADLINE - 2.0:
                    break
                for _iter in range(3):
                    best_local = best_side
                    best_local_ang = best_angles[idx]
                    for trial in range(20):
                        trial_ang = best_angles[idx] + (trial - 10) * 0.3
                        new_ang = best_angles.copy()
                        new_ang[idx] = trial_ang
                        angles_rad_new = np.radians(new_ang)
                        valid = True
                        h_moved = (best_positions[idx, 0], best_positions[idx, 1], angles_rad_new[idx])
                        for j in range(N):
                            if j == idx: continue
                            h_other = (best_positions[j, 0], best_positions[j, 1], angles_rad_new[j])
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break
                        if not valid:
                            continue
                        trial_side, _, _, _ = compute_min_outer_side(best_positions, new_ang, fine_tune=False)
                        if trial_side < best_local:
                            best_local = trial_side
                            best_local_ang = trial_ang
                    if best_local < best_side - 1e-12:
                        best_angles[idx] = best_local_ang
                        best_side = best_local
                        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                    else:
                        break
    
    # Validate and finalize
    if not is_valid_configuration(best_positions, best_angles):
        best_positions, best_angles = make_grid_2x3(1.0)
    
    final_side, final_phi, final_cx, final_cy = compute_min_outer_side(best_positions, best_angles)
    if final_side < best_side:
        best_side = final_side
        best_phi = final_phi
        best_cx = final_cx
        best_cy = final_cy
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
