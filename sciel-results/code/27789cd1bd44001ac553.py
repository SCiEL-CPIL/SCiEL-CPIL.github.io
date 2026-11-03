import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]
    
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
        if dist_sq > 4.0:
            return True
        if dist_sq < 0.25:
            return False
        
        a1 = h1[2]
        a2 = h2[2]
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
        
        normals_x = np.concatenate([en1x, en2x])
        normals_y = np.concatenate([en1y, en2y])
        
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        
        p1 = v1x[:, None] * normals_x[None, :] + v1y[:, None] * normals_y[None, :]
        p2 = v2x[:, None] * normals_x[None, :] + v2y[:, None] * normals_y[None, :]
        
        p1_min = p1.min(axis=0)
        p1_max = p1.max(axis=0)
        p2_min = p2.min(axis=0)
        p2_max = p2.max(axis=0)
        
        if np.any(p1_max < p2_min) or np.any(p2_max < p1_min):
            return True
            
        return False

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
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

    def smooth_collision_penalty(params):
        """Differentiable collision penalty based on min distance between hexagon boundaries."""
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        angles_rad = np.radians(ang)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Min center distance for non-overlapping unit hexagons: 2*sqrt(3)
                min_dist = 2.0 * sqrt3
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += 50.0 * overlap * overlap
        return penalty

    def smooth_outer_side(params):
        """Epigraph-style smooth outer side length."""
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        angles_rad = np.radians(ang)
        
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # Use finer scan for smoothness
        phi_vals = np.radians(np.arange(0, 60, 0.5))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        max_projs = np.max(np.abs(proj), axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)
        return float(np.min(side_per_phi))

    def combined_objective(params):
        """Outer side + collision penalty."""
        return smooth_outer_side(params) + smooth_collision_penalty(params)

    def combined_gradient(params):
        """Numerical gradient of combined objective."""
        eps = 1e-4
        grad = np.zeros_like(params)
        f0 = combined_objective(params)
        for k in range(len(params)):
            params_p = params.copy()
            params_p[k] += eps
            f_p = combined_objective(params_p)
            grad[k] = (f_p - f0) / eps
        return grad

    # Generate multiple initial configurations
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])

    def make_honeycomb():
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 1.5 * 2]
        return pos, ang

    def make_compact_hex():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        pos[0] = [0, 0]
        for i in range(6):
            theta = i * np.pi / 3
            pos[i + 1] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(5):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 7] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_tight_cluster():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
        for i in range(6):
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 3.0]
        return pos, ang

    def make_staggered():
        """3 rows of 4, staggered hexagonal packing"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for row in range(3):
            for col in range(4):
                idx = row * 4 + col
                pos[idx] = [col * 2 * sqrt3 + (row % 2) * sqrt3, row * 3.0]
        return pos, ang

    def make_compact_2x6():
        """2 rows of 6 with tighter vertical spacing"""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            pos[i] = [i * 2 * sqrt3, 0]
            pos[i + 6] = [i * 2 * sqrt3 + sqrt3, 2.8]
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_two_rows(),
                    make_compact_hex(), make_tight_cluster(), make_staggered(), make_compact_2x6()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Evaluate all initial configs
    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # === SMOOTH OPTIMIZATION PHASE ===
    # Use gradient-based optimization with collision penalty
    # This replaces the SA approach with a structurally different method
    
    opt_time_budget = DEADLINE * 0.55
    
    def run_gradient_optimization(init_pos, init_ang, time_limit):
        """Run L-BFGS-B on smooth combined objective."""
        t_start = time.time()
        
        params = np.zeros(3 * N)
        for i in range(N):
            params[3*i] = init_pos[i, 0]
            params[3*i+1] = init_pos[i, 1]
            params[3*i+2] = init_ang[i]
        
        best_params = params.copy()
        best_val = combined_objective(params)
        
        # Multiple restarts with perturbation
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - t_start > time_limit:
                break
            
            if r == 0:
                p = params.copy()
            else:
                p = params.copy()
                # Add random perturbation for restart
                p[:2*N] += np.random.normal(0, 0.3, 2*N)
                p[2*N:] += np.random.normal(0, 5.0, N)
            
            try:
                res = minimize(combined_objective, p, method='L-BFGS-B',
                              jac=combined_gradient,
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_val:
                    # Check if the solution has low collision penalty
                    pos_check = res.x[:2*N].reshape(N, 2)
                    ang_check = res.x[2*N:]
                    if is_valid_configuration(pos_check, ang_check):
                        best_params = res.x.copy()
                        best_val = res.fun
            except Exception:
                pass
            
            if time.time() - t_start > time_limit:
                break
        
        return best_params, best_val

    # === GRADIENT OPTIMIZATION PHASE ===
    opt_time = min(opt_time_budget, DEADLINE - (time.time() - t0))
    if opt_time > 3.0:
        opt_params, opt_val = run_gradient_optimization(best_positions, best_angles, opt_time)
        
        # Extract and validate
        new_pos = opt_params[:2*N].reshape(N, 2)
        new_ang = opt_params[2*N:]
        
        if is_valid_configuration(new_pos, new_ang):
            new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
            if new_side < best_side:
                best_side = new_side
                best_positions = new_pos
                best_angles = new_ang
                best_phi = new_phi
                best_cx = new_cx
                best_cy = new_cy

    # === REFINEMENT PHASE: Alternating optimization ===
    # Alternate between optimizing positions (fixed angles) and angles (fixed positions)
    remaining_time = DEADLINE - (time.time() - t0)
    
    if remaining_time > 5.0:
        def refine_alternating(pos, ang, time_limit):
            """Alternating coordinate optimization with gradient steps."""
            t_start = time.time()
            best_p = pos.copy()
            best_a = ang.copy()
            best_s, _, _, _ = compute_min_outer_side(best_p, best_a)
            
            # Phase 1: Optimize positions with fixed angles
            for iteration in range(50):
                if time.time() - t_start > time_limit:
                    break
                
                # Gradient descent on positions
                lr = 0.02
                for _ in range(10):
                    if time.time() - t_start > time_limit:
                        break
                    
                    # Compute gradient numerically for position params only
                    pos_params = best_p.flatten()
                    ang_fixed = best_a
                    
                    def obj_pos(p_flat):
                        pp = p_flat.reshape(N, 2)
                        s, _, _, _ = compute_min_outer_side(pp, ang_fixed)
                        # Add collision penalty
                        pen = 0.0
                        for i in range(N):
                            for j in range(i+1, N):
                                dx = pp[i,0] - pp[j,0]
                                dy = pp[i,1] - pp[j,1]
                                d = math.sqrt(dx*dx + dy*dy)
                                if d < 2.0*sqrt3:
                                    pen += 30.0 * (2.0*sqrt3 - d)**2
                        return s + pen
                    
                    # Simple gradient step
                    eps = 1e-3
                    grad = np.zeros(2*N)
                    f0 = obj_pos(pos_params)
                    for k in range(2*N):
                        pp = pos_params.copy()
                        pp[k] += eps
                        fp = obj_pos(pp)
                        grad[k] = (fp - f0) / eps
                    
                    new_pos_params = pos_params - lr * grad
                    new_pp = new_pos_params.reshape(N, 2)
                    
                    if is_valid_configuration(new_pp, best_a):
                        new_s, _, _, _ = compute_min_outer_side(new_pp, best_a)
                        if new_s < best_s:
                            best_p = new_pp
                            best_s = new_s
            
            # Phase 2: Optimize angles with fixed positions
            for iteration in range(30):
                if time.time() - t_start > time_limit:
                    break
                
                lr = 0.5
                for _ in range(5):
                    if time.time() - t_start > time_limit:
                        break
                    
                    def obj_ang(a_flat):
                        s, _, _, _ = compute_min_outer_side(best_p, a_flat)
                        return s
                    
                    eps = 0.01
                    grad = np.zeros(N)
                    f0 = obj_ang(best_a)
                    for k in range(N):
                        aa = best_a.copy()
                        aa[k] += eps
                        fp = obj_ang(aa)
                        grad[k] = (fp - f0) / eps
                    
                    new_a = best_a - lr * grad
                    new_a = np.clip(new_a, -180, 180)
                    
                    if is_valid_configuration(best_p, new_a):
                        new_s, _, _, _ = compute_min_outer_side(best_p, new_a)
                        if new_s < best_s:
                            best_a = new_a
                            best_s = new_s
            
            return best_s, best_p, best_a
        
        refine_time = min(remaining_time * 0.5, 25.0)
        r_side, r_pos, r_ang = refine_alternating(best_positions, best_angles, refine_time)
        
        if r_side < best_side:
            best_side = r_side
            best_positions = r_pos
            best_angles = r_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # === FINAL POLISH: Fine coordinate descent ===
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
        def final_polish(pos, ang, time_limit):
            t_start = time.time()
            best_p = pos.copy()
            best_a = ang.copy()
            best_s, _, _, _ = compute_min_outer_side(best_p, best_a)
            
            step = 0.01
            min_step = 0.0003
            
            while time.time() - t_start < time_limit and step > min_step:
                improved = False
                for idx in range(N):
                    if time.time() - t_start > time_limit:
                        return best_s, best_p, best_a
                    
                    for dim in range(3):
                        for direction in [1, -1]:
                            new_p = best_p.copy()
                            new_a = best_a.copy()
                            
                            if dim == 0:
                                new_p[idx, 0] += direction * step
                            elif dim == 1:
                                new_p[idx, 1] += direction * step
                            else:
                                new_a[idx] += direction * step
                            
                            if not is_valid_configuration(new_p, new_a):
                                continue
                            
                            new_s, _, _, _ = compute_min_outer_side(new_p, new_a)
                            
                            if new_s < best_s - 1e-10:
                                best_p = new_p
                                best_a = new_a
                                best_s = new_s
                                improved = True
                                break
                        if time.time() - t_start > time_limit:
                            return best_s, best_p, best_a
                    if improved:
                        break
                if not improved:
                    step *= 0.65
            
            return best_s, best_p, best_a
        
        polish_time = min(remaining_time * 0.7, 15.0)
        p_side, p_pos, p_ang = final_polish(best_positions, best_angles, polish_time)
        
        if p_side < best_side:
            best_side = p_side
            best_positions = p_pos
            best_angles = p_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    # Safety: ensure we return before deadline
    elapsed = time.time() - t0
    if elapsed > DEADLINE:
        # Emergency fallback
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        inner = []
        for i in range(N):
            inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        center = (float(best_cx), float(best_cy))
        side = float(best_side)
        angle = float(best_phi)
    
    return inner, center, side, angle
