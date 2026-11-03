import math
import time
import random
import numpy as np

N = 12
DEADLINE = 140.0

# Precompute unit hexagon vertices (pointy-top, angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for each edge (outward)
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# Fast arrays
_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

def construct():
    t0 = time.time()
    sqrt3 = math.sqrt(3)
    
    # --- Geometry helpers ---
    def rotate_vertices(px, py, ang_rad):
        """Rotate unit hex vertices by ang_rad, translate to (px, py)"""
        c = math.cos(ang_rad)
        s = math.sin(ang_rad)
        return px + UX * c - UY * s, py + UX * s + UY * c
    
    def hex_disjoint(h1, h2):
        """h = (px, py, ang_rad). SAT for two unit hexagons."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        if dx*dx + dy*dy > 4.6:
            return True
        a1, a2 = h1[2], h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + UX * c1 - UY * s1
        v1y = h1[1] + UX * s1 + UY * c1
        v2x = h2[0] + UX * c2 - UY * s2
        v2y = h2[1] + UX * s2 + UY * c2
        en1x = ENX * c1 - ENY * s1
        en1y = ENX * s1 + ENY * c1
        en2x = ENX * c2 - ENY * s2
        en2y = ENX * s2 + ENY * c2
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
    
    def compute_outer_side(positions, angles_deg):
        """Find min enclosing regular hexagon side length, its rotation, and center."""
        angles_rad = np.radians(angles_deg)
        all_vx = np.zeros((N, 6))
        all_vy = np.zeros((N, 6))
        for i in range(N):
            px, py = positions[i]
            a = angles_rad[i]
            c, s = math.cos(a), math.sin(a)
            all_vx[i] = px + UX * c - UY * s
            all_vy[i] = py + UX * s + UY * c
        verts_flat = np.column_stack((all_vx.ravel(), all_vy.ravel()))
        center = verts_flat.mean(axis=0)
        offsets = verts_flat - center
        
        best_side = float('inf')
        best_phi = 0.0
        # Scan rotation 0..60 degrees (hex symmetry)
        for phi_deg in range(0, 61, 1):
            phi = math.radians(phi_deg)
            # Outer hex edge normals at rotation phi
            na = phi + math.pi/6 + np.arange(6) * math.pi/3
            nx = np.cos(na)
            ny = np.sin(na)
            proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
            max_proj = np.abs(proj).max(axis=0)
            side_val = (2.0 / sqrt3) * max_proj.max()
            if side_val < best_side:
                best_side = side_val
                best_phi = phi_deg
        
        # Refine
        if best_side < float('inf'):
            for _ in range(3):
                best_side = float('inf')
                best_phi = 0.0
                for phi_deg in np.linspace(0, 60, 601):
                    phi = math.radians(phi_deg)
                    na = phi + math.pi/6 + np.arange(6) * math.pi/3
                    nx = np.cos(na)
                    ny = np.sin(na)
                    proj = offsets[:, 0] * nx[None, :] + offsets[:, 1] * ny[None, :]
                    max_proj = np.abs(proj).max(axis=0)
                    side_val = (2.0 / sqrt3) * max_proj.max()
                    if side_val < best_side:
                        best_side = side_val
                        best_phi = phi_deg
        return best_side, best_phi, center[0], center[1]
    
    def valid_config(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                if not hex_disjoint((positions[i,0], positions[i,1], angles_rad[i]),
                                     (positions[j,0], positions[j,1], angles_rad[j])):
                    return False
        return True
    
    # --- Initial configurations ---
    def make_honeycomb():
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        coords = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)]
        pos = np.zeros((N,2))
        ang = np.zeros(N)
        for idx,(i,j) in enumerate(coords):
            pos[idx] = i*a1 + j*a2
        return pos, ang
    
    def make_ring():
        """6 inner + 6 outer ring"""
        pos = np.zeros((N,2))
        ang = np.zeros(N)
        # Inner ring: 6 hexagons around center at radius r1
        r1 = sqrt3 + 0.1
        for i in range(6):
            a = math.radians(60*i + 30)
            pos[i,0] = r1 * math.cos(a)
            pos[i,1] = r1 * math.sin(a)
        # Outer ring: 6 hexagons at radius r2
        r2 = 2*sqrt3 + 0.2
        for i in range(6):
            a = math.radians(60*i)
            pos[6+i,0] = r2 * math.cos(a)
            pos[6+i,1] = r2 * math.sin(a)
        return pos, ang
    
    def make_grid():
        """3x4-ish rectangular grid of hexagons"""
        pos = np.zeros((N,2))
        ang = np.zeros(N)
        dx = sqrt3
        dy = 1.5
        for i in range(3):
            for j in range(4):
                idx = i*4+j
                pos[idx,0] = j*dx + (i%2)*dx/2
                pos[idx,1] = i*dy
        return pos, ang
    
    # --- New approach: Simulated Annealing with pair-wise repulsion + direct objective ---
    def sa_optimize(init_pos, init_ang, time_budget):
        """
        New SA approach: 
        - Use a penalty-based objective that combines outer hex side + overlap penalty
        - This allows the optimizer to escape deep local minima
        - Multi-start with different initial configs
        """
        pos = init_pos.copy()
        ang = init_ang.copy()
        
        def objective(p, a):
            """Penalty objective: outer_side + penalty * overlap_amount"""
            angles_rad = np.radians(a)
            # Check all pairs for overlap
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if not hex_disjoint((p[i,0], p[i,1], angles_rad[i]),
                                         (p[j,0], p[j,1], angles_rad[j])):
                        # Estimate overlap via distance (rough)
                        dx = p[i,0] - p[j,0]
                        dy = p[i,1] - p[j,1]
                        d = math.sqrt(dx*dx + dy*dy)
                        overlap_penalty += max(0, 2*sqrt3 - d) ** 2
            if overlap_penalty > 1e-6:
                return 1000.0 + overlap_penalty
            side, _, _, _ = compute_outer_side(p, a)
            return side
        
        side, phi, cx, cy = compute_outer_side(pos, ang)
        if valid_config(pos, ang):
            cur_obj = side
        else:
            cur_obj = objective(pos, ang)
        
        best_obj = cur_obj
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)
        
        T = 0.3
        T_min = 0.001
        cooling = 0.99995
        step = 0.4
        
        t_start = time.time()
        step_count = 0
        accepts = 0
        total = 0
        
        while time.time() - t0 < (DEADLINE - time_budget) and step_count < 300000:
            step_count += 1
            total += 1
            
            idx = random.randint(0, N-1)
            new_pos = pos.copy()
            new_ang = ang.copy()
            
            new_pos[idx,0] += random.gauss(0, step)
            new_pos[idx,1] += random.gauss(0, step)
            new_ang[idx] += random.gauss(0, 8.0)
            
            # Quick validity check
            angles_rad_new = np.radians(new_ang)
            valid = True
            hm = (new_pos[idx,0], new_pos[idx,1], angles_rad_new[idx])
            for j in range(N):
                if j == idx: continue
                ho = (new_pos[j,0], new_pos[j,1], angles_rad_new[j])
                if not hex_disjoint(hm, ho):
                    valid = False
                    break
            
            if valid:
                new_side, new_phi, new_cx, new_cy = compute_outer_side(new_pos, new_ang)
                new_obj = new_side
            else:
                new_obj = objective(new_pos, new_ang)
            
            if new_obj < cur_obj:
                pos = new_pos
                ang = new_ang
                cur_obj = new_obj
                accepts += 1
                if valid:
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_pos = new_pos.copy()
                        best_ang = new_ang.copy()
                        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)
            else:
                delta = new_obj - cur_obj
                if delta < 0: delta = 0
                if random.random() < math.exp(-delta / T):
                    pos = new_pos
                    ang = new_ang
                    cur_obj = new_obj
                    accepts += 1
                    if valid and new_obj < best_obj:
                        best_obj = new_obj
                        best_pos = new_pos.copy()
                        best_ang = new_ang.copy()
                        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)
            
            # Adaptive step
            if total % 1000 == 0:
                acc_rate = accepts / total
                if acc_rate < 0.15:
                    step *= 0.85
                elif acc_rate > 0.5:
                    step *= 1.15
                step = max(0.01, min(1.5, step))
                accepts = 0
                total = 0
            
            T *= cooling
            if T < T_min: T = T_min
            step *= 0.99999
        
        return best_side, best_pos, best_ang, best_phi, best_cx, best_cy
    
    # --- Multi-start SA ---
    inits = [make_honeycomb(), make_ring(), make_grid()]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    time_per_start = (DEADLINE * 0.7) / len(inits)
    
    for init_pos, init_ang in inits:
        if time.time() - t0 > DEADLINE - 20:
            break
        s, p, a, phi, cx, cy = sa_optimize(init_pos, init_ang, time_per_start)
        if s < best_side:
            best_side = s
            best_positions = p
            best_angles = a
            best_phi = phi
            best_cx = cx
            best_cy = cy
    
    # --- Final polish: local Nelder-Mead on the best config ---
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10 and best_positions is not None:
        def nm_objective(params):
            p = np.zeros((N,2))
            a = np.zeros(N)
            for i in range(N):
                p[i,0] = params[3*i]
                p[i,1] = params[3*i+1]
                a[i] = params[3*i+2]
            angles_rad = np.radians(a)
            for i in range(N):
                for j in range(i+1, N):
                    if not hex_disjoint((p[i,0],p[i,1],angles_rad[i]),
                                         (p[j,0],p[j,1],angles_rad[j])):
                        return 1e6
            side, _, _, _ = compute_outer_side(p, a)
            return side
        
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = best_positions[i,0]
            x0[3*i+1] = best_positions[i,1]
            x0[3*i+2] = best_angles[i]
        
        try:
            from scipy.optimize import minimize
            res = minimize(nm_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_side and res.fun < 1e5:
                np_ = np.zeros((N,2))
                na = np.zeros(N)
                for i in range(N):
                    np_[i,0] = res.x[3*i]
                    np_[i,1] = res.x[3*i+1]
                    na[i] = res.x[3*i+2]
                if valid_config(np_, na):
                    s, phi, cx, cy = compute_outer_side(np_, na)
                    if s < best_side:
                        best_side = s
                        best_positions = np_
                        best_angles = na
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass
    
    # --- Final verification ---
    if best_positions is None or not valid_config(best_positions, best_angles):
        pos, ang = make_honeycomb()
        best_positions = pos
        best_angles = ang
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i,0]), float(best_positions[i,1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
