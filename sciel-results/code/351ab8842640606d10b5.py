import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 148.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _hex_distance(p1, a1, p2, a2):
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    def support(a, theta):
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)
    
    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)
    
    return d - h1 - h2

def _softmin(values, beta=50.0):
    m = np.min(values)
    if m < -1000:
        return m
    shifted = values - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def _objective(params, outer_angle_deg):
    n = N
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    min_dist = float('inf')
    all_dists = []
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            all_dists.append(dist)
            if dist < min_dist:
                min_dist = dist
    
    if min_dist < -0.01:
        return 1000.0 + min_dist * 100
    
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def _build_positions(layout_idx):
    sqrt3 = _SQRT3
    scale = 1.0
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _nelder_mead_polish(x0, outer_angle_deg, max_iter=2000, tol=1e-14):
    """Nelder-Mead optimization for local polish."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x_trial = x0.copy()
        step = 0.01 if abs(x0[i]) > 0.1 else 0.001
        x_trial[i] += step
        simplex.append(x_trial)
    
    f = [_objective(s, outer_angle_deg) for s in simplex]
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        idx = np.argsort(f)
        simplex = [simplex[i] for i in idx]
        f = [f[i] for i in idx]
        
        if abs(f[-1] - f[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + (centroid - simplex[-1])
        fr = _objective(xr, outer_angle_deg)
        
        if fr < f[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = _objective(xe, outer_angle_deg)
            if fe < fr:
                simplex[-1] = xe
                f[-1] = fe
            else:
                simplex[-1] = xr
                f[-1] = fr
        elif fr < f[-2]:
            simplex[-1] = xr
            f[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = _objective(xc, outer_angle_deg)
            if fc < f[-1]:
                simplex[-1] = xc
                f[-1] = fc
            else:
                for i in range(1, n+1):
                    simplex[i] = 0.5 * (simplex[0] + simplex[i])
                    f[i] = _objective(simplex[i], outer_angle_deg)
    
    best_idx = np.argmin(f)
    return simplex[best_idx], f[best_idx]

def construct():
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_params = None
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        for outer_angle_deg in np.arange(0, 30.0, 0.25):
            if time.time() > _DEADLINE - 10.0:
                break
            
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
            best_params = []
            for pos in positions:
                best_params.append(pos[0])
                best_params.append(pos[1])
                best_params.append(math.radians(inner_angle))
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Phase 1: Nelder-Mead polish on best layout
    if time.time() < _DEADLINE - 20.0:
        x0 = best_params[:]
        
        outer_angles_to_try = list(np.arange(max(0, best_outer_angle - 5), min(30, best_outer_angle + 5), 1.0))
        
        for outer_angle_deg in outer_angles_to_try:
            if time.time() > _DEADLINE - 20.0:
                break
            
            try:
                res_x, res_f = _nelder_mead_polish(x0, outer_angle_deg, max_iter=1000, tol=1e-12)
                
                if res_f < best_side and res_f < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res_x[3*i], res_x[3*i+1]))
                        new_angles.append(math.degrees(res_x[3*i+2]))
                    
                    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_outer_angle = outer_angle_deg
                        best_params = res_x[:]
            except Exception:
                pass
    
    # Phase 2: L-BFGS-B with smooth objective
    if time.time() < _DEADLINE - 10.0:
        try:
            x0 = best_params[:]
            
            for polish_iter in range(3):
                if time.time() > _DEADLINE - 10.0:
                    break
                
                res = minimize(_objective, x0, args=(best_outer_angle,),
                              method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if res.fun < best_side and res.fun < 100:
                    new_positions = []
                    new_angles = []
                    for i in range(N):
                        new_positions.append((res.x[3*i], res.x[3*i+1]))
                        new_angles.append(math.degrees(res.x[3*i+2]))
                    
                    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                    
                    if side < best_side:
                        best_side = side
                        best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                        best_center = (cx, cy)
                        best_params = res.x[:]
                    else:
                        break
                else:
                    break
        except Exception:
            pass
    
    # Phase 3: Final high-precision Nelder-Mead polish
    if time.time() < _DEADLINE - 5.0 and best_params is not None:
        try:
            res_x, res_f = _nelder_mead_polish(best_params, best_outer_angle, max_iter=2000, tol=1e-14)
            
            if res_f < best_side and res_f < 100:
                new_positions = []
                new_angles = []
                for i in range(N):
                    new_positions.append((res_x[3*i], res_x[3*i+1]))
                    new_angles.append(math.degrees(res_x[3*i+2]))
                
                all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, best_outer_angle)
                
                if side < best_side:
                    best_side = side
                    best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                    best_center = (cx, cy)
        except Exception:
            pass
    
    return best_inner, best_center, best_side, best_outer_angle
