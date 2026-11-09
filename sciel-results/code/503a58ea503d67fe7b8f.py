import numpy as np
import time
from scipy.optimize import minimize

N = 11
_SQRT3 = np.sqrt(3.0)
_TOL = 1e-9

def _hex_verts(cx, cy, angle_deg):
    a = np.deg2rad(angle_deg)
    th = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + a
    return np.column_stack([np.cos(th), np.sin(th)]) + np.array([cx, cy])

def _point_in_convex(p, P):
    s = 0
    for i in range(6):
        e = P[(i + 1) % 6] - P[i]
        cross = e[0] * (p[1] - P[i][1]) - e[1] * (p[0] - P[i][0])
        if cross > _TOL:
            s |= 1
        elif cross < -_TOL:
            s |= 2
    return not (s == 3)

def _min_dist_convex(P, Q):
    best = float('inf')
    for i in range(6):
        a1 = P[i]; a2 = P[(i + 1) % 6]
        d1 = a2 - a1
        for j in range(6):
            b1 = Q[j]; b2 = Q[(j + 1) % 6]
            d2 = b2 - b1
            r = a1 - b1
            a = np.dot(d1, d1); c = np.dot(d2, d2)
            b = np.dot(d1, d2); d = np.dot(d1, r); e = np.dot(d2, r)
            denom = a * c - b * b
            s = t = 0.0
            if denom <= _TOL:
                if a <= _TOL:
                    s = 0.0
                else:
                    s = np.clip(-d / a, 0.0, 1.0)
                t = np.clip((b * s + e) / c if c > _TOL else 0.0, 0.0, 1.0)
            else:
                s = np.clip((b * e - c * d) / denom, 0.0, 1.0)
                t = (b * s + e) / c
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-d / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - d) / a, 0.0, 1.0)
            ca = a1 + s * d1
            cb = b1 + t * d2
            dd = ca - cb
            dist = np.dot(dd, dd)
            if dist < best:
                best = dist
    best = np.sqrt(best)
    for p in P:
        if _point_in_convex(p, Q):
            return 0.0
    for p in Q:
        if _point_in_convex(p, P):
            return 0.0
    return best

def _containment_dist(P, outer):
    mn = float('inf')
    for i in range(6):
        v1 = outer[i]; v2 = outer[(i + 1) % 6]
        n = np.array([-(v2[1] - v1[1]), v2[0] - v1[0]])
        nn = np.linalg.norm(n)
        if nn > _TOL:
            n = n / nn
        else:
            n = np.zeros(2)
        if np.dot(n, v1) > 0:
            n = -n
        for p in P:
            d = np.dot(p - v1, n)
            if d < mn:
                mn = d
    return mn

def _honeycomb_init():
    """Compact hexagonal tiling of 11 flat-top unit hexagons, centered."""
    S = _SQRT3
    # Use a proper hex grid layout
    # Rows of hexagons in a honeycomb pattern
    pos = []
    # Pattern: 2-3-3-2-1 or similar to get 11
    # Let's use axial coordinates
    # Center at (0,0), then 6 neighbors, then 4 more
    hex_dirs = [(S, 0), (S/2, 1.5), (-S/2, 1.5), (-S, 0), (-S/2, -1.5), (S/2, -1.5)]
    # Actually for flat-top hexagons with side 1:
    # Adjacent centers are at distance sqrt(3) in 6 directions
    # Directions: 0, 60, 120, 180, 240, 300 degrees
    # Distance between adjacent centers = 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)
    
    # 1 center
    pos.append([0.0, 0.0])
    # 6 ring
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pos.append([S * np.cos(angle), S * np.sin(angle)])
    # 4 more at distance 2*sqrt(3) in 4 directions
    for k in [0, 90, 180, 270]:
        angle = np.deg2rad(k)
        pos.append([2 * S * np.cos(angle), 2 * S * np.sin(angle)])
    
    pos = np.array(pos[:N])
    cx = (pos[:, 0].max() + pos[:, 0].min()) / 2
    cy = (pos[:, 1].max() + pos[:, 1].min()) / 2
    return pos - np.array([cx, cy])

def construct():
    start_time = time.time()
    deadline = start_time + 142.0
    
    base = _honeycomb_init()
    rng = np.random.default_rng(777)
    
    def get_verts(x):
        """Get all hexagon vertices and outer hexagon from parameter vector."""
        pos = x[:3*N].reshape(N, 3)[:, :2]
        ang = x[:3*N].reshape(N, 3)[:, 2]
        side = x[-2]
        oang = x[-1]
        
        hexes = []
        for i in range(N):
            hexes.append(_hex_verts(pos[i, 0], pos[i, 1], ang[i]))
        outer = _hex_verts(0.0, 0.0, oang) * side
        return hexes, outer, pos, ang, side, oang
    
    def penalty(x, weight=100.0):
        hexes, outer, pos, ang, side, oang = get_verts(x)
        
        pen = side
        # Containment penalty
        for i in range(N):
            d = _containment_dist(hexes[i], outer)
            if d < 0:
                pen += weight * d * d
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                d = _min_dist_convex(hexes[i], hexes[j])
                if d < 0:
                    pen += weight * d * d
        return pen
    
    def is_valid(x, tol=1e-6):
        hexes, outer, pos, ang, side, oang = get_verts(x)
        for i in range(N):
            if _containment_dist(hexes[i], outer) < -tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                if _min_dist_convex(hexes[i], hexes[j]) < -tol:
                    return False
        return True
    
    # Build initial x
    x0 = np.zeros(3 * N + 2)
    for i in range(N):
        x0[3*i] = base[i, 0]
        x0[3*i+1] = base[i, 1]
        x0[3*i+2] = 0.0
    # Estimate initial side
    hexes0 = [_hex_verts(base[i,0], base[i,1], 0) for i in range(N)]
    # Find bounding hexagon side
    max_r = 0
    for h in hexes0:
        for v in h:
            max_r = max(max_r, np.linalg.norm(v))
    x0[-2] = max_r + 0.5
    x0[-1] = 0.0
    
    best_x = x0.copy()
    best_pen = penalty(best_x)
    best_side = best_x[-2] if is_valid(best_x) else float('inf')
    
    # Phase 1: Simulated annealing to find valid tight packing
    T = 0.5
    T_min = 0.0005
    cooling = 0.997
    step = 0.3
    n_iter = 0
    
    while time.time() < deadline - 20.0 and n_iter < 80000:
        n_iter += 1
        x = best_x.copy()
        
        # Choose what to perturb
        r = rng.random()
        if r < 0.7:
            # Perturb a position
            idx = rng.randint(0, N)
            dim = rng.randint(0, 2)
            x[3*idx + dim] += rng.normal(0, step)
        elif r < 0.85:
            # Perturb an angle
            idx = rng.randint(0, N)
            x[3*idx + 2] += rng.normal(0, step * 5)
            x[3*idx + 2] = ((x[3*idx + 2] + 180) % 360) - 180
        elif r < 0.92:
            # Perturb outer angle
            x[-1] += rng.normal(0, step * 3)
            x[-1] = ((x[-1] + 180) % 360) - 180
        else:
            # Perturb side
            x[-2] += rng.normal(0, step * 0.5)
            x[-2] = max(1.0, x[-2])
        
        new_pen = penalty(x)
        delta = new_pen - best_pen
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            best_x = x
            best_pen = new_pen
            if is_valid(best_x):
                s = best_x[-2]
                if s < best_side:
                    best_side = s
        
        T = max(T * cooling, T_min)
    
    # Phase 2: Local refinement with Nelder-Mead
    if is_valid(best_x):
        try:
            res = minimize(
                penalty,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            if is_valid(res.x) and res.x[-2] < best_side:
                best_x = res.x
                best_side = res.x[-2]
        except:
            pass
    
    # Phase 3: Try multiple random restarts if time permits
    remaining = deadline - 5.0 - time.time()
    if remaining > 15.0:
        for restart in range(5):
            if time.time() > remaining + start_time - 5.0:
                break
            x_r = best_x.copy()
            for i in range(N):
                x_r[3*i] += rng.normal(0, 1.0)
                x_r[3*i+1] += rng.normal(0, 1.0)
                x_r[3*i+2] += rng.uniform(-30, 30)
                x_r[3*i+2] = ((x_r[3*i+2] + 180) % 360) - 180
            x_r[-1] += rng.uniform(-15, 15)
            x_r[-1] = ((x_r[-1] + 180) % 360) - 180
            
            # Quick SA
            T_r = 0.3
            for it in range(3000):
                if time.time() > deadline - 5.0:
                    break
                xr = x_r.copy()
                idx = rng.randint(0, N)
                dim = rng.randint(0, 3)
                if dim == 2:
                    xr[3*idx+2] += rng.normal(0, 0.5)
                    xr[3*idx+2] = ((xr[3*idx+2] + 180) % 360) - 180
                else:
                    xr[3*idx+dim] += rng.normal(0, 0.2)
                new_p = penalty(xr)
                delta = new_p - penalty(x_r)
                if delta < 0 or rng.random() < np.exp(-delta / max(T_r, 1e-12)):
                    x_r = xr
                T_r = max(T_r * 0.998, 0.001)
            
            if is_valid(x_r) and x_r[-2] < best_side:
                best_x = x_r
                best_side = x_r[-2]
    
    # Final answer
    if not is_valid(best_x):
        # Fallback: use honeycomb with estimated side
        hexes_fb = [_hex_verts(base[i,0], base[i,1], 0) for i in range(N)]
        max_r = 0
        for h in hexes_fb:
            for v in h:
                max_r = max(max_r, np.linalg.norm(v))
        side_fb = max_r + 0.3
        inner = [[float(base[i,0]), float(base[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(side_fb), 0.0
    
    pos = best_x[:3*N].reshape(N, 3)[:, :2]
    ang = best_x[:3*N].reshape(N, 3)[:, 2]
    side = best_x[-2]
    oang = best_x[-1]
    
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(side), float(oang)
