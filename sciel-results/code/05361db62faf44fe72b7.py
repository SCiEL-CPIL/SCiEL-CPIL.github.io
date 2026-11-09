"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = 145.0
_C30 = math.cos(math.radians(30))

def _hex_inradius(s):
    return s * _C30

def _point_in_hex(px, py, cx, cy, s, rot):
    dx = px - cx
    dy = py - cy
    ca, sa = math.cos(-rot), math.sin(-rot)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r_in = _hex_inradius(s)
    d1 = abs(_C30 * lx + 0.5 * ly)
    d2 = abs(ly)
    d3 = abs(-_C30 * lx + 0.5 * ly)
    return max(d1, d2, d3) <= r_in + 1e-9

def _hex_verts(cx, cy, s, rot):
    vs = np.empty((6, 2))
    for k in range(6):
        a = math.radians(60.0 * k) + rot
        vs[k, 0] = cx + s * math.cos(a)
        vs[k, 1] = cy + s * math.sin(a)
    return vs

def _min_dist_poly(A, B):
    """Compute minimum distance between two convex polygons. Returns 0.0 if overlapping."""
    nA = len(A)
    nB = len(B)
    
    def point_in_poly(pt, poly):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            nx = -(y2 - y1)
            ny = (x2 - x1)
            d = nx * (pt[0] - x1) + ny * (pt[1] - y1)
            if d < -1e-9:
                return False
        return True
    
    for v in A:
        if point_in_poly(v, B):
            return 0.0
    for v in B:
        if point_in_poly(v, A):
            return 0.0
    
    min_d = float('inf')
    for i in range(nA):
        ax1, ay1 = A[i]
        ax2, ay2 = A[(i+1) % nA]
        for j in range(nB):
            bx1, by1 = B[j]
            bx2, by2 = B[(j+1) % nB]
            
            def pt_seg_dist(px, py, x1, y1, x2, y2):
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    return math.sqrt((px-x1)**2 + (py-y1)**2)
                t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
                t = max(0.0, min(1.0, t))
                qx = x1 + t*dx
                qy = y1 + t*dy
                return math.sqrt((px-qx)**2 + (py-qy)**2)
            
            d = min(pt_seg_dist(ax1, ay1, bx1, by1, bx2, by2),
                    pt_seg_dist(ax2, ay2, bx1, by1, bx2, by2),
                    pt_seg_dist(bx1, by1, ax1, ay1, ax2, ay2),
                    pt_seg_dist(bx2, by2, ax1, ay1, ax2, ay2))
            if d < min_d:
                min_d = d
    return min_d

def _is_valid_config(inner, center, side, angle):
    cx, cy = center
    s_out = side
    r_out = _hex_inradius(s_out)
    
    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            if not _point_in_hex(v[0], v[1], cx, cy, s_out, angle):
                return False

    for i in range(len(inner)):
        xi, yi, ai_deg = inner[i]
        ai = math.radians(ai_deg)
        vi = _hex_verts(xi, yi, 1.0, ai)
        for j in range(i + 1, len(inner)):
            xj, yj, aj_deg = inner[j]
            aj = math.radians(aj_deg)
            vj = _hex_verts(xj, yj, 1.0, aj)
            
            dc = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dc > 2.0:
                continue
            
            if _min_dist_poly(vi, vj) < 1e-7:
                return False
    return True

def _generate_lattice_positions(max_radius, step=0.1):
    """Generate a dense triangular lattice of candidate positions."""
    positions = []
    r_max = int(math.ceil(max_radius / step)) + 1
    for i in range(-r_max, r_max + 1):
        for j in range(-r_max, r_max + 1):
            x = i * step
            y = j * step * math.sqrt(3) / 2
            if math.sqrt(x*x + y*y) <= max_radius:
                positions.append((x, y))
    return positions

def _construct():
    t_start = time.time()
    n = N

    def _remaining():
        return _DEADLINE - (time.time() - t_start)

    # Initial honeycomb configuration
    init_pos = []
    init_pos.append((0.0, 0.0, 0.0))
    for k in range(6):
        a = math.radians(60.0 * k + 30.0)
        init_pos.append((2.0*math.cos(a), 2.0*math.sin(a), 0.0))
    for k in range(4):
        a = math.radians(60.0 * k + 30.0)
        init_pos.append((4.0*math.cos(a), 4.0*math.sin(a), 0.0))
    init_pos = init_pos[:n]
    
    # Compute initial outer side length
    max_r = 0.0
    for (xi, yi, ai) in init_pos:
        verts = _hex_verts(xi, yi, 1.0, ai)
        for v in verts:
            r = math.sqrt(v[0]**2 + v[1]**2)
            if r > max_r:
                max_r = r
    
    s_init = max_r / _C30
    s_init *= 1.01
    
    # Generate lattice positions for discrete search
    lattice_positions = _generate_lattice_positions(max_r * 1.1, step=0.15)
    
    # Angular candidates (0 to 180 degrees in 30-degree steps)
    angle_candidates = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    best_val = float('inf')
    best_config = None
    
    # Phase 1: Discrete search over lattice positions and angles
    for trial in range(12):
        if _remaining() < 20.0:
            break
            
        # Randomly sample a subset of lattice positions
        rng = np.random.RandomState(42 + trial)
        if len(lattice_positions) > 100:
            sampled_idx = rng.choice(len(lattice_positions), size=100, replace=False)
            sampled_positions = [lattice_positions[i] for i in sampled_idx]
        else:
            sampled_positions = lattice_positions
        
        # Try different combinations
        for pos_idx in range(min(50, len(sampled_positions))):
            if _remaining() < 15.0:
                break
                
            # Assign positions greedily
            assigned = [False] * n
            pos_assignment = [None] * n
            
            # Start from center, then assign others
            center_pos = (0.0, 0.0)
            pos_assignment[0] = center_pos
            assigned[0] = True
            
            for i in range(1, n):
                best_pos = None
                best_min_dist = -1.0
                
                for px, py in sampled_positions:
                    # Check if this position is valid (no overlap with already assigned)
                    valid = True
                    min_dist_to_assigned = float('inf')
                    
                    for j in range(i):
                        if assigned[j]:
                            dx = px - pos_assignment[j][0]
                            dy = py - pos_assignment[j][1]
                            d = math.sqrt(dx*dx + dy*dy)
                            if d < 1.95:  # Too close, potential overlap
                                valid = False
                                break
                            min_dist_to_assigned = min(min_dist_to_assigned, d)
                    
                    if valid and min_dist_to_assigned > best_min_dist:
                        best_min_dist = min_dist_to_assigned
                        best_pos = (px, py)
                
                if best_pos is not None:
                    pos_assignment[i] = best_pos
                    assigned[i] = True
                else:
                    # Fallback to initial position
                    pos_assignment[i] = init_pos[i][:2]
                    assigned[i] = True
            
            # Compute required outer side length for this configuration
            max_r_config = 0.0
            for i in range(n):
                xi, yi = pos_assignment[i]
                verts = _hex_verts(xi, yi, 1.0, 0.0)
                for v in verts:
                    r = math.sqrt(v[0]**2 + v[1]**2)
                    if r > max_r_config:
                        max_r_config = r
            
            S_config = max_r_config / _C30 * 1.005
            
            # Quick validity check
            inner_test = [[pos_assignment[i][0], pos_assignment[i][1], 0.0] for i in range(n)]
            if _is_valid_config(inner_test, (0.0, 0.0), S_config, 0.0):
                val = S_config
                if val < best_val:
                    best_val = val
                    best_config = (inner_test, S_config)
    
    # Phase 2: Local refinement using continuous optimization
    if best_config is not None:
        inner0, S0 = best_config
    else:
        inner0 = [[p[0], p[1], p[2]] for p in init_pos]
        S0 = s_init
    
    p0 = np.zeros(3*n + 1)
    for i in range(n):
        p0[3*i] = inner0[i][0]
        p0[3*i+1] = inner0[i][1]
        p0[3*i+2] = math.radians(inner0[i][2])
    p0[3*n] = S0
    
    def objective(p):
        x = p[0:3*n:3]
        y = p[1:3*n:3]
        angles = p[2:3*n:3]
        S = p[3*n]
        
        if S < 1.0:
            return 1e6
            
        pen = 0.0
        r_out = _hex_inradius(S)
        
        # Boundary penalty
        for i in range(n):
            verts = _hex_verts(x[i], y[i], 1.0, angles[i])
            for v in verts:
                dx = v[0]
                dy = v[1]
                d1 = abs(_C30 * dx + 0.5 * dy)
                d2 = abs(dy)
                d3 = abs(-_C30 * dx + 0.5 * dy)
                d_max = max(d1, d2, d3)
                if d_max > r_out:
                    pen += (d_max - r_out)**2
                    
        # Overlap penalty using smooth min distance
        for i in range(n):
            vi = _hex_verts(x[i], y[i], 1.0, angles[i])
            for j in range(i+1, n):
                dc = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dc > 2.5:
                    continue
                vj = _hex_verts(x[j], y[j], 1.0, angles[j])
                d = _min_dist_poly(vi, vj)
                if d < 0.1:
                    pen += 50.0 * (0.1 - d)**2
        
        return 0.5 * S * S + pen

    # Run local optimization
    for trial in range(6):
        if _remaining() < 10.0:
            break
            
        p_init = p0.copy()
        if trial > 0:
            noise = 0.1
            p_init[0:3*n] += np.random.randn(3*n) * noise
            p_init[3*n] = S0 * (1.0 + np.random.rand()*0.03)
            
        try:
            res = minimize(objective, p_init, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                x = res.x[0:3*n:3]
                y = res.x[1:3*n:3]
                angles = res.x[2:3*n:3]
                S = res.x[3*n]
                inner_refined = [[float(x[i]), float(y[i]), float(math.degrees(angles[i]))] for i in range(n)]
                
                if _is_valid_config(inner_refined, (0.0, 0.0), S, 0.0):
                    best_config = (inner_refined, S)
        except Exception:
            pass
    
    # Return best result
    if best_config is not None:
        inner, S = best_config
        return inner, (0.0, 0.0), float(S), 0.0
    else:
        inner_init = [[float(p[0]), float(p[1]), float(math.degrees(p[2]))] for p in init_pos]
        return inner_init, (0.0, 0.0), float(s_init), 0.0

def construct():
    return _construct()
