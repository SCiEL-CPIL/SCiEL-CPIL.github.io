"""Improved packing: direct outer-size optimization with smooth constraints and multi-restart local search."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 115.0  # safety margin before 120s kill

def construct():
    start_time = time.time()
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.radians(np.array([0, 60, 120, 180, 240, 300]))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        dx = px - cx
        dy = py - cy
        a = np.radians(angle_deg)
        cos_a = np.cos(-a)
        sin_a = np.sin(-a)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        apothem = side * np.sqrt(3) / 2
        for na in np.radians(np.array([30, 90, 150, 210, 270, 330])):
            dist = lx * np.cos(na) + ly * np.sin(na)
            if dist > apothem + 1e-9:
                return False
        return True

    def sat_disjoint(v1, v2, eps=1e-9):
        """Check if two convex hexagons are disjoint using SAT."""
        n1, n2 = len(v1), len(v2)
        axes = []
        for i in range(n1):
            p1, p2 = v1[i], v1[(i+1) % n1]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / norm)
        for i in range(n2):
            p1, p2 = v2[i], v2[(i+1) % n2]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / norm)
        
        for axis in axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                return True
        return False

    def compute_penalties(params):
        """
        Compute objective: outer side + penalties for violations.
        params: [x1,y1,a1, ..., x12,y12,a12, ox,oy,os,oa]
        """
        n = N
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(n)]
        ox, oy = params[3*n], params[3*n+1]
        os = max(params[3*n+2], 0.5)
        oa = params[3*n+3]
        
        penalty = 0.0
        penalty_weight = 50.0
        
        # Check inside outer
        for i in range(n):
            cx, cy, ang = inner[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, ox, oy, os, oa):
                    penalty += 1.0
        
        # Check pairwise disjoint
        for i in range(n):
            for j in range(i+1, n):
                ci = (inner[i][0], inner[i][1])
                cj = (inner[j][0], inner[j][1])
                v1 = hex_vertices(ci[0], ci[1], 1.0, inner[i][2])
                v2 = hex_vertices(cj[0], cj[1], 1.0, inner[j][2])
                if not sat_disjoint(v1, v2):
                    penalty += 1.0
        
        return os + penalty * penalty_weight

    def is_valid(inner, oc, os, oa):
        """Check full validity."""
        # Inside check
        for i in range(len(inner)):
            cx, cy, ang = inner[i]
            verts = hex_vertices(cx, cy, 1.0, ang)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, oc[0], oc[1], os, oa):
                    return False
        # Disjoint check
        for i in range(len(inner)):
            for j in range(i+1, len(inner)):
                v1 = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not sat_disjoint(v1, v2):
                    return False
        return True

    def make_params(inner, oc, os, oa):
        params = []
        for p in inner:
            params.extend([p[0], p[1], p[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    def parse_params(params):
        n = N
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(n)]
        oc = (params[3*n], params[3*n+1])
        os = params[3*n+2]
        oa = params[3*n+3]
        return inner, oc, os, oa

    # Initial configurations
    configs = []
    
    # Config 1: 4x3 grid
    cfg1_inner = []
    for y in [-1.8, 0.0, 1.8]:
        for x in [-3.0, -1.0, 1.0, 3.0]:
            cfg1_inner.append([x, y, 0.0])
    configs.append((cfg1_inner, [0, 0], 5.5, 30.0))
    
    # Config 2: Hexagonal cluster
    cfg2_inner = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.radians(60 * k)
        cfg2_inner.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        cfg2_inner.append([3.5 * np.cos(ang), 3.5 * np.sin(ang), 0.0])
    configs.append((cfg2_inner, [0, 0], 6.0, 30.0))
    
    # Config 3: 2 rows of 6 staggered
    cfg3_inner = []
    for i in range(6):
        cfg3_inner.append([i * 2.0 - 5.0, -0.9, 0.0])
        cfg3_inner.append([i * 2.0 - 4.0, 0.9, 0.0])
    configs.append((cfg3_inner, [0, 0], 6.5, 30.0))
    
    # Config 4: Compact triangular arrangement
    cfg4_inner = []
    # Row 1: 4 hexagons
    for i in range(4):
        cfg4_inner.append([i * 2.0 - 3.0, 0.0, 0.0])
    # Row 2: 4 hexagons
    for i in range(4):
        cfg4_inner.append([i * 2.0 - 2.0, 1.732, 0.0])
    # Row 3: 4 hexagons
    for i in range(4):
        cfg4_inner.append([i * 2.0 - 3.0, 3.464, 0.0])
    configs.append((cfg4_inner, [0, 1.732], 5.5, 30.0))
    
    # Find best initial
    best_params = None
    best_score = float('inf')
    
    for inner_cfg, oc_cfg, os_cfg, oa_cfg in configs:
        inner = [(p[0], p[1], p[2]) for p in inner_cfg]
        # Expand outer if needed
        os_try = os_cfg
        while not all(point_in_hexagon(vx, vy, oc_cfg[0], oc_cfg[1], os_try, oa_cfg) 
                     for p in inner for vx, vy in hex_vertices(p[0], p[1], 1.0, p[2])):
            os_try += 0.1
            if os_try > 12:
                break
        params = make_params(inner, oc_cfg, os_try, oa_cfg)
        score = compute_penalties(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    # Local optimization with multiple restarts
    np.random.seed(42)
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if restart == 0:
            current = best_params.copy()
        else:
            noise_scale = 0.2 + 0.1 * restart
            current = best_params + np.random.uniform(-noise_scale, noise_scale, size=len(best_params))
            current[3*N+2] = max(current[3*N+2], 4.0)
        
        try:
            res = minimize(compute_penalties, current, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if best_score < 4.0:
            break
    
    # Final refinement
    if time.time() - start_time < DEADLINE - 5 and best_score < 4.5:
        try:
            res2 = minimize(compute_penalties, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 0.002, 'fatol': 0.002, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    inner, oc, os, oa = parse_params(best_params)
    
    # Validate; if invalid, use best valid config
    if not is_valid(inner, oc, os, oa):
        # Try each initial config
        for inner_cfg, oc_cfg, os_cfg, oa_cfg in configs:
            inner_test = [(p[0], p[1], p[2]) for p in inner_cfg]
            os_test = os_cfg
            # Expand if needed
            while not all(point_in_hexagon(vx, vy, oc_cfg[0], oc_cfg[1], os_test, oa_cfg) 
                         for p in inner_test for vx, vy in hex_vertices(p[0], p[1], 1.0, p[2])):
                os_test += 0.1
                if os_test > 12:
                    break
            if is_valid(inner_test, oc_cfg, os_test, oa_cfg):
                inner, oc, os, oa = inner_test, tuple(oc_cfg), os_test, oa_cfg
                break
        else:
            # Fallback: large hexagon
            inner = [(0, 0, 0)] * N
            os = 8.0
            inner = []
            for i in range(N):
                ang = 2 * np.pi * i / N
                inner.append([3 * np.cos(ang), 3 * np.sin(ang), 0.0])
            oc = (0.0, 0.0)
            oa = 30.0
    
    return inner, oc, os, oa
