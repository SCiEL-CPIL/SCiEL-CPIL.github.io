"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    s = side
    if abs(lx) > s + 1e-12:
        return False
    if abs(ly) > s * math.sqrt(3) / 2 + 1e-12:
        return False
    if abs(lx) / s + 2 * abs(ly) / (s * math.sqrt(3)) > 1 + 1e-12:
        return False
    return True

def _hexagon_in_hexagon(inner_verts, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    for v in inner_verts:
        if not _point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _hexagons_disjoint(centers, angles, eps=1e-4):
    """Check if unit hexagons at given centers/angles are disjoint."""
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            ci = centers[i]
            cj = centers[j]
            ai = angles[i]
            aj = angles[j]
            dx = ci[0] - cj[0]
            dy = ci[1] - cj[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist >= 2.0 + eps:
                continue
            if dist < 1.5 - eps:
                return False
            vi = _hex_vertices(ci[0], ci[1], 1.0, ai)
            vj = _hex_vertices(cj[0], cj[1], 1.0, aj)
            for v in vi:
                if _point_in_hexagon(v[0], v[1], cj[0], cj[1], 1.0, aj):
                    return False
            for v in vj:
                if _point_in_hexagon(v[0], v[1], ci[0], ci[1], 1.0, ai):
                    return False
    return True

def _min_enclosing_hexagon_side(centers, angles, max_time=None, start_time=None):
    """Find the minimum side length of a regular hexagon that contains all unit hexagons."""
    best_side = float('inf')
    if max_time is not None and start_time is not None:
        if time.time() - start_time > max_time:
            return best_side
    
    for outer_angle in range(0, 30, 5):
        for trial in range(15):
            if max_time is not None and start_time is not None:
                if time.time() - start_time > max_time:
                    return best_side
            if trial == 0:
                ocx, ocy = 0.0, 0.0
            else:
                ocx = (np.random.rand() - 0.5) * 4
                ocy = (np.random.rand() - 0.5) * 4
            
            lo, hi = 2.0, 20.0
            for _ in range(35):
                mid = (lo + hi) / 2
                ok = True
                for k in range(len(centers)):
                    iv = _hex_vertices(centers[k][0], centers[k][1], 1.0, angles[k])
                    if not _hexagon_in_hexagon(iv, ocx, ocy, mid, outer_angle):
                        ok = False
                        break
                if ok:
                    hi = mid
                else:
                    lo = mid
            if hi < best_side:
                best_side = hi
    return best_side

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Initial configuration: hexagonal lattice
    centers = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, math.sqrt(3)],
        [-1.0, math.sqrt(3)],
        [-2.0, 0.0],
        [-1.0, -math.sqrt(3)],
        [1.0, -math.sqrt(3)],
        [3.0, math.sqrt(3)],
        [0.0, 2*math.sqrt(3)],
        [-3.0, math.sqrt(3)],
        [3.0, -math.sqrt(3)],
    ])
    angles = np.zeros(N)
    
    best_side = _min_enclosing_hexagon_side(centers, angles, DEADLINE, start_time)
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_ocx, best_ocy, best_oangle = 0.0, 0.0, 0.0
    
    # Nelder-Mead optimization
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return best_side
        c = params[:2*N].reshape(N, 2)
        a = params[2*N:]
        if not _hexagons_disjoint(c, a):
            return best_side + 100.0
        return _min_enclosing_hexagon_side(c, a, DEADLINE, start_time)
    
    for restart in range(4):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_centers.flatten(), best_angles])
        else:
            c = best_centers + np.random.randn(N, 2) * 0.3
            a = best_angles + np.random.randn(N) * 5
            x0 = np.concatenate([c.flatten(), a])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 0.02, 'fatol': 0.02})
            if res.fun < best_side:
                c = res.x[:2*N].reshape(N, 2)
                a = res.x[2*N:]
                if _hexagons_disjoint(c, a):
                    best_side = res.fun
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Final polish
    if time.time() - start_time < DEADLINE - 5:
        for outer_angle in range(0, 30, 2):
            if time.time() - start_time > DEADLINE - 5:
                break
            for trial in range(8):
                if time.time() - start_time > DEADLINE - 5:
                    break
                if trial == 0:
                    ocx, ocy = 0.0, 0.0
                else:
                    ocx = (np.random.rand() - 0.5) * 2
                    ocy = (np.random.rand() - 0.5) * 2
                
                lo, hi = 2.0, 20.0
                for _ in range(45):
                    mid = (lo + hi) / 2
                    ok = True
                    for k in range(N):
                        iv = _hex_vertices(best_centers[k][0], best_centers[k][1], 1.0, best_angles[k])
                        if not _hexagon_in_hexagon(iv, ocx, ocy, mid, outer_angle):
                            ok = False
                            break
                    if ok:
                        hi = mid
                    else:
                        lo = mid
                if hi < best_side:
                    best_side = hi
                    best_ocx, best_ocy, best_oangle = ocx, ocy, outer_angle
                    break
    
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, (float(best_ocx), float(best_ocy)), float(best_side), float(best_oangle)
