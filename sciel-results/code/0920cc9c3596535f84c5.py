import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_support_unit(angle_deg, dir_angle):
    """Support function of unit hexagon (side=1, rotated by angle_deg) in direction dir_angle."""
    a = np.radians(angle_deg)
    d = np.radians(dir_angle)
    # Vertices at angles a + pi/6 + k*pi/3
    # Support = max_k cos(d - (a + pi/6 + k*pi/3))
    vals = np.cos(d - a - np.pi/6 - np.arange(6)*np.pi/3)
    return np.max(vals)

def _objective(params):
    """Minimize outer hexagon side length.
    params: [x1,y1,a1, ..., x12,y12,a12, cx,cy,cangle]
    """
    inner = []
    for i in range(N):
        inner.append((params[3*i], params[3*i+1], params[3*i+2]))
    ocx, ocy = params[3*N], params[3*N+1]
    oangle = params[3*N+2]
    
    # Check non-overlap using SAT
    for i in range(N):
        for j in range(i+1, N):
            if _hex_overlap(inner[i], inner[j]):
                return 1e10
    
    # Compute minimum outer side
    oa = np.radians(oangle)
    # 6 outward normals of outer hexagon
    normal_angles = oa + np.arange(6) * np.pi/3
    
    max_proj = 0
    for (ix, iy, ia_deg) in inner:
        ia = np.radians(ia_deg)
        for na in normal_angles:
            # Support of inner hex in this normal direction
            h = _hex_support_unit(ia_deg, np.degrees(na))
            proj = (ix - ocx) * np.cos(na) + (iy - ocy) * np.sin(na) + h
            if proj > max_proj:
                max_proj = proj
    
    if max_proj <= 0:
        return 0.0
    s = 2.0 * max_proj / np.sqrt(3.0)
    return s

def _hex_overlap(h1, h2, tol=1e-8):
    """Check if two unit hexagons overlap using SAT."""
    x1, y1, a1 = h1
    x2, y2, a2 = h2
    a1r = np.radians(a1)
    a2r = np.radians(a2)
    
    v1x = x1 + np.cos(a1r + np.pi/6 + np.arange(6)*np.pi/3)
    v1y = y1 + np.sin(a1r + np.pi/6 + np.arange(6)*np.pi/3)
    v2x = x2 + np.cos(a2r + np.pi/6 + np.arange(6)*np.pi/3)
    v2y = y2 + np.sin(a2r + np.pi/6 + np.arange(6)*np.pi/3)
    
    axes = []
    for k in range(6):
        dx = v1x[(k+1)%6] - v1x[k]
        dy = v1y[(k+1)%6] - v1y[k]
        ln = np.sqrt(dx*dx + dy*dy)
        if ln > 1e-12:
            axes.append((-dy/ln, dx/ln))
    for k in range(6):
        dx = v2x[(k+1)%6] - v2x[k]
        dy = v2y[(k+1)%6] - v2y[k]
        ln = np.sqrt(dx*dx + dy*dy)
        if ln > 1e-12:
            axes.append((-dy/ln, dx/ln))
    
    for ax, ay in axes:
        p1 = v1x*ax + v1y*ay
        p2 = v2x*ax + v2y*ay
        if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    # Hexagonal lattice initial placement
    positions = [(0.0, 0.0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions.append((2.0*np.cos(ang), 2.0*np.sin(ang)))
    # Add 5 from second ring
    for k in range(5):
        ang = k * np.pi / 3
        positions.append((4.0*np.cos(ang), 4.0*np.sin(ang)))
    
    inner_init = [(x, y, 0.0) for x, y in positions[:12]]
    
    params = np.zeros(3*N + 3)
    for i in range(N):
        params[3*i] = inner_init[i][0]
        params[3*i+1] = inner_init[i][1]
        params[3*i+2] = inner_init[i][2]
    params[3*N] = 0.0
    params[3*N+1] = 0.0
    params[3*N+2] = 0.0
    
    best_params = params.copy()
    best_score = _objective(params)
    
    # L-BFGS-B
    try:
        res = minimize(_objective, params, method='L-BFGS-B',
                      options={'maxiter': 1000, 'maxfun': 10000})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline - 30:
        inner_result = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
        return inner_result, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
    
    # Nelder-Mead polish
    try:
        res2 = minimize(_objective, best_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    # Random restarts
    np.random.seed(123)
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        p = best_params + np.random.randn(len(best_params)) * 0.3
        try:
            r = minimize(_objective, p, method='L-BFGS-B',
                        options={'maxiter': 500, 'maxfun': 5000})
            if r.fun < best_score:
                best_score = r.fun
                best_params = r.x.copy()
        except Exception:
            pass
    
    inner_result = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    return inner_result, (best_params[3*N], best_params[3*N+1]), best_score, best_params[3*N+2]
# EVOLVE-BLOCK-END
