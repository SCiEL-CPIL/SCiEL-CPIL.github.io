import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START = time.time()
_DEADLINE = _START + 140.0

# Precompute base hexagon
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])

def _hex_verts(center, angle):
    return center + _BASE @ _rot(angle)

def _separating_axis_dist(V, W):
    # Vectorized SAT
    normals = []
    for poly in (V, W):
        for i in range(6):
            e = poly[(i + 1) % 6] - poly[i]
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                normals.append(n / n_norm)
    normals = np.array(normals)
    proj_V = V @ normals.T
    proj_W = W @ normals.T
    r1_min = proj_V.min(axis=0)
    r1_max = proj_V.max(axis=0)
    r2_min = proj_W.min(axis=0)
    r2_max = proj_W.max(axis=0)
    gap1 = r2_min - r1_max
    gap2 = r1_min - r2_max
    valid1 = gap1 > 0
    valid2 = gap2 > 0
    min_dist = 1e18
    if np.any(valid1):
        min_dist = min(min_dist, float(gap1[valid1].min()))
    if np.any(valid2):
        min_dist = min(min_dist, float(gap2[valid2].min()))
    if min_dist > 1e10:
        return 0.0
    return min_dist

def _enclosing_side(points, alpha):
    # points: (N, 6, 2) -> flatten to (N*6, 2)
    pts = points.reshape(-1, 2)
    # Normals for outer hexagon rotated by alpha
    # Base normals: 30, 90, 150
    base_angles = np.array([30.0, 90.0, 150.0])
    angles = base_angles + math.degrees(alpha)
    normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles])
    widths = []
    for n in normals:
        p = pts @ n
        widths.append(p.max() - p.min())
    return max(widths) / math.sqrt(3.0)

def _eval(params):
    # params: [x1, y1, a1, ..., x11, y11, a11, alpha_outer]
    n = N
    verts = np.empty((n, 6, 2))
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ca = params[3*i+2]
        verts[i] = np.array([cx, cy]) + _BASE @ _rot(ca)
    
    alpha = params[3*n]
    side = _enclosing_side(verts, alpha)
    
    pen = 0.0
    margin = 0.01
    for i in range(n):
        vi = verts[i]
        for j in range(i+1, n):
            d = _separating_axis_dist(vi, verts[j])
            if d < margin:
                pen += (margin - d)**2
    
    return side + 100.0 * pen

def _initial_guess():
    # Hexagonal lattice
    centers = []
    # Center
    centers.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        ang = math.radians(60*k)
        centers.append([math.cos(ang)*math.sqrt(3), math.sin(ang)*math.sqrt(3), 0.0])
    # 4 more: place at distance 3*sqrt(3)?
    # Let's try a 3x3 block + 2
    # Or just random scatter
    for k in range(4):
        ang = math.radians(60*k + 30)
        centers.append([math.cos(ang)*3.0*1.732, math.sin(ang)*3.0*1.732, 0.0])
    
    params = np.array(centers).flatten()
    params = np.append(params, 0.0) # outer rotation
    return params

def construct():
    best_score = 1e18
    best_params = None
    
    # Try a few random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > _DEADLINE: break
        
        # Generate initial guess
        if restart == 0:
            p0 = _initial_guess()
        else:
            p0 = _initial_guess() + rng.normal(0, 0.5, size=3*N+1)
        
        # Optimize
        res = minimize(_eval, p0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    
    if best_params is None:
        # Fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N-N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    # Extract results
    theta_g = float(best_params[3*N])
    inner = []
    for i in range(N):
        cx = float(best_params[3*i])
        cy = float(best_params[3*i+1])
        ca = float(best_params[3*i+2])
        ang = (ca) % 360.0
        inner.append([cx, cy, ang])
    
    side = _enclosing_side(np.array([np.array([best_params[3*i], best_params[3*i+1]]) + _BASE @ _rot(best_params[3*i+2]) for i in range(N)]), theta_g)
    
    return inner, (0.0, 0.0), float(side), float(theta_g)
