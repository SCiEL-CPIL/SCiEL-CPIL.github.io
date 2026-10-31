import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area among all triangles."""
    return float(np.min(_all_areas(pts)))

def _soft_min(pts, beta=100.0):
    """Smooth approximation of min area for gradient-based methods."""
    areas = _all_areas(pts)
    m = np.min(areas)
    # Use log-sum-exp for stability
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _clamp_point(p):
    """Clamp point to triangle using barycentric coordinates."""
    x, y = p
    # Barycentric coordinates relative to V0, V1, V2
    # Area of triangle V0 V1 V2 is H/2 * 1 = H/2? No, area is sqrt(3)/4.
    # Let's use the standard barycentric calculation.
    # p = u*V0 + v*V1 + w*V2, u+v+w=1
    # We can solve for v, w using linear algebra or geometric formulas.
    # Vector V1-V0 = (1, 0), V2-V0 = (0.5, H)
    # p - V0 = (x, y)
    # (x, y) = v * (1, 0) + w * (0.5, H)
    # y = w * H  => w = y / H
    # x = v + w * 0.5 => v = x - 0.5 * w
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    
    eps = 1e-9
    u = max(eps, min(1.0 - 2*eps, u))
    v = max(eps, min(1.0 - 2*eps, v))
    w = max(eps, min(1.0 - 2*eps, w))
    s = u + v + w
    u /= s; v /= s; w /= s
    return u * V0 + v * V1 + w * V2

def _clamp_all(pts):
    """Clamp all points inside triangle."""
    return np.array([_clamp_point(p) for p in pts])

def _obj_smooth(flat, beta=100.0):
    """Objective for L-BFGS-B: negative soft-min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_soft_min(pts, beta)

def _obj_hard(flat):
    """Objective for Nelder-Mead: negative hard min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_min_area(pts)

def _init_hexagonal():
    """Hexagonal grid initialization, which is often close to optimal for packing."""
    # 11 points in a hexagonal pattern
    # Center + 6 around it + 4 more?
    # Let's try a structured 4-3-4 grid or similar.
    # A known good configuration for 11 points is a deformed hexagonal lattice.
    pts = []
    # Center
    cx, cy = 1/3.0, H/3.0
    pts.append([cx, cy])
    # Ring 1: 6 points
    r1 = 0.15
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)])
    # Ring 2: 4 points (corners)
    r2 = 0.35
    # Adjust angles to avoid edges
    angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for angle in angles:
        pts.append([cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)])
    return np.array(pts[:N])

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        pts.append(r1 * V1 + r2 * V2)
    return np.array(pts)

def _lbfgs_optimize(pts, beta_start=50.0, beta_end=500.0, n_steps=10, maxiter_per_step=300, deadline=None):
    """L-BFGS-B optimization with annealing beta."""
    current = pts
    for i in range(n_steps):
        if deadline and time.time() > deadline:
            break
        # Interpolate beta
        beta = beta_start + (beta_end - beta_start) * (i / max(n_steps - 1, 1))
        try:
            res = minimize(
                _obj_smooth, current.flatten(), args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': maxiter_per_step, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            current = _clamp_all(res.x.reshape(-1, 2))
        except Exception:
            break
    return current

def _nm_optimize(pts, maxiter=2000, deadline=None):
    """Nelder-Mead optimization on hard objective."""
    try:
        res = minimize(
            _obj_hard, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        return _clamp_all(res.x.reshape(-1, 2))
    except Exception:
        return pts

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    def tleft():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Initializations
    inits = [_init_hexagonal()]
    for s in [1, 7, 13, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000, 2500, 3000]:
        inits.append(_init_random(s))
    
    # Phase 1: Multi-start L-BFGS with annealing
    for init in inits:
        if tleft() < 15:
            break
        pts = _lbfgs_optimize(init, beta_start=50, beta_end=500, n_steps=8, maxiter_per_step=200, deadline=deadline)
        # Quick NM polish
        pts = _nm_optimize(pts, maxiter=1000, deadline=deadline)
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _init_hexagonal()
        best_score = _min_area(best_pts)
        
    # Phase 2: Local search around best solution
    rng = np.random.default_rng(99999)
    for trial in range(40):
        if tleft() < 10:
            break
        # Perturb with decreasing noise
        scale = 0.05 * (0.9 ** trial)
        if scale < 0.001:
            scale = 0.001
        pert = best_pts + rng.normal(0, scale, size=(N, 2))
        pert = _clamp_all(pert)
        
        # Optimize perturbation
        pts = _lbfgs_optimize(pert, beta_start=100, beta_end=1000, n_steps=5, maxiter_per_step=150, deadline=deadline)
        pts = _nm_optimize(pts, maxiter=1500, deadline=deadline)
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Phase 3: Final high-precision polish
    if tleft() > 5:
        # One more NM with higher precision
        best_pts = _nm_optimize(best_pts, maxiter=3000, deadline=deadline)
        best_score = _min_area(best_pts)
        
    return best_pts
