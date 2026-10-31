"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
TRI_AREA = (3 ** 0.5) / 4.0

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _bary_to_cart(bary):
    """Convert barycentric coords (n,3) to Cartesian (n,2)."""
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the simplex."""
    eps = 1e-10
    bary = np.clip(bary, eps, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return bary

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from Cartesian coords."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Minimum area among all triangles, normalized by TRI_AREA."""
    return float(np.min(_all_areas_cart(pts))) / TRI_AREA

def _min_area_bary(bary):
    """Minimum area from barycentric coords, normalized."""
    pts = _bary_to_cart(bary)
    return _min_area_cart(pts)

def _delaunay_tris(pts):
    """Get Delaunay triangulation triangle indices (local to point set)."""
    try:
        tri = Delaunay(pts)
        return tri.simplices
    except Exception:
        return None

def _soft_min_delaunay(bary_flat, beta=200.0):
    """Smooth objective using only Delaunay triangles for the min computation."""
    bary = _clamp_bary(bary_flat.reshape(-1, 3))
    pts = _bary_to_cart(bary)
    areas = _all_areas_cart(pts)
    
    # Try to get Delaunay triangles
    try:
        tri = Delaunay(pts)
        # Map simplices to our triangle indices
        del_areas = []
        for simplex in tri.simplices:
            i, j, k = sorted(simplex)
            idx = np.where((_TRI_IDX[:, 0] == i) & (_TRI_IDX[:, 1] == j) & (_TRI_IDX[:, 2] == k))[0]
            if len(idx) > 0:
                del_areas.append(areas[idx[0]])
        if del_areas:
            del_areas = np.array(del_areas)
            m = np.min(del_areas)
            shifted = -beta * (del_areas - m)
            shifted = np.clip(shifted, -400, 400)
            log_sum = np.log(np.sum(np.exp(shifted)))
            return -(m - log_sum / beta) / TRI_AREA
    except Exception:
        pass
    
    # Fallback: use all triangles
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta) / TRI_AREA

def _soft_min_all(bary_flat, beta=100.0):
    """Smooth objective using all 165 triangles."""
    bary = _clamp_bary(bary_flat.reshape(-1, 3))
    pts = _bary_to_cart(bary)
    areas = _all_areas_cart(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return -(m - log_sum / beta) / TRI_AREA

def _hard_min_bary(bary_flat):
    """Hard min objective in barycentric coordinates."""
    bary = _clamp_bary(bary_flat.reshape(-1, 3))
    return -_min_area_bary(bary)

def _init_hexagonal():
    """Hexagonal/symmetric initialization in barycentric coords."""
    bary = []
    # Center
    bary.append([1/3, 1/3, 1/3])
    # 6 points around center in hexagonal pattern
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.15
    for k in range(6):
        angle = k * np.pi / 3
        # Convert circular pattern to barycentric
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        # Map to barycentric: x = b + 0.5c, y = sqrt(3)/2 * c
        c_new = 2 * dy / np.sqrt(3) + cz
        b_new = dx + 0.5 * c_new + bx - 0.5 * cz  # approximate
        a_new = 1 - b_new - c_new
        bary.append([max(0.01, a_new), max(0.01, b_new), max(0.01, c_new)])
    # 4 more points at strategic locations
    bary.append([0.05, 0.475, 0.475])
    bary.append([0.475, 0.05, 0.475])
    bary.append([0.475, 0.475, 0.05])
    bary.append([0.15, 0.15, 0.70])
    bary = _clamp_bary(np.array(bary[:N]))
    return bary

def _init_structured_bary():
    """3-fold symmetric structured initialization in barycentric."""
    bary = []
    # Center
    bary.append([1/3, 1/3, 1/3])
    # Inner ring (3 points, 120 deg apart)
    r1 = 0.18
    for k in range(3):
        angle = k * 2 * np.pi / 3 + np.pi / 6
        # In barycentric, a circle around center
        # Use parametric: perturb each coordinate
        a = 1/3 + r1 * np.cos(angle) * 0.7
        b = 1/3 + r1 * np.cos(angle + 2*np.pi/3) * 0.7
        c = 1/3 + r1 * np.cos(angle + 4*np.pi/3) * 0.7
        bary.append([max(0.01, a), max(0.01, b), max(0.01, c)])
    # Outer ring (6 points)
    r2 = 0.32
    for k in range(6):
        angle = k * np.pi / 3
        a = 1/3 + r2 * np.cos(angle) * 0.7
        b = 1/3 + r2 * np.cos(angle + 2*np.pi/3) * 0.7
        c = 1/3 + r2 * np.cos(angle + 4*np.pi/3) * 0.7
        bary.append([max(0.01, a), max(0.01, b), max(0.01, c)])
    return _clamp_bary(np.array(bary[:N]))

def _init_random_bary(seed):
    """Random initialization in barycentric coords."""
    rng = np.random.default_rng(seed)
    bary = rng.dirichlet(np.ones(N), size=1)
    return _clamp_bary(bary)

def _init_grid_bary():
    """Triangular grid in barycentric."""
    bary = []
    # Rows of points
    rows = [1, 2, 3, 3, 2]
    for row_idx, count in enumerate(rows):
        frac = (row_idx + 1) / (len(rows) + 1)
        # y goes from 0 to H, in barycentric c = 2y/sqrt(3)
        c_val = frac
        a_max = 1 - c_val
        for i in range(count):
            b_val = a_max * (i + 0.5) / count
            a_val = a_max - b_val
            bary.append([a_val, b_val, c_val])
    return _clamp_bary(np.array(bary[:N]))

def _lbfgs_bary(bary, beta, maxiter, deadline):
    """L-BFGS-B in barycentric space."""
    try:
        res = minimize(
            _soft_min_delaunay, bary.flatten(), args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        opt = _clamp_bary(res.x.reshape(-1, 3))
        return opt, _min_area_bary(opt)
    except Exception:
        return bary, _min_area_bary(bary)

def _nm_bary(bary, deadline):
    """Nelder-Mead in barycentric space."""
    try:
        res = minimize(
            _hard_min_bary, bary.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
        )
        opt = _clamp_bary(res.x.reshape(-1, 3))
        return opt, _min_area_bary(opt)
    except Exception:
        return bary, _min_area_bary(bary)

def _local_polish_bary(bary, deadline, max_rounds=300):
    """Local coordinate descent in barycentric space with adaptive steps."""
    current = bary.copy()
    score = _min_area_bary(current)
    steps = [0.03, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in steps:
        rounds = max(20, max_rounds // len(steps))
        for _ in range(rounds):
            if time.time() > deadline:
                return current, score
            improved = False
            for i in range(N):
                for coord in range(3):
                    for delta in [-step, step]:
                        trial = current.copy()
                        trial[i, coord] += delta
                        trial = _clamp_bary(trial)
                        s = _min_area_bary(trial)
                        if s > score + 1e-14:
                            current = trial
                            score = s
                            improved = True
            if not improved:
                break
    return current, score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def tleft():
        return deadline - time.time()
    
    best_bary = None
    best_score = 0.0
    
    # Generate diverse initializations in barycentric space
    inits = [_init_hexagonal(), _init_structured_bary(), _init_grid_bary()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000, 3000, 5000]:
        inits.append(_init_random_bary(s))
    
    # Phase 1: Multi-start L-BFGS with escalating beta
    for init in inits:
        if tleft() < 10:
            break
        for beta in [80.0, 150.0, 300.0]:
            if tleft() < 4:
                break
            bary, sc = _lbfgs_bary(init, beta, 1000, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    if best_bary is None:
        best_bary = _init_structured_bary()
        best_score = _min_area_bary(best_bary)
    
    # Phase 2: NM polish on best
    if tleft() > 12:
        bary, sc = _nm_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 3: Local coordinate descent
    if tleft() > 8:
        bary, sc = _local_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    # Phase 4: Perturbation multi-start
    rng = np.random.default_rng(99999)
    for trial in range(40):
        if tleft() < 8:
            break
        scale = 0.005 + 0.003 * trial
        pert = best_bary + rng.normal(0, scale, size=(N, 3))
        pert = _clamp_bary(pert)
        for beta in [100.0, 200.0]:
            if tleft() < 4:
                break
            bary, sc = _lbfgs_bary(pert, beta, 800, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
        if tleft() > 6:
            bary, sc = _nm_bary(pert, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    # Phase 5: Final refinement cycle
    if tleft() > 10:
        bary, sc = _local_polish_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
    
    if tleft() > 15:
        bary, sc = _nm_bary(best_bary, deadline)
        if sc > best_score:
            best_score = sc
            best_bary = bary.copy()
        if tleft() > 8:
            bary, sc = _local_polish_bary(best_bary, deadline)
            if sc > best_score:
                best_score = sc
                best_bary = bary.copy()
    
    # Convert to Cartesian and verify
    best_pts = _bary_to_cart(best_bary)
    
    # Ensure feasibility
    for i in range(N):
        x, y = best_pts[i]
        if x < -1e-9 or y < -1e-9 or x > 1 + 1e-9:
            best_pts[i] = _clamp_bary(np.array([[best_pts[i][0], best_pts[i][1]]]).reshape(1, -1) if False else np.array([_cart_to_bary(best_pts[i:i+1]).flatten()])).reshape(1, -1) if False else best_pts[i]
    
    # Final clamp
    final_bary = _cart_to_bary(best_pts)
    final_bary = _clamp_bary(final_bary)
    best_pts = _bary_to_cart(final_bary)
    
    return best_pts
