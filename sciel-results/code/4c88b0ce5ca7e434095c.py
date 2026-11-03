"""Stronger packing: multi-start hexagonal lattice + Nelder-Mead local refinement + deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
SEED = 42

# ---- helpers ---------------------------------------------------------------

def _pairwise_distances(pts):
    """Vectorised pairwise distances, shape (N, N)."""
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
    return d

def _score(pts):
    """Largest common radius: min(min_{i<j} |c_i-c_j|/2, min_i(1-|c_i|))."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(pts)):
        return -np.inf
    # container constraint
    dist_to_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    container = 1.0 - dist_to_origin
    # separation constraint
    d = _pairwise_distances(pts)
    # zero out diagonal
    np.fill_diagonal(d, np.inf)
    sep = np.min(d) / 2.0
    return min(container.min(), sep)

def _score_smooth(pts, beta=50.0):
    """Smooth surrogate using log-sum-exp soft-min for both constraints."""
    pts = np.asarray(pts, dtype=np.float64)
    dist_to_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    # soft-min of (1 - |c_i|): use -log-sum-exp of negatives
    a = -(1.0 - dist_to_origin) * beta
    a_max = a.max()
    softmin_container = -(a_max + np.log(np.sum(np.exp(a - a_max)))) / beta
    # pairwise distances
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    # soft-min of d/2
    b = -d * beta
    b_max = b.max(axis=None)
    # For each pair we want soft-min; do a global soft-min over all off-diagonal entries
    b_flat = b.flatten()
    b_max_flat = b_flat.max()
    softmin_sep = -(b_max_flat + np.log(np.sum(np.exp(b_flat - b_max_flat)))) / beta
    return min(softmin_container, softmin_sep)

def _make_hex_lattice(rings=8):
    """Generate points on a hexagonal lattice inside unit disk, return list of (x,y)."""
    pts = []
    for q in range(-rings, rings + 1):
        for r in range(max(-rings, -q - rings), min(rings, -q + rings) + 1):
            x = 0.5 * q + 0.86602540378 * r
            y = 0.86602540378 * r
            if np.hypot(x, y) <= 1.0:
                pts.append((x, y))
    return pts

def _scale_and_select(pts, n):
    """Scale hex lattice points to fit inside unit circle with margin, pick n closest to origin."""
    pts = np.array(pts, dtype=np.float64)
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= 1.0]
    if len(pts) < n:
        # expand rings
        return _make_hex_lattice(rings=15)
    # scale so max radius is ~0.95
    scale = 0.95 / r.max() if r.max() > 0 else 1.0
    pts = pts * scale
    r2 = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r2, kind='stable')[:n]
    return pts[idx]

def _refine(pts, max_iter=3000, tol=1e-12, time_limit=12.0):
    """Refine packing using Nelder-Mead on smooth objective, then hard score check."""
    start_time = time.time()
    x0 = pts.flatten().copy()
    best = _score(pts)
    
    def objective(x):
        p = x.reshape(N, 2)
        if not np.all(np.isfinite(p)):
            return 1e10
        s = _score_smooth(p, beta=80.0)
        # add penalty for infeasibility
        d = np.sqrt(np.sum(p ** 2, axis=1))
        penalty = np.maximum(0.0, d - 1.0 + 0.01).sum()
        return -s + 10.0 * penalty
    
    options = {'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'disp': False}
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        p = res.x.reshape(N, 2)
        s = _score(p)
        if s > best:
            best = s
    except Exception:
        pass
    
    if time.time() - start_time > time_limit:
        return pts, best
    
    # Polish with tighter Nelder-Mead on hard score (piecewise)
    def hard_obj(x):
        p = x.reshape(N, 2)
        if not np.all(np.isfinite(p)):
            return 1e10
        return -_score(p)
    
    x0b = pts.flatten().copy()
    try:
        res2 = minimize(hard_obj, x0b, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'disp': False})
        p2 = res2.x.reshape(N, 2)
        s2 = _score(p2)
        if s2 > best:
            best = s2
            pts = p2
    except Exception:
        pass
    
    return pts, best

# ---- main construct --------------------------------------------------------

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    rng = np.random.RandomState(SEED)
    
    best_pts = None
    best_score = -np.inf
    
    # Generate several hex-lattice initializations at different scales
    scales = [0.90, 0.92, 0.94, 0.96, 0.98]
    for sc in scales:
        if time.time() > deadline - 30.0:
            break
        hex_pts = _make_hex_lattice(rings=10)
        hex_pts = np.array(hex_pts, dtype=np.float64)
        r = np.hypot(hex_pts[:, 0], hex_pts[:, 1])
        hex_pts = hex_pts[r <= 1.0]
        scale = sc / r.max() if r.max() > 0 else 1.0
        hex_pts = hex_pts * scale
        r2 = np.hypot(hex_pts[:, 0], hex_pts[:, 1])
        idx = np.argsort(r2, kind='stable')[:N]
        init = hex_pts[idx].copy()
        # small random perturbation to break symmetry
        init += rng.normal(0, 0.001, size=init.shape)
        # clamp inside
        dr = np.hypot(init[:, 0], init[:, 1])
        mask = dr > 0.999
        init[mask] *= 0.999 / dr[mask]
        
        if time.time() > deadline - 15.0:
            s = _score(init)
            if s > best_score:
                best_score = s
                best_pts = init.copy()
            continue
        
        refined, sc2 = _refine(init, max_iter=2000, tol=1e-12, time_limit=10.0)
        if sc2 > best_score:
            best_score = sc2
            best_pts = refined.copy()
    
    # If we still have time, do a final polish on the best
    if best_pts is not None and time.time() < deadline - 5.0:
        final_pts, final_score = _refine(best_pts, max_iter=3000, tol=1e-14, time_limit=10.0)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    if best_pts is None:
        # fallback: simple ring
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    return best_pts
