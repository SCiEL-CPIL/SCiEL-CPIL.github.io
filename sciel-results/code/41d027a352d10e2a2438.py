import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 560  # safety margin
    n = 14

    def normalize_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        mask = norms < 1e-12
        if np.any(mask):
            norms[mask] = 1.0
            pts[mask, 0] = 1.0
            pts[mask, 1:] = 0.0
        return pts / norms

    def ratio(pts):
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        return d_min / d_max

    def neg_ratio(x):
        pts = x.reshape(n, 3)
        pts = normalize_sphere(pts)
        return -ratio(pts)

    def soft_neg_ratio(x, beta=50.0):
        pts = x.reshape(n, 3)
        pts = normalize_sphere(pts)
        d = pdist(pts)
        d_min_soft = -np.logaddexp.reduce(-d / beta) * beta
        d_max_soft = np.logaddexp.reduce(d / beta) * beta
        if d_max_soft <= 1e-15:
            return 0.0
        return -(d_min_soft / d_max_soft)

    # Base configurations on sphere
    k = np.arange(n)
    phi = np.arccos(1 - 2 * (k + 0.5) / n)
    golden = np.pi * (1 + np.sqrt(5))
    fib = np.column_stack([
        np.sin(phi) * np.cos(golden * k),
        np.sin(phi) * np.sin(golden * k),
        np.cos(phi)
    ])

    # Icosahedron 12 vertices
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
        [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
        [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 poles to make 14
    ico14 = np.vstack([ico, [[0, 0, 1.0], [0, 0, -1.0]]])

    # Create more diverse initial candidates on sphere
    np.random.seed(777)
    candidates = [
        fib,
        ico14,
    ]
    
    # Add perturbed versions
    for base in [fib, ico14]:
        for noise_level in [0.05, 0.1, 0.2, 0.3]:
            perturbed = base + noise_level * np.random.randn(n, 3)
            perturbed = normalize_sphere(perturbed)
            candidates.append(perturbed)
    
    # Add random sphere points
    for _ in range(8):
        v = np.random.randn(n, 3)
        v = normalize_sphere(v)
        candidates.append(v)

    best_pts = None
    best_r = -1.0

    # Phase 1: Powell optimization with exact ratio
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 180:
            break
        x0 = cand.flatten()
        try:
            res = minimize(neg_ratio, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            r = -float(res.fun)
            if np.isfinite(r) and r > best_r:
                best_r = r
                best_pts = res.x.reshape(n, 3)
        except Exception:
            pass

    # Phase 2: Refine best with soft objective for smoother landscape
    if best_pts is not None and time.time() < deadline - 120:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 90:
                break
            try:
                res_soft = minimize(lambda x: soft_neg_ratio(x, beta), 
                                   best_pts.flatten(), 
                                   method='Powell',
                                   options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11})
                r_soft = -float(res_soft.fun)
                # Evaluate with exact ratio
                pts_soft = normalize_sphere(res_soft.x.reshape(n, 3))
                r_exact = ratio(pts_soft)
                if np.isfinite(r_exact) and r_exact > best_r:
                    best_r = r_exact
                    best_pts = pts_soft
            except Exception:
                pass

    # Phase 3: Final high-precision polish with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 60:
        try:
            res2 = minimize(neg_ratio, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            r2 = -float(res2.fun)
            if np.isfinite(r2) and r2 > best_r:
                best_r = r2
                best_pts = normalize_sphere(res2.x.reshape(n, 3))
        except Exception:
            pass

    if best_pts is None:
        best_pts = fib

    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.0, neginf=0.0)
    return best_pts
