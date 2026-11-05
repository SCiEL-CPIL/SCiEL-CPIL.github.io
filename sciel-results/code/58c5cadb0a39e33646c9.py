"""Combined program: soft-max diameter surrogate + L-BFGS-B + diverse inits + Nelder-Mead final polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 8

def _softmax_diameter(dists, beta):
    """Smooth approximation to max(dists). dists is a flat array of positive distances."""
    b = beta * dists
    b = b - np.max(b)
    return (1.0 / beta) * np.log(np.sum(np.exp(b)))

def _neg_log_score_surrogate(points_flat, beta):
    """Negative log of the scale-invariant score using soft-max diameter surrogate."""
    pts = points_flat.reshape(N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    if np.any(d < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(d))
    D = _softmax_diameter(d, beta)
    if D < 1e-12:
        return 1e10
    log_score = log_prod - N * (N - 1) * np.log(D)
    return -log_score

def _neg_log_score_exact(points_flat):
    """Negative log of the scale-invariant score using exact max (for Nelder-Mead)."""
    pts = points_flat.reshape(N, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    if np.any(d < 1e-12):
        return 1e10
    log_prod = np.sum(np.log(d))
    D = np.max(d)
    if D < 1e-12:
        return 1e10
    log_score = log_prod - N * (N - 1) * np.log(D)
    return -log_score

def _true_score(points):
    """True scale-invariant score."""
    pts = np.array(points)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    diameter = np.max(d)
    if diameter < 1e-12:
        return -np.inf
    log_prod = np.sum(np.log(d))
    log_score = log_prod - N * (N - 1) * np.log(diameter)
    return np.exp(log_score)

def _scale_to_diameter2(pts):
    pts = np.array(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(N, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-12:
        pts = pts * (2.0 / diameter)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # ---- Build diverse initializations ----
    inits = []
    rng = np.random.default_rng(7)

    # 1. Regular octagon
    angles = 2 * np.pi * np.arange(N) / N
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(octagon.copy())

    # 2. Antipodal pairs with varying radii
    base_angles = rng.uniform(0, np.pi, 4)
    radii = rng.uniform(0.4, 1.0, 4)
    antipodal_pts = np.zeros((8, 2))
    for i in range(4):
        a = base_angles[i]
        r = radii[i]
        antipodal_pts[2*i] = [r*np.cos(a), r*np.sin(a)]
        antipodal_pts[2*i+1] = [-r*np.cos(a), -r*np.sin(a)]
    inits.append(antipodal_pts)

    # 3. Alternating radii on octagon angles
    radii_alt = np.array([1.0, 0.7, 1.0, 0.7, 1.0, 0.7, 1.0, 0.7])
    alt_rad = np.column_stack([radii_alt * np.cos(angles), radii_alt * np.sin(angles)])
    inits.append(alt_rad)

    # 4. Two concentric squares
    angles_sq1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    angles_sq2 = angles_sq1 + np.pi/4
    r1, r2 = 1.0, 0.55
    two_sq = np.column_stack([
        np.concatenate([r1*np.cos(angles_sq1), r2*np.cos(angles_sq2)]),
        np.concatenate([r1*np.sin(angles_sq1), r2*np.sin(angles_sq2)])
    ])
    inits.append(two_sq)

    # 5. Three-cluster: 3+3+2 grouped at different angular sectors
    cluster_angles = np.array([0, 0.15, -0.15, 2.1, 2.25, 1.95, 4.2, 4.35])
    cluster_radii = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6])
    cluster_pts = np.column_stack([cluster_radii * np.cos(cluster_angles),
                                    cluster_radii * np.sin(cluster_angles)])
    inits.append(cluster_pts)

    # 6. Perturbed octagon
    inits.append(octagon + rng.normal(0, 0.3, size=(8, 2)))

    # 7. Random
    inits.append(rng.uniform(-1, 1, size=(8, 2)))

    # 8. Rectangular grid
    grid = np.array([
        [-1.0, -0.6], [1.0, -0.6], [-0.5, 0.6], [0.5, 0.6],
        [-1.0, 0.6], [1.0, 0.6], [-0.5, -0.6], [0.5, -0.6]
    ])
    inits.append(grid)

    best_pts = octagon.copy()
    best_score = _true_score(octagon)

    def _try_lbfgs(x0, beta, maxiter=2000):
        """Run L-BFGS-B from x0 with given beta."""
        nonlocal best_pts, best_score
        try:
            result = minimize(
                _neg_log_score_surrogate,
                x0.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            candidate = result.x.reshape(N, 2)
            cand_score = _true_score(candidate)
            if cand_score > best_score:
                best_pts = candidate.copy()
                best_score = cand_score
                return True
        except Exception:
            pass
        return False

    def _try_nelder_mead(x0, maxiter=5000):
        """Run Nelder-Mead on the exact (non-surrogate) objective."""
        nonlocal best_pts, best_score
        try:
            result = minimize(
                _neg_log_score_exact,
                x0.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10,
                         'adaptive': True}
            )
            candidate = result.x.reshape(N, 2)
            cand_score = _true_score(candidate)
            if cand_score > best_score:
                best_pts = candidate.copy()
                best_score = cand_score
                return True
        except Exception:
            pass
        return False

    # ---- Phase 1: Coarse L-BFGS-B from each initialization ----
    for init in inits:
        if time.time() > deadline - 30:
            break
        _try_lbfgs(init, beta=50.0, maxiter=1500)

    # ---- Phase 2: Beta annealing polish (increasingly sharp soft-max) ----
    polish_betas = [100.0, 200.0, 400.0, 800.0, 1500.0, 3000.0, 6000.0]
    for beta in polish_betas:
        if time.time() > deadline - 20:
            break
        _try_lbfgs(best_pts, beta=beta, maxiter=3000)

    # ---- Phase 3: Nelder-Mead polish on exact objective ----
    if time.time() < deadline - 15:
        _try_nelder_mead(best_pts, maxiter=5000)

    # ---- Phase 4: Random restarts with L-BFGS-B + Nelder-Mead ----
    rng2 = np.random.default_rng(123)
    for restart in range(40):
        if time.time() > deadline - 10:
            break
        noise = rng2.normal(0, 0.15, size=(8, 2))
        x0 = best_pts + noise
        _try_lbfgs(x0, beta=500.0, maxiter=2000)
        if time.time() > deadline - 10:
            break
        _try_nelder_mead(best_pts, maxiter=3000)

    # ---- Phase 5: Final Nelder-Mead polish ----
    if time.time() < deadline - 5:
        _try_nelder_mead(best_pts, maxiter=10000)

    # ---- Final: scale to diameter 2 and return ----
    pts = _scale_to_diameter2(best_pts)
    return [[float(p[0]), float(p[1])] for p in pts]
# EVOLVE-BLOCK-END
