"""Combined program: structured asymmetric inits + L-BFGS-B with soft-max diameter + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 8

def _softmax_diameter(dists, beta):
    """Smooth approximation to max(dists)."""
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
    """Negative log of the scale-invariant score using exact max."""
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

    # ---- Build diverse initializations with emphasis on asymmetric structures ----
    inits = []
    rng = np.random.default_rng(7)

    # 1. Regular octagon (baseline)
    angles = 2 * np.pi * np.arange(N) / N
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(octagon.copy())

    # 2. Antipodal pairs with varying radii (breaks rotational symmetry)
    base_angles = rng.uniform(0, np.pi, 4)
    radii = rng.uniform(0.5, 1.0, 4)
    antipodal_pts = np.zeros((8, 2))
    for i in range(4):
        a = base_angles[i]
        r = radii[i]
        antipodal_pts[2*i] = [r*np.cos(a), r*np.sin(a)]
        antipodal_pts[2*i+1] = [-r*np.cos(a), -r*np.sin(a)]
    inits.append(antipodal_pts)

    # 3. Two concentric squares with optimized radius ratio
    angles_sq1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    angles_sq2 = angles_sq1 + np.pi/4
    for r2 in [0.4, 0.5, 0.6, 0.7]:
        two_sq = np.column_stack([
            np.concatenate([np.cos(angles_sq1), r2*np.cos(angles_sq2)]),
            np.concatenate([np.sin(angles_sq1), r2*np.sin(angles_sq2)])
        ])
        inits.append(two_sq)

    # 4. Alternating radii on octagon angles (different patterns)
    for pattern in [[1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
                    [1.0, 0.6, 0.9, 0.6, 1.0, 0.6, 0.9, 0.6],
                    [1.0, 0.7, 0.85, 0.7, 1.0, 0.7, 0.85, 0.7]]:
        radii_alt = np.array(pattern)
        alt_rad = np.column_stack([radii_alt * np.cos(angles), radii_alt * np.sin(angles)])
        inits.append(alt_rad)

    # 5. Three-cluster configuration
    cluster_angles = np.array([0, 0.1, -0.1, 2.0, 2.1, 1.9, 4.0, 4.1])
    cluster_radii = np.array([1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.55, 0.55])
    cluster_pts = np.column_stack([cluster_radii * np.cos(cluster_angles),
                                    cluster_radii * np.sin(cluster_angles)])
    inits.append(cluster_pts)

    # 6. Perturbed octagon with larger perturbation
    inits.append(octagon + rng.normal(0, 0.5, size=(8, 2)))

    # 7. Random points
    inits.append(rng.uniform(-1, 1, size=(8, 2)))

    # 8. Rectangular grid with varying aspect ratios
    for aspect in [0.5, 0.7, 1.0]:
        grid = np.array([
            [-1.0, -aspect*0.5], [1.0, -aspect*0.5],
            [-0.5, aspect*0.5], [0.5, aspect*0.5],
            [-1.0, aspect*0.5], [1.0, aspect*0.5],
            [-0.5, -aspect*0.5], [0.5, -aspect*0.5]
        ])
        inits.append(grid)

    # 9. Elliptical configurations
    for a, b in [(1.0, 0.8), (1.0, 0.6), (1.2, 0.9)]:
        ell_angles = 2 * np.pi * np.arange(N) / N
        ellipse = np.column_stack([a * np.cos(ell_angles), b * np.sin(ell_angles)])
        inits.append(ellipse)

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
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        # Mix of small and large perturbations
        scale = rng2.choice([0.1, 0.2, 0.3, 0.5])
        noise = rng2.normal(0, scale, size=(8, 2))
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
