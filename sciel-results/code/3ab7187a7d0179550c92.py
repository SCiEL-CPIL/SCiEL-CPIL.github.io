"""Combined program: two-square symmetric model + multi-phase polar/Cartesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_score(pts):
    """Compute Delta/n^n where Delta = product of all ordered-pair distances after scaling to diameter 2."""
    n = 8
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    diameter = np.max(valid_dists)
    if diameter < 1e-12:
        return 0.0
    log_prod = np.sum(np.log(valid_dists))
    # Delta = product / diameter^(n*(n-1))
    log_delta = log_prod - n * (n - 1) * np.log(diameter)
    score = np.exp(log_delta) / (n ** n)
    return score

def _neg_log_score(pts):
    """Negative log of (product / diameter^(n*(n-1))) for minimization."""
    n = 8
    pts = np.asarray(pts, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    diameter = np.max(valid_dists)
    if diameter < 1e-12:
        return 1e10
    log_prod = np.sum(np.log(valid_dists))
    return -log_prod + n * (n - 1) * np.log(diameter)

def _two_square_points(params):
    """
    Generate 8 points as two concentric squares.
    params = [r1, r2, theta1, theta2]
    Square 1: radius r1, rotation theta1, 4 points at angles theta1 + k*pi/2
    Square 2: radius r2, rotation theta2, 4 points at angles theta2 + k*pi/2
    """
    r1, r2, theta1, theta2 = params
    angles1 = theta1 + np.pi / 2 * np.arange(4)
    angles2 = theta2 + np.pi / 2 * np.arange(4)
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    return np.vstack([pts1, pts2])

def _neg_log_score_two_square(params):
    """Objective for two-square parameterization."""
    pts = _two_square_points(params)
    return _neg_log_score(pts)

def _neg_log_score_polar(polar_flat):
    """Objective in polar coordinates (r, theta) for each point."""
    n = 8
    polar = polar_flat.reshape(n, 2)
    radii = np.abs(polar[:, 0])
    thetas = polar[:, 1]
    pts = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
    return _neg_log_score(pts)

def _polar_to_cart(polar_flat):
    polar = polar_flat.reshape(8, 2)
    radii = np.abs(polar[:, 0])
    thetas = polar[:, 1]
    return np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])

def _cart_to_polar(pts):
    x, y = pts[:, 0], pts[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.column_stack([r, theta]).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    n = 8
    # Regular octagon baseline
    angles = 2 * np.pi * np.arange(n) / n
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _compute_score(octagon)
    best_polar = _cart_to_polar(octagon)
    best_two_sq = np.array([1.0, 1.0, 0.0, np.pi / 4.0])  # r1=1, r2=1, theta1=0, theta2=pi/4 (octagon)

    # --- Phase 1: Grid search over two-square parameters (fast, 4D) ---
    # r1, r2 in [0.1, 3.0], theta1, theta2 in [0, pi/2) (symmetry: pi/2 period for squares)
    r_vals = np.linspace(0.1, 3.0, 25)
    t_vals = np.linspace(0, np.pi / 2, 25)
    grid_best_val = 1e10
    grid_best_params = best_two_sq.copy()
    for r1 in r_vals:
        for r2 in r_vals:
            for t1 in t_vals:
                for t2 in t_vals:
                    params = np.array([r1, r2, t1, t2])
                    val = _neg_log_score_two_square(params)
                    if val < grid_best_val:
                        grid_best_val = val
                        grid_best_params = params.copy()
    if grid_best_val < _neg_log_score(best_pts):
        cand_pts = _two_square_points(grid_best_params)
        cand_score = _compute_score(cand_pts)
        if cand_score > best_score:
            best_pts = cand_pts
            best_score = cand_score
            best_polar = _cart_to_polar(cand_pts)
            best_two_sq = grid_best_params.copy()

    # --- Phase 2: Local refinement of two-square parameters ---
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _neg_log_score_two_square,
                best_two_sq,
                method='L-BFGS-B',
                bounds=[(0.01, 5.0), (0.01, 5.0), (0.0, np.pi / 2), (0.0, np.pi / 2)],
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < _neg_log_score_two_square(best_two_sq):
                cand_pts = _two_square_points(res.x)
                cand_score = _compute_score(cand_pts)
                if cand_score > best_score:
                    best_pts = cand_pts
                    best_score = cand_score
                    best_polar = _cart_to_polar(cand_pts)
                    best_two_sq = res.x.copy()
        except Exception:
            pass

    # --- Phase 3: Differential Evolution in polar coordinates (global exploration) ---
    if time.time() < deadline - 50:
        polar_bounds = [(0.05, 4.0)] * 8 + [(0.0, 2.0 * np.pi)] * 8
        try:
            de_result = differential_evolution(
                _neg_log_score_polar,
                bounds=polar_bounds,
                seed=42,
                maxiter=150,
                tol=1e-8,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            cand_pts = _polar_to_cart(de_result.x)
            cand_score = _compute_score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
                best_polar = de_result.x.copy()
        except Exception:
            pass

    # --- Phase 4: L-BFGS-B local refinement in polar coordinates ---
    if time.time() < deadline - 20:
        try:
            res = minimize(
                _neg_log_score_polar,
                best_polar,
                method='L-BFGS-B',
                bounds=polar_bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand_pts = _polar_to_cart(res.x)
            cand_score = _compute_score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
                best_polar = res.x.copy()
        except Exception:
            pass

    # --- Phase 5: L-BFGS-B in Cartesian coordinates ---
    if time.time() < deadline - 10:
        try:
            cart_flat = best_pts.flatten()
            res = minimize(
                _neg_log_score,
                cart_flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand_pts = res.x.reshape(8, 2)
            cand_score = _compute_score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
                best_polar = _cart_to_polar(cand_pts)
        except Exception:
            pass

    # --- Phase 6: Perturbation restarts in polar space ---
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        for i in range(15):
            if time.time() > deadline - 5:
                break
            noise_r = rng.normal(0, 0.3, size=8)
            noise_t = rng.normal(0, 0.4, size=8)
            perturbed = best_polar.copy()
            perturbed[:8] = np.clip(perturbed[:8] + noise_r, 0.05, 4.0)
            perturbed[8:] = (perturbed[8:] + noise_t) % (2 * np.pi)
            try:
                res = minimize(
                    _neg_log_score_polar,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=polar_bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                cand_pts = _polar_to_cart(res.x)
                cand_score = _compute_score(cand_pts)
                if cand_score > best_score:
                    best_pts = cand_pts
                    best_score = cand_score
                    best_polar = res.x.copy()
            except Exception:
                pass

    # --- Phase 7: Final high-precision polish ---
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_log_score_polar,
                best_polar,
                method='L-BFGS-B',
                bounds=polar_bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            cand_pts = _polar_to_cart(res.x)
            cand_score = _compute_score(cand_pts)
            if cand_score > best_score:
                best_pts = cand_pts
                best_score = cand_score
        except Exception:
            pass

    # Final scaling to diameter 2
    pts = np.array(best_pts, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = ~np.eye(n, dtype=bool)
    diameter = np.max(dists[mask])
    if diameter > 1e-10:
        pts = pts * (2.0 / diameter)

    return [[float(x), float(y)] for x, y in pts]

# EVOLVE-BLOCK-END
