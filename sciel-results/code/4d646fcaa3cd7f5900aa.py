import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

DEADLINE = 625.0


def _spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _cartesian_to_spherical(points):
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    return np.column_stack([theta, phi])


def _raw_score(points):
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-10 or d_min < 1e-10:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _distances_from_sph(x_flat):
    """Return the 91 pairwise distances from a 28-vector of (theta, phi) pairs."""
    theta = x_flat[0::2]
    phi = x_flat[1::2]
    points = _spherical_to_cartesian(theta, phi)
    D = points[:, None, :] - points[None, :, :]
    D2 = np.sum(D * D, axis=2)
    dists_mat = np.sqrt(np.maximum(D2, 0.0))
    mask = np.triu(np.ones((14, 14), dtype=bool), k=1)
    return dists_mat[mask], points, dists_mat, mask


def _score_spherical(x_flat):
    """Negative (dmin/dmax)^2 for minimization. Robust to invalid inputs."""
    try:
        dists, _, _, _ = _distances_from_sph(x_flat)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10 or d_min < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return -(ratio * ratio)
    except Exception:
        return 0.0


def _score_smooth(x_flat, temperature):
    """Smooth surrogate via log-sum-exp soft min/max. Returns (value, grad)."""
    try:
        dists, points, dists_mat, mask = _distances_from_sph(x_flat)
        if len(dists) == 0 or np.any(dists < 1e-10):
            return 0.0, np.zeros_like(x_flat)

        scale = 1.0 / temperature
        d_min_raw = np.min(dists)
        d_max_raw = np.max(dists)

        exp_neg = np.exp(-(dists - d_min_raw) * scale)
        sum_exp_neg = np.sum(exp_neg)
        soft_min = d_min_raw - temperature * np.log(sum_exp_neg)

        exp_pos = np.exp((dists - d_max_raw) * scale)
        sum_exp_pos = np.sum(exp_pos)
        soft_max = d_max_raw + temperature * np.log(sum_exp_pos)

        if soft_max < 1e-10 or soft_min < 1e-10:
            return 0.0, np.zeros_like(x_flat)

        score = (soft_min / soft_max) ** 2
        val = -score

        w_min = exp_neg / sum_exp_neg
        w_max = exp_pos / sum_exp_pos

        coeff_min = 2.0 * soft_min / (soft_max * soft_max)
        coeff_max = -2.0 * (soft_min * soft_min) / (soft_max * soft_max * soft_max)

        # cartesian gradient
        theta = x_flat[0::2]
        phi = x_flat[1::2]
        
        grad_cart = np.zeros((14, 3))
        indices = np.argwhere(mask)
        # map (i,j) -> flat index in dists
        flat_idx = np.zeros((14, 14), dtype=int)
        c = 0
        for i in range(14):
            for j in range(i + 1, 14):
                flat_idx[i, j] = c
                c += 1

        for i, j in indices:
            d_val = dists_mat[i, j]
            if d_val > 1e-10:
                unit_vec = (points[i] - points[j]) / d_val
                idx = flat_idx[i, j]
                g = (coeff_min * w_min[idx] + coeff_max * w_max[idx]) * unit_vec
                grad_cart[i] += g
                grad_cart[j] -= g

        grad_sph = np.zeros(28)
        sin_theta = np.sin(theta)
        for k in range(14):
            dtheta_vec = np.array([
                np.cos(theta[k]) * np.cos(phi[k]),
                np.cos(theta[k]) * np.sin(phi[k]),
                -sin_theta[k]
            ])
            dphi_vec = np.array([
                -sin_theta[k] * np.sin(phi[k]),
                sin_theta[k] * np.cos(phi[k]),
                0.0
            ])
            grad_sph[2 * k] = np.dot(grad_cart[k], dtheta_vec)
            grad_sph[2 * k + 1] = np.dot(grad_cart[k], dphi_vec)

        return val, grad_sph
    except Exception:
        return 0.0, np.zeros_like(x_flat)


def _generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _generate_fibonacci_sphere(n=14):
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        radius = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)


def _optimize_from_sph(x0, deadline):
    """Multi-stage local polish from a 28-vector spherical starting point."""
    if time.time() > deadline:
        cand = _spherical_to_cartesian(x0[0::2], x0[1::2])
        return cand, _raw_score(cand)

    x = x0.copy()

    # Stage 1: Smooth surrogate optimization with L-BFGS-B (faster than NM for high D)
    # We use a fixed temperature for L-BFGS-B as it requires consistent gradients
    # We will try a few temperatures in sequence, or just one good one.
    # The smooth surrogate is a good landscape to navigate.
    temps = [0.5, 0.2, 0.1, 0.05]
    for temp in temps:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda xx: _score_smooth(xx, temp)[0],
                x,
                jac=lambda xx: _score_smooth(xx, temp)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass

    # Stage 2: Exact objective, Nelder-Mead for final polish
    if time.time() < deadline:
        try:
            res = minimize(
                lambda xx: _score_spherical(xx),
                x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass

    # Stage 3: High-precision polish
    if time.time() < deadline:
        try:
            res = minimize(
                lambda xx: _score_spherical(xx),
                x,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                x = res.x
        except Exception:
            pass

    cand = _spherical_to_cartesian(x[0::2], x[1::2])
    return cand, _raw_score(cand)


def min_max_dist_dim3_14():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_score = 0.0
    best_points = None

    rng = np.random.default_rng(7)

    # Build diverse initial guesses (cartesian on unit sphere)
    guesses = []
    guesses.append(_generate_icosahedron()[:12])
    p = rng.normal(size=3); p = p / np.linalg.norm(p)
    q = rng.normal(size=3); q = q / np.linalg.norm(q)
    guesses[0] = np.vstack([guesses[0], p, q])

    guesses.append(_generate_fibonacci_sphere(14))

    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    cube = cube / np.linalg.norm(cube, axis=1, keepdims=True)
    guesses.append(cube)

    for _ in range(6):
        pts = rng.normal(size=(14, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        guesses.append(pts)

    # ---- GLOBAL PHASE: differential evolution over the 28D sphere space ----
    bounds = [(0.0, np.pi), (0.0, 2 * np.pi)] * 14

    def de_obj(x_flat):
        if time.time() > deadline:
            return 0.0
        return _score_spherical(x_flat)

    de_result = None
    if time.time() < deadline - 120:
        try:
            de_result = differential_evolution(
                de_obj,
                bounds,
                seed=11,
                popsize=12,
                maxiter=150,
                tol=1e-9,
                mutation=(0.4, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=False,
                updating='immediate',
                callback=None,
            )
        except Exception:
            de_result = None

    candidates = []

    if de_result is not None and de_result.x is not None and np.all(np.isfinite(de_result.x)):
        candidates.append(de_result.x)

    # Add the diverse structured starts as spherical vectors
    for g in guesses:
        try:
            g = g / np.linalg.norm(g, axis=1, keepdims=True)
            sph = _cartesian_to_spherical(g).flatten()
            candidates.append(sph)
        except Exception:
            pass

    # Add a few random spherical starts
    for _ in range(8):
        theta = rng.uniform(0.0, np.pi, size=14)
        phi = rng.uniform(0.0, 2 * np.pi, size=14)
        candidates.append(np.column_stack([theta, phi]).flatten())

    # ---- LOCAL PHASE: polish each candidate, keep best ----
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        try:
            pts, score = _optimize_from_sph(cand, deadline)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue

    # ---- FILL REMAINING TIME with extra DE restarts / random starts ----
    extra_seeds = [23, 37, 53, 71, 97]
    for seed in extra_seeds:
        if time.time() > deadline - 120:
            break
        try:
            de_r = differential_evolution(
                de_obj,
                bounds,
                seed=seed,
                popsize=10,
                maxiter=80,
                tol=1e-9,
                mutation=(0.4, 1.0),
                recombination=0.8,
                init='latinhypercube',
                polish=False,
                updating='immediate',
            )
            if de_r.x is not None and np.all(np.isfinite(de_r.x)):
                pts, score = _optimize_from_sph(de_r.x, deadline)
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            continue

    # Random-start local polish to spend spare time
    while time.time() < deadline - 20:
        theta = rng.uniform(0.0, np.pi, size=14)
        phi = rng.uniform(0.0, 2 * np.pi, size=14)
        cand = np.column_stack([theta, phi]).flatten()
        try:
            pts, score = _optimize_from_sph(cand, deadline)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue

    # ---- FINAL POLISH on best ----
    if time.time() < deadline - 5 and best_points is not None:
        try:
            sph = _cartesian_to_spherical(best_points).flatten()
            pts, score = _optimize_from_sph(sph, deadline)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass

    # Fallbacks
    if best_points is None or not np.all(np.isfinite(best_points)) or best_score <= 0.0:
        pts = _generate_fibonacci_sphere(14)
        best_points = pts
        best_score = _raw_score(pts)

    # Ensure shape and finiteness
    best_points = np.asarray(best_points, dtype=float)
    if best_points.shape != (14, 3) or not np.all(np.isfinite(best_points)):
        best_points = _generate_fibonacci_sphere(14)

    return best_points


if __name__ == "__main__":
    pts = min_max_dist_dim3_14()
    print(_raw_score(pts))
