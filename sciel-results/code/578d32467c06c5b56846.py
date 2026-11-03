"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    # Best known packing for N=50 has radius ~0.1258254895304
    # Structure: 1 center, 6 ring 1, 12 ring 2, 12 ring 3, 19 ring 4
    ring_sizes = [6, 12, 12, 19]
    n_rings = len(ring_sizes)
    n_params = 2 * n_rings  # radial distances + angular offsets

    def get_centers(params):
        centers = [[0.0, 0.0]]
        for i in range(n_rings):
            r = max(params[i], 0.01)
            offset = params[i + n_rings]
            n = ring_sizes[i]
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            centers.extend(zip(xs, ys))
        return np.array(centers)

    def compute_radius(centers):
        if len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        norms = np.linalg.norm(centers, axis=1)
        min_margin = 1.0 - np.max(norms)
        if min_margin <= 0:
            return 0.0
        # Minimum pair distance
        min_dist = 2.0
        for i in range(N):
            d = np.linalg.norm(centers - centers[i], axis=1)
            if i > 0:
                min_dist = min(min_dist, np.min(d[:i]))
            else:
                min_dist = min(min_dist, np.min(d[1:]))
        return min(min_dist / 2.0, min_margin)

    def objective(params):
        radii = np.abs(params[:n_rings])
        radii = np.clip(radii, 0.01, 0.98)
        p = np.concatenate([radii, params[n_rings:]])
        centers = get_centers(p)
        norms = np.linalg.norm(centers, axis=1)
        max_norm = np.max(norms)
        if max_norm >= 1.0:
            return 1.0 + (max_norm - 1.0)
        # Pair distances
        min_pair = 2.0
        for i in range(N):
            d = np.linalg.norm(centers - centers[i], axis=1)
            if i > 0:
                min_pair = min(min_pair, np.min(d[:i]))
            else:
                min_pair = min(min_pair, np.min(d[1:]))
        margin = 1.0 - max_norm
        # Smooth approximation of min(min_pair/2, margin)
        a = min_pair / 2.0
        b = margin
        beta = 100.0  # inverse temperature for soft-min
        soft_min = -np.logaddexp(-beta * a, -beta * b) / beta
        return -soft_min

    # Initial guesses based on known packing structure
    initial_params_list = []
    
    # Guess 1: based on known structure
    initial_params_list.append(np.array([0.252, 0.45, 0.65, 0.85,
                                         0.0, np.pi / 12, np.pi / 6, 0.0]))
    
    # Guess 2: slightly different offsets
    initial_params_list.append(np.array([0.25, 0.46, 0.66, 0.86,
                                         np.pi / 6, np.pi / 12, 0.0, np.pi / 12]))
    
    # Guess 3: perturbed
    np.random.seed(42)
    for _ in range(4):
        p = initial_params_list[0].copy()
        p += np.random.randn(n_params) * 0.03
        p[:n_rings] = np.clip(np.abs(p[:n_rings]), 0.05, 0.95)
        initial_params_list.append(p)

    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    best_params = initial_params_list[0].copy()
    best_score = -objective(best_params)
    best_centers = get_centers(best_params)

    methods = ['Nelder-Mead', 'Powell']

    for method in methods:
        for init in initial_params_list:
            if time.time() - start_time > time_limit - 15.0:
                break
            try:
                options = {}
                if method == 'Nelder-Mead':
                    options = {'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                elif method == 'Powell':
                    options = {'maxiter': 2000, 'ftol': 1e-10, 'xtol': 1e-10}
                res = minimize(objective, init, method=method, options=options)
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
                    best_params[:n_rings] = np.clip(np.abs(best_params[:n_rings]), 0.05, 0.95)
                    best_centers = get_centers(best_params)
                    # Verify
                    dist_to_center = np.linalg.norm(best_centers, axis=1)
                    if np.max(dist_to_center) < 1.0:
                        current_r = compute_radius(best_centers)
                        if current_r > 0:
                            best_score = -current_r
            except Exception:
                pass
        if time.time() - start_time > time_limit - 15.0:
            break

    # Final high-precision polish
    if time.time() - start_time < time_limit - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_params[:n_rings] = np.clip(np.abs(res.x[:n_rings]), 0.05, 0.95)
                best_params[n_rings:] = res.x[n_rings:]
                best_centers = get_centers(best_params)
                dist_to_center = np.linalg.norm(best_centers, axis=1)
                if np.max(dist_to_center) < 1.0:
                    current_r = compute_radius(best_centers)
                    if current_r > 0:
                        best_score = -current_r
        except Exception:
            pass

    # Final validation
    dist_to_center = np.linalg.norm(best_centers, axis=1)
    if np.max(dist_to_center) >= 1.0 or len(best_centers) != N or not np.all(np.isfinite(best_centers)):
        # Fallback to hexagonal
        return _fallback_hexagonal()

    return best_centers


def _fallback_hexagonal():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

# EVOLVE-BLOCK-END
