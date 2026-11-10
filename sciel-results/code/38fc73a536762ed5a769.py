import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.linalg import eigh
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 580  # Safety margin

    def score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def solve_sdp():
        """
        Solve SDP relaxation with beta annealing on the Gram matrix.
        """
        pts = fibonacci_sphere(n)
        G = pts @ pts.T

        def gram_to_points(G_mat):
            G_mat = (G_mat + G_mat.T) / 2
            eigenvalues, eigenvectors = eigh(G_mat)
            idx = np.argsort(eigenvalues)[::-1][:d]
            eigenvalues_d = eigenvalues[idx]
            eigenvectors_d = eigenvectors[:, idx]
            mask = eigenvalues_d > 1e-10
            if np.sum(mask) < d:
                return None
            eigenvalues_d = eigenvalues_d[mask]
            eigenvectors_d = eigenvectors_d[:, mask]
            pts = eigenvectors_d * np.sqrt(np.maximum(eigenvalues_d, 0))
            return pts

        def make_sdp_objective(beta):
            def sdp_objective(x):
                G = np.zeros((n, n))
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        G[i, j] = x[idx]
                        G[j, i] = x[idx]
                        idx += 1

                eigvals = np.linalg.eigvalsh(G)
                if np.min(eigvals) < -1e-5:
                    return 1e6

                diag_penalty = np.sum((np.diag(G) - 1.0) ** 2)

                dist_sq = 2.0 - 2.0 * G
                np.fill_diagonal(dist_sq, 0.0)

                # Soft-min with log-sum-exp
                max_dsq = np.max(dist_sq)
                min_dist_sq = max_dsq + np.log(np.sum(np.exp(-(dist_sq - max_dsq) * beta))) / beta

                return -min_dist_sq + 0.05 * diag_penalty
            return sdp_objective

        # Initial point
        G_init = pts @ pts.T
        x0 = np.zeros(n * (n + 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i, n):
                x0[idx] = G_init[i, j]
                idx += 1

        # Beta annealing schedule
        betas = [10, 25, 50, 100, 200, 400]
        x_current = x0.copy()

        for beta in betas:
            if time.time() - start_time > time_limit - 60:
                break
            try:
                obj = make_sdp_objective(beta)
                res = minimize(
                    obj,
                    x_current,
                    method='Powell',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if res.fun < 1e5:
                    x_current = res.x
            except Exception:
                pass

        # Extract final points
        G_opt = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i, n):
                G_opt[i, j] = x_current[idx]
                G_opt[j, i] = x_current[idx]
                idx += 1

        eigvals, eigvecs = eigh(G_opt)
        eigvals = np.maximum(eigvals, 0)
        G_opt = eigvecs @ np.diag(eigvals) @ eigvecs.T

        pts_opt = gram_to_points(G_opt)
        if pts_opt is not None and np.all(np.isfinite(pts_opt)):
            return pts_opt
        return pts

    best_points = None
    best_score = 0.0

    # Try SDP approach first
    sdp_pts = solve_sdp()
    if sdp_pts is not None:
        sdp_score = score(sdp_pts)
        best_points = sdp_pts
        best_score = sdp_score

    # Fallback: Fibonacci sphere
    init_pts = fibonacci_sphere(n)
    fib_score = score(init_pts)
    if fib_score > best_score:
        best_points = init_pts
        best_score = fib_score

    # Smooth surrogate objective for local polishing
    def smooth_objective(x, beta=100.0):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        # Use soft-min on distances for smoothness
        max_d = np.max(dists)
        soft_min = max_d + np.log(np.sum(np.exp(-(dists - max_d) * beta))) / beta
        return -(soft_min / d_max) ** 2

    def exact_objective(x):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        return -(d_min / d_max) ** 2

    rng = np.random.default_rng(42)

    # Phase 1: Polish with smooth objective (more restarts, larger perturbations)
    restarts = 0
    max_restarts_smooth = 30
    while time.time() - start_time < time_limit - 30 and restarts < max_restarts_smooth:
        if restarts == 0:
            x0 = best_points.ravel()
        else:
            scale = 0.05 + 0.1 * (restarts / max_restarts_smooth)
            x0 = best_points.ravel() + rng.normal(0, scale, size=(n, d)).ravel()

        try:
            res = minimize(
                smooth_objective,
                x0,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            restarts += 1
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    # Phase 2: High-precision polish with exact objective (fewer restarts, small perturbations)
    restarts = 0
    max_restarts_exact = 20
    while time.time() - start_time < time_limit - 5 and restarts < max_restarts_exact:
        if restarts == 0:
            x0 = best_points.ravel()
        else:
            x0 = best_points.ravel() + rng.normal(0, 0.01, size=(n, d)).ravel()

        try:
            res = minimize(
                exact_objective,
                x0,
                method='Powell',
                options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            restarts += 1
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)

    return best_points
