import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    """
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # Safety margin before 630s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if np.any(dists == 0):
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(p):
        """Center and normalize scale for stability."""
        p = p - np.mean(p, axis=0)
        norm_sq = np.sum(p ** 2)
        if norm_sq < 1e-15:
            return p
        return p * np.sqrt(n / norm_sq)

    def objective(x, beta):
        """
        Minimize soft_max(log(d)) - soft_min(log(d))
        using logsumexp for numerical stability.
        Points are centered and normalized to remove invariances.
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            return 1e10

        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        # Center the points to remove translational invariance
        p = p - np.mean(p, axis=0)

        # Normalize scale to remove scaling invariance
        norm_sq = np.sum(p ** 2)
        if norm_sq < 1e-12:
            return 1e10 + 1.0 / norm_sq

        # Fix scale: sum(p^2) = n
        p = p * np.sqrt(n / norm_sq)

        dists = pdist(p, metric='euclidean')

        min_dist = np.min(dists)
        if min_dist < 1e-12:
            return 1e10 + 1.0 / min_dist

        log_dists = np.log(dists)

        # Soft max/min of log_dists
        bm = beta * log_dists
        bmin = -beta * log_dists

        soft_max_val = (1.0 / beta) * logsumexp(bm)
        soft_min_val = -(1.0 / beta) * logsumexp(bmin)

        return soft_max_val - soft_min_val

    def stochastic_grad_ascent(p, steps=50, lr=0.01, beta=100.0):
        """
        Stochastic gradient ascent on the soft-min objective.
        Maximizes soft_min(log_dists) - soft_max(log_dists)
        """
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:
            return p

        best_p = p.copy()
        best_obj = objective(p.ravel(), beta)

        for step in range(steps):
            if time.time() - start_time > time_limit - 10:
                break

            p = p - np.mean(p, axis=0)
            norm_sq = np.sum(p ** 2)
            if norm_sq < 1e-12:
                break
            p = p * np.sqrt(n / norm_sq)

            dists = pdist(p, metric='euclidean')
            min_dist = np.min(dists)
            if min_dist < 1e-12:
                break

            log_dists = np.log(dists)

            # Compute gradients numerically (finite differences)
            # We want to maximize: -(soft_max - soft_min) = soft_min - soft_max
            # So we minimize: soft_max - soft_min
            # Gradient of objective w.r.t. p

            grad = np.zeros_like(p)
            eps = 1e-5
            for i in range(n):
                for j in range(d):
                    p_plus = p.copy()
                    p_plus[i, j] += eps
                    p_minus = p.copy()
                    p_minus[i, j] -= eps

                    f_plus = objective(p_plus.ravel(), beta)
                    f_minus = objective(p_minus.ravel(), beta)

                    if f_plus < 1e9 and f_minus < 1e9:
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0

            # Gradient ascent: move in direction of -grad (since we minimize)
            p_new = p - lr * grad
            p_new = normalize_pts(p_new)

            new_obj = objective(p_new.ravel(), beta)
            if new_obj < 1e9 and new_obj < best_obj:
                best_obj = new_obj
                best_p = p_new.copy()

            p = p_new

        return best_p

    # Initial configurations
    initializations = []

    # 1. Icosahedron + 2 poles
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
    poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    initializations.append(np.vstack([ico_verts, poles]))

    # 2. Fibonacci sphere
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        radius = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        fib_pts[i] = [x, y, z]
    initializations.append(fib_pts)

    # 3. Random points on sphere (multiple seeds)
    for seed in [42, 123, 456, 789, 101, 202, 303]:
        np.random.seed(seed)
        rand_pts = np.random.randn(n, d)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
        initializations.append(rand_pts)

    best_points = initializations[0].copy()
    best_score = get_score(best_points)

    # Phase 1: L-BFGS-B optimization with increasing beta
    beta_schedule = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > time_limit - 90:
            break

        try:
            current_pts = init_pts.copy()
            for beta_val in beta_schedule:
                if time.time() - start_time > time_limit - 60:
                    break

                res = minimize(objective, current_pts.ravel(), method='L-BFGS-B',
                               args=(beta_val,),
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})

                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        opt_points = normalize_pts(opt_points)
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
                        current_pts = opt_points.copy()

        except Exception:
            pass

    # Phase 2: Stochastic gradient ascent from best point
    if time.time() - start_time < time_limit - 60:
        try:
            sga_points = stochastic_grad_ascent(best_points, steps=100, lr=0.01, beta=300.0)
            sga_points = normalize_pts(sga_points)
            sga_score = get_score(sga_points)
            if sga_score > best_score:
                best_score = sga_score
                best_points = sga_points.copy()
        except Exception:
            pass

    # Phase 3: High-precision L-BFGS-B polish with very large beta
    if time.time() - start_time < time_limit - 40:
        try:
            for beta_val in [2000.0, 5000.0, 10000.0]:
                if time.time() - start_time > time_limit - 20:
                    break
                res = minimize(objective, best_points.ravel(), method='L-BFGS-B',
                               args=(beta_val,),
                               options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200})

                if res.fun < 1e9:
                    opt_points = res.x.reshape((n, d))
                    if np.all(np.isfinite(opt_points)):
                        opt_points = normalize_pts(opt_points)
                        opt_score = get_score(opt_points)
                        if opt_score > best_score:
                            best_score = opt_score
                            best_points = opt_points.copy()
        except Exception:
            pass

    # Phase 4: Nelder-Mead polish on the exact nonsmooth metric
    if time.time() - start_time < time_limit - 15:
        try:
            def exact_neg_score(x):
                elapsed = time.time() - start_time
                if elapsed > time_limit - 5:
                    return 0.0
                p = x.reshape((n, d))
                if not np.all(np.isfinite(p)):
                    return 0.0
                p = normalize_pts(p)
                return -get_score(p)

            res_nm = minimize(exact_neg_score, best_points.ravel(), method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            nm_points = res_nm.x.reshape((n, d))
            if np.all(np.isfinite(nm_points)):
                nm_points = normalize_pts(nm_points)
                nm_score = get_score(nm_points)
                if nm_score > best_score:
                    best_score = nm_score
                    best_points = nm_points.copy()
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_points)):
        best_points = initializations[0].copy()

    return best_points
