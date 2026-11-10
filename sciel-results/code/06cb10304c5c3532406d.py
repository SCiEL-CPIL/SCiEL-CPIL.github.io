import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, minimize_scalar

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    def true_score(points):
        """Compute (d_min/d_max)^2 using squared distances."""
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def score_from_flat(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return true_score(pts)

    # Objective for minimizers: negative of the score
    def neg_score(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return -true_score(pts)

    # Smooth surrogate: minimize sum of inverse squared distances
    def smooth_obj(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        return np.sum(1.0 / dists_sq)

    # Gradient of smooth_obj via finite differences
    def smooth_obj_grad(x_flat):
        eps = 1e-7
        grad = np.zeros_like(x_flat)
        f0 = smooth_obj(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[i] += eps
            f_plus = smooth_obj(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    # Best tracking
    best_score = 0.0
    best_points = None

    def try_update(points):
        nonlocal best_score, best_points
        if points is None:
            return
        points = normalize_rms(points)
        if not np.all(np.isfinite(points)):
            return
        s = true_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # =================================================================
    # PHASE 1: Generate diverse initial candidates via symmetry models
    # =================================================================
    candidates = []

    # --- Model A: Two antipodal rings (r1, r2) with relative twist ---
    # 7 points on ring 1 at height h, 7 points on ring 2 at -h (antipodal symmetry)
    # Parameters: h (height), r1 (radius of ring 1), r2 (radius of ring 2),
    #              theta1 (angular offset), twist (relative rotation between rings)
    # With antipodal symmetry, ring 2 is the negative of ring 1 rotated by twist.
    # Actually for antipodal: point i on ring 1 at angle a_i, z=h;
    # point 7+i on ring 2 at angle a_i + twist, z=-h.
    # For 7 points per ring, equally spaced: a_i = 2*pi*i/7 + theta1

    def make_anticipodal_rings(h, r1, r2, theta1, twist):
        angles1 = np.linspace(0, 2*np.pi, 7, endpoint=False) + theta1
        ring1 = np.column_stack([
            r1 * np.cos(angles1),
            r1 * np.sin(angles1),
            np.full(7, h)
        ])
        angles2 = angles1 + twist
        ring2 = np.column_stack([
            r2 * np.cos(angles2),
            r2 * np.sin(angles2),
            np.full(7, -h)
        ])
        pts = np.vstack([ring1, ring2])
        return normalize_rms(pts)

    # Grid search over parameters for Model A
    h_vals = np.linspace(0.0, 1.0, 15)
    r_ratio_vals = np.linspace(0.2, 1.0, 12)  # r2/r1
    theta1_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)
    twist_vals = np.linspace(0, 2*np.pi/7, 7, endpoint=False)

    for h in h_vals:
        if time.time() > deadline - 50:
            break
        for rr in r_ratio_vals:
            for th1 in theta1_vals:
                for tw in twist_vals:
                    pts = make_anticipodal_rings(h, 1.0, rr, th1, tw)
                    candidates.append(pts)

    # --- Model B: icosahedron + 2 poles (as in current) ---
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(normalize_rms(np.vstack([ico, poles])))

    # --- Model C: Cube + octahedron ---
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    candidates.append(normalize_rms(np.vstack([cube, octa])))

    # --- Model D: Random points with antipodal symmetry ---
    for seed in range(20):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        pts = normalize_rms(np.vstack([pts_upper, pts_lower]))
        candidates.append(pts)

    # Score all candidates
    for cand in candidates:
        try_update(cand)

    # =================================================================
    # PHASE 2: Local optimization from best candidates
    # =================================================================
    if best_points is not None:
        # Take top few candidates to polish
        # We'll polish the best candidate and a few others
        polish_list = [best_points]

        # Add a few diverse candidates if they have decent scores
        cand_scores = [(true_score(c), c) for c in candidates]
        cand_scores.sort(key=lambda x: -x[0])
        for s, c in cand_scores[1:6]:
            if s > best_score * 0.8:  # only add if reasonably close
                polish_list.append(c)

        for init_pts in polish_list:
            if time.time() > deadline - 60:
                break

            init_pts = normalize_rms(init_pts)
            x0 = init_pts.flatten()

            # Step 2a: L-BFGS-B on smooth surrogate
            try:
                res1 = minimize(
                    smooth_obj,
                    x0,
                    jac=smooth_obj_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                pts1 = normalize_rms(res1.x.reshape(n, d))
            except Exception:
                pts1 = init_pts

            # Step 2b: Nelder-Mead on true objective
            try:
                res2 = minimize(
                    neg_score,
                    pts1.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                             'adaptive': True, 'maxfev': 10000}
                )
                pts2 = normalize_rms(res2.x.reshape(n, d))
                try_update(pts2)
            except Exception:
                pass

            # Step 2c: More NM iterations if time allows
            if time.time() < deadline - 70:
                try:
                    res3 = minimize(
                        neg_score,
                        pts2.flatten() if 'pts2' in dir() else pts1.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-17,
                                 'adaptive': True, 'maxfev': 20000}
                    )
                    pts3 = normalize_rms(res3.x.reshape(n, d))
                    try_update(pts3)
                except Exception:
                    pass

    # =================================================================
    # PHASE 3: Differential evolution for global search
    # =================================================================
    if time.time() < deadline - 100:
        try:
            # Bounds: points in [-2, 2]^3, normalization handles scale
            bounds = [(-2, 2)] * (n * d)
            res_de = differential_evolution(
                neg_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=30,
                tol=1e-12,
                mutation=(0.3, 1.0),
                recombination=0.7,
                polish=True,
                init='sobol',
                update_interval=1,
                callback=None
            )
            pts_de = normalize_rms(res_de.x.reshape(n, d))
            try_update(pts_de)
        except Exception:
            pass

    # =================================================================
    # PHASE 4: Fine-tuning with many random restarts + local polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 40:
        rng = np.random.RandomState(123)
        for restart in range(30):
            if time.time() > deadline - 20:
                break
            # Perturb the best solution
            noise_scale = 0.1 * (1.0 - restart / 40.0)  # decreasing noise
            perturbed = best_points + rng.randn(n, d) * noise_scale
            perturbed = normalize_rms(perturbed)

            try:
                res = minimize(
                    neg_score,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                             'adaptive': True, 'maxfev': 5000}
                )
                pts_r = normalize_rms(res.x.reshape(n, d))
                try_update(pts_r)
            except Exception:
                continue

    # =================================================================
    # PHASE 5: Final ultra-fine polish
    # =================================================================
    if best_points is not None and time.time() < deadline - 10:
        try:
            res_final = minimize(
                neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-18,
                         'adaptive': True, 'maxfev': 50000}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            try_update(pts_final)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    if not np.all(np.isfinite(best_points)):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    return best_points
