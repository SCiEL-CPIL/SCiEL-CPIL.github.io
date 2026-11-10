import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill

    def true_score(points):
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def smooth_repulsion_obj(x_flat):
        """Minimize sum of 1/d^2 to spread points."""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-8)
        return np.sum(1.0 / dists_sq)

    def soft_log_ratio_obj(x_flat, k=10.0):
        """
        Minimize log(d_max) - log(d_min) using soft approximations.
        log(d_max) ~ (1/k) * log(sum(exp(k * d_i^2)))
        log(d_min) ~ -(1/k) * log(sum(exp(-k * d_i^2)))
        Objective: (1/k) * [log(sum(exp(k*d^2))) + log(sum(exp(-k*d^2)))]
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        
        # Numerical stability for log-sum-exp
        # term1 = log(sum(exp(k * d_sq)))
        max_d_sq = np.max(dists_sq)
        if max_d_sq < 1e-12:
            return 1e10
        term1 = np.log(np.sum(np.exp(k * (dists_sq - max_d_sq)))) + k * max_d_sq
        
        # term2 = log(sum(exp(-k * d_sq)))
        min_d_sq = np.min(dists_sq)
        if min_d_sq < 1e-12:
            return 1e10
        term2 = np.log(np.sum(np.exp(-k * (dists_sq - min_d_sq)))) - k * min_d_sq
        
        return (term1 + term2) / k

    # DIRECT construction: symmetry-based candidate configurations
    candidates = []

    # Candidate 1: Icosahedron vertices + 2 poles
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([ico, poles]))

    # Candidate 2: Regular dodecagon in xy-plane + 2 poles
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dodec = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
    candidates.append(np.vstack([dodec, poles]))

    # Candidate 3: Two concentric hexagons (offset) + 2 poles
    for offset in [0.1, 0.2, 0.3, 0.5]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = ang1 + np.pi / 6  # offset by 30 degrees
        hex1 = np.column_stack([np.cos(ang1), np.sin(ang1), np.ones(6) * offset])
        hex2 = np.column_stack([np.cos(ang2), np.sin(ang2), np.ones(6) * (-offset)])
        all_pts = np.vstack([hex1, hex2, poles])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    # Candidate 4: Cube + Octahedron
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    all_pts = np.vstack([cube, octa])
    all_pts = normalize_rms(all_pts)
    candidates.append(all_pts)

    # Candidate 5: Random antipodal pairs
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper[:, 2] = np.abs(pts_upper[:, 2])  # ensure upper hemisphere
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        all_pts = np.vstack([pts_upper, pts_lower])
        all_pts = normalize_rms(all_pts)
        candidates.append(all_pts)

    best_score = 0.0
    best_points = None

    # Score all candidates first
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Optimization Loop
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        cand = normalize_rms(cand)

        try:
            # Phase 1: L-BFGS-B on smooth repulsion
            res1 = minimize(
                smooth_repulsion_obj,
                cand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = normalize_rms(res1.x.reshape(n, d))

            # Phase 2: L-BFGS-B on soft log-ratio with increasing sharpness
            pts2 = pts1
            for k in [5.0, 10.0, 20.0, 50.0]:
                if time.time() > deadline - 40:
                    break
                try:
                    res2 = minimize(
                        lambda x: soft_log_ratio_obj(x, k=k),
                        pts2.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    pts2 = normalize_rms(res2.x.reshape(n, d))
                except Exception:
                    continue

            # Phase 3: Nelder-Mead on true objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res3 = minimize(
                nm_obj,
                pts2.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts3 = normalize_rms(res3.x.reshape(n, d))

            s = true_score(pts3)
            if s > best_score:
                best_score = s
                best_points = pts3.copy()

        except Exception:
            continue

    # Final high-precision polish from the best point found
    if best_points is not None and time.time() < deadline - 10:
        try:
            def nm_obj_tight(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res_final = minimize(
                nm_obj_tight,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            s_final = true_score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_points = pts_final.copy()
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
