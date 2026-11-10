import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 595  # safety margin before 630s kill

    def true_score(points):
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        return d_min_sq / d_max_sq

    def min_dist_max_constraint_obj(x_flat, t_target=None):
        """
        Maximize the minimum pairwise squared distance, with the constraint
        that max pairwise squared distance = 1 (via scaling).
        We return -d_min_sq + penalty if d_max_sq > 1.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 1e10
        # Scale so d_max_sq = 1, then the ratio is just d_min_sq
        scale = 1.0 / d_max_sq
        scaled_dists_sq = dists_sq * scale
        return -np.min(scaled_dists_sq)

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def normalize_maxdist(points):
        """Scale so that max pairwise distance is 1."""
        dists_sq = pdist(points, metric='sqeuclidean')
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return points
        return points / np.sqrt(d_max_sq)

    def soft_max_dist_sq(dists_sq, k=50.0):
        """Smooth approximation of max(dists_sq) using log-sum-exp."""
        m = np.max(dists_sq)
        if m < 1e-15:
            return 0.0
        shifted = dists_sq / k - m / k
        return k * (m / k + np.log(np.sum(np.exp(shifted))))

    def soft_min_dist_sq(dists_sq, k=50.0):
        """Smooth approximation of min(dists_sq) using -log-sum-exp(-d_i^2/k)."""
        m = np.min(dists_sq)
        if m < 1e-15:
            return 0.0
        shifted = -dists_sq / k + m / k
        return k * (-m / k - np.log(np.sum(np.exp(shifted))))

    def smooth_ratio_obj(x_flat, k=20.0):
        """Minimize -soft_min/soft_max to maximize the ratio."""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_max_sq = soft_max_dist_sq(dists_sq, k)
        d_min_sq = soft_min_dist_sq(dists_sq, k)
        if d_max_sq < 1e-10:
            return 0.0
        return -d_min_sq / d_max_sq

    def repulsion_obj(x_flat):
        """Minimize sum of 1/d^2 to spread points (Fekete-like)."""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        return np.sum(1.0 / dists_sq)

    def fokete_obj(x_flat):
        """Fekete: maximize product of pairwise distances = minimize -log(sum 1/d^2) weighted."""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        return -np.sum(np.log(dists_sq))

    def barrier_obj(x_flat, mu=0.1):
        """
        Barrier-augmented: maximize d_min^2 with constraint d_max^2 <= 1.
        Use log barrier: -mu*log(1 - d_max^2) if d_max^2 < 1, else large penalty.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 1e10
        if d_max_sq < 1.0 - 1e-4:
            barrier = -mu * np.log(1.0 - d_max_sq)
        else:
            barrier = mu * (d_max_sq - 1.0)**2 + 100.0 * (d_max_sq - 1.0)
        return -d_min_sq + barrier

    def smooth_ratio_direct(x_flat, k=10.0):
        """
        Direct smooth ratio: soft_min / soft_max of squared distances.
        This is scale-invariant and directly optimizes the ratio.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_max_sq = soft_max_dist_sq(dists_sq, k)
        d_min_sq = soft_min_dist_sq(dists_sq, k)
        if d_max_sq < 1e-10:
            return 0.0
        return -d_min_sq / d_max_sq

    def alternating_min_max(x_flat):
        """
        Alternating optimization:
        1. Fix the pair with min distance, optimize all other points to increase it
        2. Fix the pair with max distance, optimize to decrease it
        Returns a combined objective for BFGS.
        """
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        ratio = d_min_sq / d_max_sq
        # Add a small regularization to avoid degeneracy
        return -ratio

    def epigraph_obj(x_flat):
        """
        Objective for epigraph formulation:
        Maximize t such that all pairwise d^2 >= t and all pairwise d^2 <= 1.
        We minimize -t + penalty for violations.
        x_flat has 42 point coords + 1 t value (last element).
        """
        pts = x_flat[:n*d].reshape(n, d)
        t = x_flat[n*d]
        
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        
        # Penalty for d^2 < t (want all >= t)
        viol_min = np.maximum(0, t - dists_sq)
        penalty_min = np.sum(viol_min**2)
        
        # Penalty for d^2 > 1 (want all <= 1)
        viol_max = np.maximum(0, dists_sq - 1.0)
        penalty_max = np.sum(viol_max**2)
        
        return -t + 100.0 * (penalty_min + penalty_max)

    # Build diverse initial candidates
    candidates = []

    # 1. Cuboctahedron vertices (12) + 2 poles (±z)
    cuboct = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    cuboct = cuboct / np.sqrt(2.0)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([cuboct, poles]))

    # 2. Icosahedron (12) + 2 poles
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    candidates.append(np.vstack([ico, poles]))

    # 3. Dodecagon in xy-plane + 2 poles
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dodec = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
    candidates.append(np.vstack([dodec, poles]))

    # 4. Two hexagons at different z levels + 2 poles
    for z_off in [0.3, 0.5, 0.7]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = ang1 + np.pi / 6
        hex1 = np.column_stack([np.cos(ang1), np.sin(ang1), np.ones(6) * z_off])
        hex2 = np.column_stack([np.cos(ang2), np.sin(ang2), np.ones(6) * (-z_off)])
        all_pts = np.vstack([hex1, hex2, poles])
        candidates.append(normalize_rms(all_pts))

    # 5. Cube (8) + octahedron (6)
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    candidates.append(normalize_rms(np.vstack([cube, octa])))

    # 6. Antipodal pairs: 7 points on upper hemisphere + their antipodes
    for seed in range(8):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper[:, 2] = np.abs(pts_upper[:, 2]) + 0.3  # ensure upper hemisphere
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        all_pts = np.vstack([pts_upper, pts_lower])
        candidates.append(normalize_rms(all_pts))

    # 7. Tilted dodecagon + poles
    for tilt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        dodec_t = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
        c, s = np.cos(tilt), np.sin(tilt)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        dodec_t = dodec_t @ R.T
        all_pts = np.vstack([dodec_t, poles])
        candidates.append(normalize_rms(all_pts))

    # 8. Cuboctahedron with tilted poles
    for pole_angle in [0.1, 0.2, 0.3, 0.5]:
        p1 = np.array([np.sin(pole_angle), 0.0, np.cos(pole_angle)])
        p2 = -p1
        all_pts = np.vstack([cuboct, p1.reshape(1,3), p2.reshape(1,3)])
        candidates.append(normalize_rms(all_pts))

    # 9. Icosahedron with moved poles
    for z_pole in [0.6, 0.8, 1.0, 1.3, 1.5]:
        p1 = np.array([0.0, 0.0, z_pole])
        p2 = np.array([0.0, 0.0, -z_pole])
        all_pts = np.vstack([ico, p1.reshape(1,3), p2.reshape(1,3)])
        candidates.append(normalize_rms(all_pts))

    # 10. Random starts
    for seed in range(10):
        rng = np.random.RandomState(100 + seed)
        pts = rng.randn(n, 3)
        candidates.append(normalize_rms(pts))

    # Score all candidates
    best_score = 0.0
    best_points = None
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Sort candidates by score and optimize top ones first
    scored_cands = []
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        scored_cands.append((s, cand))
    scored_cands.sort(key=lambda x: -x[0])

    # Optimize top candidates
    num_candidates = min(len(scored_cands), 20)  # limit to save time
    for i in range(num_candidates):
        if time.time() > deadline - 30:
            break

        cand = scored_cands[i][1]
        try:
            # Phase 1: Fekete (log-product) optimization directly
            res_f = minimize(
                fokete_obj,
                cand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = normalize_rms(res_f.x.reshape(n, d))

            # Phase 2: Smooth ratio optimization with multiple k values
            best_phase2_pts = pts1
            best_phase2_score = true_score(pts1)
            for k in [5.0, 10.0, 20.0]:
                if time.time() > deadline - 60:
                    break
                try:
                    res2 = minimize(
                        lambda x, k=k: smooth_ratio_direct(x, k),
                        pts1.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    pts2 = normalize_rms(res2.x.reshape(n, d))
                    s2 = true_score(pts2)
                    if s2 > best_phase2_score:
                        best_phase2_score = s2
                        best_phase2_pts = pts2
                except Exception:
                    continue

            # Phase 3: Barrier optimization
            pts3_init = normalize_maxdist(best_phase2_pts)
            best_barrier_pts = best_phase2_pts
            best_barrier_score = best_phase2_score
            for mu in [0.01, 0.05, 0.1, 0.3]:
                if time.time() > deadline - 80:
                    break
                try:
                    res_b = minimize(
                        lambda x, mu=mu: barrier_obj(x, mu),
                        pts3_init.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}
                    )
                    pts_b = normalize_rms(res_b.x.reshape(n, d))
                    s_b = true_score(pts_b)
                    if s_b > best_barrier_score:
                        best_barrier_score = s_b
                        best_barrier_pts = pts_b
                except Exception:
                    continue

            # Phase 4: Direct min-dist with max-constraint
            res_md = minimize(
                min_dist_max_constraint_obj,
                best_barrier_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            pts_md = normalize_rms(res_md.x.reshape(n, d))
            s_md = true_score(pts_md)
            if s_md > best_barrier_score:
                best_barrier_pts = pts_md
                best_barrier_score = s_md

            # Phase 5: Nelder-Mead on true objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res_nm = minimize(
                nm_obj,
                best_barrier_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16,
                         'adaptive': True}
            )
            pts_nm = normalize_rms(res_nm.x.reshape(n, d))
            s_nm = true_score(pts_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_points = pts_nm.copy()

        except Exception:
            continue

    # Additional local search from best solution
    if best_points is not None and time.time() < deadline - 90:
        # Deterministic perturbation seeds
        local_seed_base = 42
        for outer_iter in range(8):
            if time.time() > deadline - 40:
                break
            try:
                for pert_size in [0.02, 0.05, 0.1, 0.15]:
                    if time.time() > deadline - 50:
                        break
                    seed = local_seed_base + outer_iter * 100 + int(pert_size * 1000)
                    rng = np.random.RandomState(seed)
                    perturbed = best_points + rng.randn(n, d) * pert_size
                    perturbed = normalize_rms(perturbed)

                    # Fekete
                    res_p = minimize(
                        fokete_obj,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    pts_p = normalize_rms(res_p.x.reshape(n, d))

                    best_local_pts = pts_p
                    best_local_score = true_score(pts_p)

                    # Smooth ratio
                    for k in [5.0, 10.0]:
                        if time.time() > deadline - 60:
                            break
                        try:
                            res_sr = minimize(
                                lambda x, k=k: smooth_ratio_direct(x, k),
                                pts_p.flatten(),
                                method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                            )
                            pts_sr = normalize_rms(res_sr.x.reshape(n, d))
                            s_sr = true_score(pts_sr)
                            if s_sr > best_local_score:
                                best_local_score = s_sr
                                best_local_pts = pts_sr
                        except Exception:
                            continue

                    # Barrier
                    for mu in [0.01, 0.05]:
                        if time.time() > deadline - 70:
                            break
                        try:
                            res_lb = minimize(
                                lambda x, mu=mu: barrier_obj(x, mu),
                                best_local_pts.flatten(),
                                method='L-BFGS-B',
                                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
                            )
                            pts_lb = normalize_rms(res_lb.x.reshape(n, d))
                            s_lb = true_score(pts_lb)
                            if s_lb > best_local_score:
                                best_local_score = s_lb
                                best_local_pts = pts_lb
                        except Exception:
                            continue

                    # Nelder-Mead polish
                    def nm_obj2(x_flat):
                        pts = x_flat.reshape(n, d)
                        pts = normalize_rms(pts)
                        return -true_score(pts)

                    res_nm2 = minimize(
                        nm_obj2,
                        best_local_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16,
                                 'adaptive': True}
                    )
                    pts_final = normalize_rms(res_nm2.x.reshape(n, d))
                    s_final = true_score(pts_final)
                    if s_final > best_score:
                        best_score = s_final
                        best_points = pts_final.copy()
            except Exception:
                continue

    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 15:
        try:
            def nm_tight(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            # Tight BFGS on barrier
            res_bf = minimize(
                lambda x: barrier_obj(x, mu=0.001),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts_bf = normalize_rms(res_bf.x.reshape(n, d))
            s_bf = true_score(pts_bf)
            if s_bf > best_score:
                best_score = s_bf
                best_points = pts_bf.copy()

            # Tight BFGS on min_dist constraint
            res_md2 = minimize(
                min_dist_max_constraint_obj,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts_md2 = normalize_rms(res_md2.x.reshape(n, d))
            s_md2 = true_score(pts_md2)
            if s_md2 > best_score:
                best_score = s_md2
                best_points = pts_md2.copy()

            # Nelder-Mead with very tight tolerances
            res_final = minimize(
                nm_tight,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-17,
                         'adaptive': True}
            )
            pts_final = normalize_rms(res_final.x.reshape(n, d))
            s_final = true_score(pts_final)
            if s_final > best_score:
                best_score = s_final
                best_points = pts_final.copy()

            # One more round of smooth ratio with very low k for precision
            if time.time() < deadline - 5:
                for k in [2.0, 3.0]:
                    try:
                        res_sr_final = minimize(
                            lambda x, k=k: smooth_ratio_direct(x, k),
                            best_points.flatten(),
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                        )
                        pts_sr_f = normalize_rms(res_sr_final.x.reshape(n, d))
                        s_sr_f = true_score(pts_sr_f)
                        if s_sr_f > best_score:
                            best_score = s_sr_f
                            best_points = pts_sr_f.copy()
                    except Exception:
                        pass
        except Exception:
            pass

    # Safety checks
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback to cuboctahedron + poles
        best_points = np.vstack([cuboct, poles])
        best_points = normalize_rms(best_points)

    return best_points
