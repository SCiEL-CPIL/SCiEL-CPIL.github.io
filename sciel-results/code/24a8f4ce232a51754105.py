# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 30  # seconds before limit to stop
    
    def get_score(pts):
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def band_objective(x, gamma=0.01, beta=0.05):
        """
        Smooth objective: normalize distances by soft_max, then penalize
        deviation from the ideal band [1-gamma, 1+gamma].
        """
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        if np.all(dists == 0):
            return 1e10

        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_max <= 1e-12:
            return 1e10

        norm_dists = dists / soft_max
        deviations = np.abs(norm_dists - 1.0)
        excess = np.maximum(deviations - gamma, 0.0)
        penalty = np.sum(excess ** 2)
        
        center = np.mean(p, axis=0)
        spread = np.mean(np.sum((p - center) ** 2, axis=1))
        
        return penalty + 1e-4 * spread

    def t_objective(x):
        """
        Epigraph-style: minimize t such that all distances are in [1/t, t]
        after scaling.
        """
        p = x[:n*d].reshape((n, d))
        t = x[n*d]
        
        if not np.all(np.isfinite(p)) or not np.isfinite(t):
            return 1e10
        if t < 1e-6:
            return 1e10

        dists = pdist(p, metric='euclidean')
        d_max = np.max(dists)
        d_min = np.min(dists)
        
        if d_max <= 1e-12:
            return 1e10

        scale = d_max
        norm_dists = dists / scale
        
        lower_bound = 1.0 / t
        lower_violations = np.maximum(lower_bound - norm_dists, 0.0)
        lower_penalty = np.sum(lower_violations ** 2)
        
        t_penalty = t ** 2
        
        return t_penalty + 100.0 * lower_penalty

    def smooth_objective(x, beta=0.05):
        """
        Minimizes (d_max / d_min)^2 using soft min/max approximations.
        """
        p = x.reshape((n, d))
        
        if not np.all(np.isfinite(p)):
            return 1e10

        dists = pdist(p, metric='euclidean')
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        if soft_min <= 0 or soft_max <= 0:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        return ratio_sq

    def exact_objective(x):
        """Exact (d_max/d_min)^2 for final polish."""
        p = x.reshape((n, d))
        if not np.all(np.isfinite(p)):
            return 1e10
        dists = pdist(p, metric='euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max == 0:
            return 1e10
        return (d_max / d_min) ** 2

    def sdp_objective(G_flat, beta=0.05):
        """
        SDP-based objective: work with Gram matrix G (n x n).
        G must be PSD and have rank <= 3.
        Distances squared: d_ij^2 = G_ii + G_jj - 2*G_ij
        We minimize (d_max/d_min)^2 using soft approximations on sqrt of these.
        """
        G = G_flat.reshape((n, n))
        
        # Ensure symmetry
        G = (G + G.T) / 2.0
        
        # Project to PSD: clip negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals_clipped = np.maximum(eigvals, 0.0)
        G_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        G_psd = (G_psd + G_psd.T) / 2.0
        
        # Extract squared distances
        diag = np.diag(G_psd).copy()
        sq_dists = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * G_psd
        sq_dists = np.maximum(sq_dists, 0.0)
        
        # Get upper triangle (exclude diagonal)
        iu = np.triu_indices(n, k=1)
        sq_dists_upper = sq_dists[iu]
        
        if len(sq_dists_upper) == 0 or np.all(sq_dists_upper == 0):
            return 1e10
        
        dists = np.sqrt(sq_dists_upper)
        
        # Soft max/min on distances
        a_pos = dists / beta
        max_a_pos = np.max(a_pos)
        log_sum_pos = max_a_pos + np.log(np.sum(np.exp(a_pos - max_a_pos)))
        soft_max = beta * log_sum_pos
        
        a_neg = -dists / beta
        max_a_neg = np.max(a_neg)
        log_sum_neg = max_a_neg + np.log(np.sum(np.exp(a_neg - max_a_neg)))
        soft_min = -beta * log_sum_neg
        
        if soft_min <= 1e-12 or soft_max <= 1e-12:
            return 1e10
            
        ratio_sq = (soft_max / soft_min) ** 2
        
        # Penalty for rank > 3 (encourage low-rank)
        eigvals_psd = np.linalg.eigvalsh(G_psd)
        rank_penalty = np.sum(np.maximum(eigvals_psd[3:], 0.0) ** 2) * 0.01
        
        # Penalty for non-PSD (should be small after projection)
        psd_penalty = np.sum(np.maximum(-eigvals, 0.0) ** 2) * 1.0
        
        return ratio_sq + rank_penalty + psd_penalty

    def sdp_to_points(G):
        """Convert Gram matrix to point coordinates via eigendecomposition."""
        G = (G + G.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(G)
        
        # Take top 3 eigenvalues
        idx = np.argsort(eigvals)[::-1][:d]
        eigvals_top = eigvals[idx]
        eigvecs_top = eigvecs[:, idx]
        
        # Points: P = diag(sqrt(eigvals_top)) @ eigvecs_top.T
        # Actually: G = V diag(w) V^T, so points are V diag(sqrt(w))
        sqrt_w = np.sqrt(np.maximum(eigvals_top, 0.0))
        points = eigvecs_top * sqrt_w[np.newaxis, :]
        
        # Center
        points = points - np.mean(points, axis=0)
        
        return points

    # Initial guess: Icosahedron vertices + 2 poles, normalized
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    
    points_init = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    points_init = points_init - np.mean(points_init, axis=0)
    
    best_points = points_init.copy()
    best_score = get_score(best_points)
    
    # SDP initialization: build Gram matrix from initial points
    G_init = points_init @ points_init.T
    G_init = (G_init + G_init.T) / 2.0
    
    # 1. SDP optimization in Gram space
    if time.time() - start_time < time_limit - safety_margin:
        try:
            sdp_deadline = start_time + (time_limit * 0.25)
            
            def sdp_obj_wrapper(G_flat):
                if time.time() > sdp_deadline:
                    return 1e10
                return sdp_objective(G_flat, beta=0.1)
            
            res_sdp = minimize(
                sdp_obj_wrapper,
                G_init.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res_sdp.fun < 1e9:
                G_opt = res_sdp.x.reshape((n, n))
                cand_points = sdp_to_points(G_opt)
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 2. Global Optimization using Differential Evolution
    if time.time() - start_time < time_limit - safety_margin:
        try:
            de_deadline = start_time + (time_limit * 0.40)
            
            def de_objective(x):
                if time.time() > de_deadline:
                    return 1e10
                return smooth_objective(x, beta=0.1)
            
            res_de = differential_evolution(
                de_objective, 
                bounds=[(-5, 5)] * (n * d), 
                maxiter=30, 
                popsize=10, 
                tol=1e-5, 
                seed=42,
                init='latinhypercube'
            )
            
            if res_de.fun < 1e9:
                cand_points = res_de.x.reshape((n, d))
                cand_score = get_score(cand_points)
                if cand_score > best_score:
                    best_points = cand_points
                    best_score = cand_score
                    
        except Exception:
            pass

    # 3. Local Refinement using L-BFGS-B with adaptive beta
    candidates_to_polish = [best_points.copy(), points_init.copy()]
    
    np.random.seed(123)
    for _ in range(4):
        perturb = best_points + np.random.randn(n, d) * 0.15
        candidates_to_polish.append(perturb)
    
    def fibonacci_sphere(n_points, seed=0):
        rng = np.random.RandomState(seed)
        points = []
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / n_points
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        pts = pts - np.mean(pts, axis=0)
        return pts
    
    for s in [1, 2, 3]:
        candidates_to_polish.append(fibonacci_sphere(n, seed=s))
    
    for init_pts in candidates_to_polish:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            res_local = minimize(
                lambda x: smooth_objective(x, beta=0.05), 
                init_pts.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_local.fun < 1e9:
                opt_points = res_local.x.reshape((n, d))
                opt_score = get_score(opt_points)
                
                if opt_score > best_score:
                    best_points = opt_points
                    best_score = opt_score
                    
                    if time.time() - start_time < time_limit - safety_margin:
                        res_local2 = minimize(
                            lambda x: smooth_objective(x, beta=0.01), 
                            opt_points.ravel(), 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        if res_local2.fun < 1e9:
                            opt_points2 = res_local2.x.reshape((n, d))
                            opt_score2 = get_score(opt_points2)
                            if opt_score2 > best_score:
                                best_points = opt_points2
                                best_score = opt_score2
        except Exception:
            pass

    # 4. Band objective polish
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_band = minimize(
                lambda x: band_objective(x, gamma=0.02, beta=0.05),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_band.fun < 1e9:
                band_points = res_band.x.reshape((n, d))
                band_score = get_score(band_points)
                if band_score > best_score:
                    best_points = band_points
                    best_score = band_score
        except Exception:
            pass

    # 5. T-objective (epigraph) polish
    if time.time() - start_time < time_limit - safety_margin:
        try:
            dists_best = pdist(best_points, metric='euclidean')
            t_init = np.max(dists_best) / np.min(dists_best)
            
            x_t_init = np.concatenate([best_points.ravel(), [t_init]])
            
            res_t = minimize(
                t_objective,
                x_t_init,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * (n * d) + [(1.0, 100.0)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_t.fun < 1e9:
                t_points = res_t.x[:n*d].reshape((n, d))
                t_score = get_score(t_points)
                if t_score > best_score:
                    best_points = t_points
                    best_score = t_score
        except Exception:
            pass

    # 6. Second SDP pass from improved points
    if time.time() - start_time < time_limit - safety_margin:
        try:
            G_from_best = best_points @ best_points.T
            G_from_best = (G_from_best + G_from_best.T) / 2.0
            
            sdp_deadline2 = start_time + (time_limit * 0.70)
            
            def sdp_obj_wrapper2(G_flat):
                if time.time() > sdp_deadline2:
                    return 1e10
                return sdp_objective(G_flat, beta=0.05)
            
            res_sdp2 = minimize(
                sdp_obj_wrapper2,
                G_from_best.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res_sdp2.fun < 1e9:
                G_opt2 = res_sdp2.x.reshape((n, n))
                cand_points2 = sdp_to_points(G_opt2)
                cand_score2 = get_score(cand_points2)
                if cand_score2 > best_score:
                    best_points = cand_points2
                    best_score = cand_score2
                    
        except Exception:
            pass

    # 7. Final high-precision polish with exact objective
    if time.time() - start_time < time_limit - safety_margin:
        try:
            res_final = minimize(
                exact_objective, 
                best_points.ravel(), 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            if res_final.fun < 1e9:
                final_points = res_final.x.reshape((n, d))
                final_score = get_score(final_points)
                if final_score > best_score:
                    best_points = final_points
                    best_score = final_score
        except Exception:
            pass

    # 8. Final band polish with very tight gamma
    if time.time() - start_time < time_limit - 15:
        try:
            res_band2 = minimize(
                lambda x: band_objective(x, gamma=0.005, beta=0.02),
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_band2.fun < 1e9:
                band2_points = res_band2.x.reshape((n, d))
                band2_score = get_score(band2_points)
                if band2_score > best_score:
                    best_points = band2_points
                    best_score = band2_score
        except Exception:
            pass

    # 9. Third SDP pass with tighter beta
    if time.time() - start_time < time_limit - 15:
        try:
            G_final = best_points @ best_points.T
            G_final = (G_final + G_final.T) / 2.0
            
            sdp_deadline3 = start_time + (time_limit * 0.90)
            
            def sdp_obj_wrapper3(G_flat):
                if time.time() > sdp_deadline3:
                    return 1e10
                return sdp_objective(G_flat, beta=0.02)
            
            res_sdp3 = minimize(
                sdp_obj_wrapper3,
                G_final.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if res_sdp3.fun < 1e9:
                G_opt3 = res_sdp3.x.reshape((n, n))
                cand_points3 = sdp_to_points(G_opt3)
                cand_score3 = get_score(cand_points3)
                if cand_score3 > best_score:
                    best_points = cand_points3
                    best_score = cand_score3
                    
        except Exception:
            pass

    # 10. Try a few more random restarts if time allows
    if time.time() - start_time < time_limit - 15:
        for seed in [777, 888, 999, 1000, 1111]:
            if time.time() - start_time > time_limit - 10:
                break
            try:
                rp = fibonacci_sphere(n, seed=seed)
                res_q = minimize(
                    lambda x: smooth_objective(x, beta=0.03), 
                    rp.ravel(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-10}
                )
                if res_q.fun < 1e9:
                    qp = res_q.x.reshape((n, d))
                    qs = get_score(qp)
                    if qs > best_score:
                        best_points = qp
                        best_score = qs
                        
                        if time.time() - start_time < time_limit - 10:
                            res_q2 = minimize(
                                lambda x: band_objective(x, gamma=0.01, beta=0.03),
                                qp.ravel(),
                                method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11}
                            )
                            if res_q2.fun < 1e9:
                                qp2 = res_q2.x.reshape((n, d))
                                qs2 = get_score(qp2)
                                if qs2 > best_score:
                                    best_points = qp2
                                    best_score = qs2
            except Exception:
                pass

    # Final validity check
    if not np.all(np.isfinite(best_points)):
        return points_init

    return best_points

# EVOLVE-BLOCK-END
