"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    
    def log_score(pts):
        """Compute log of the score Delta/n^n where Delta is product of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Product over ordered pairs = (product over unordered pairs)^2
        log_delta = 2.0 * np.sum(np.log(d_ij))
        log_score_val = log_delta - n * np.log(n)
        return log_score_val, dists, d_ij
    
    def neg_log_score(pts, penalty_weight, soft_max_temp):
        """
        Objective: -log_score + penalty * smooth_max(sqrt(s_ij) - 2)
        where s_ij = squared distance. smooth_max uses log-sum-exp in log domain
        of squared distances for numerical stability.
        """
        log_s, dists, d_ij = log_score(pts)
        s = d_ij * d_ij  # squared distances, shape (num_pairs,)
        # We want smooth max of (sqrt(s) - 2).
        # Use log-sum-exp of (s - 4) with temperature T (in squared space).
        # smooth_max_sqrt = sqrt( (1/T) * log(sum(exp(T*(s-4)))) )  when max(s-4)>0
        # This is an approximation; a cleaner formulation:
        # Let m = smooth_max of s via LSE. Then approximate max(sqrt(s)-2) ~ sqrt(m) - 2.
        x = (s - 4.0) * soft_max_temp
        x_clipped = np.clip(x, -500, 500)
        m = x_clipped.max()
        lse = m / soft_max_temp + np.log(np.sum(np.exp(x_clipped - m)))
        # lse ≈ smooth_max(s - 4)
        m_sq = 4.0 + lse  # smooth_max(s)
        m_sq = np.maximum(m_sq, 1e-12)
        smooth_max_val = np.sqrt(m_sq) - 2.0
        penalty = penalty_weight * np.maximum(smooth_max_val, 0.0)
        return -log_s + penalty

    def neg_log_score_grad(pts, penalty_weight, soft_max_temp):
        """
        Analytical gradient of -log_score + penalty * (sqrt(smooth_max(s)) - 2)_+
        where s_ij = ||p_i - p_j||^2.
        """
        log_s, dists, d_ij = log_score(pts)

        # Gradient of log_score w.r.t. points
        grad_log = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            grad_log[i] += 2.0 * diff / (d * d)
            grad_log[j] -= 2.0 * diff / (d * d)

        # Gradient of penalty term
        s = d_ij * d_ij
        x = (s - 4.0) * soft_max_temp
        x_clipped = np.clip(x, -500, 500)
        m = x_clipped.max()
        exp_x = np.exp(x_clipped - m)
        sum_exp = np.sum(exp_x)
        weights = exp_x / sum_exp  # d/ds_i smooth_max(s) = weights[i]

        # smooth_max(s) = 4 + lse
        lse = m / soft_max_temp + np.log(sum_exp)
        m_sq = 4.0 + lse
        m_sq = np.maximum(m_sq, 1e-12)
        smooth_max_val = np.sqrt(m_sq) - 2.0

        if smooth_max_val > 0:
            # d/d(s_i) penalty = penalty_weight * (1/(2*sqrt(m_sq))) * weights[i]
            dpen_ds = penalty_weight * (0.5 / np.sqrt(m_sq)) * weights
            grad_penalty = np.zeros_like(pts)
            for idx in range(num_pairs):
                i, j = iu[0][idx], iu[1][idx]
                diff = pts[i] - pts[j]
                # d/dp_i s_ij = 2*diff, d/dp_j s_ij = -2*diff
                grad_penalty[i] += dpen_ds[idx] * 2.0 * diff
                grad_penalty[j] -= dpen_ds[idx] * 2.0 * diff
            total_grad = -grad_log + grad_penalty
        else:
            total_grad = -grad_log

        return -log_s + penalty_weight * max(smooth_max_val, 0.0), total_grad
    
    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)
    
    # Generate diverse initial points
    def generate_initial_points():
        """Generate multiple diverse initial configurations."""
        inits = []
        
        # Regular octagon
        theta = 2 * np.pi * np.arange(n) / n
        octagon = np.column_stack([np.cos(theta), np.sin(theta)])
        inits.append(octagon.copy())
        
        # Perturbed octagons
        rng = np.random.RandomState(42)
        for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
            perturbed = octagon + rng.randn(n, 2) * scale
            inits.append(perturbed.copy())
        
        # Asymmetric configurations
        # Two clusters of 4
        for angle_offset in [0, np.pi/4, np.pi/3]:
            cluster1 = np.column_stack([np.cos(theta[:4]), np.sin(theta[:4])]) * 0.7
            cluster2 = np.column_stack([np.cos(theta[4:]), np.sin(theta[4:])]) * 0.7 + np.array([0.5, 0])
            asym = np.vstack([cluster1, cluster2])
            inits.append(asym.copy())
        
        # Random points in unit disk
        for _ in range(5):
            angles = rng.uniform(0, 2*np.pi, n)
            radii = rng.uniform(0.5, 1.0, n)
            rand_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            inits.append(rand_pts.copy())
        
        # Square with corners and midpoints (scaled)
        square = np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ])
        inits.append(square.copy())
        
        return inits
    
    best_score = -np.inf
    best_points = None
    
    # Evaluate all initial points
    for init_pts in generate_initial_points():
        if time.time() > deadline - 10:
            break
        ls, _, _ = log_score(init_pts)
        d = diameter(init_pts)
        if d > 0:
            scaled = init_pts * (2.0 / d)
            ls_scaled, _, _ = log_score(scaled)
            if ls_scaled > best_score:
                best_score = ls_scaled
                best_points = scaled.copy()
    
    # Optimization with multiple penalty weights and initial points
    # soft_max_temp is now in the squared-distance domain (T*(s-4))
    penalty_weights = [1.0, 5.0, 10.0, 50.0, 100.0]
    soft_max_temps = [20.0, 50.0, 100.0]

    attempts = []
    for init_pts in generate_initial_points()[:8]:  # Limit number of attempts
        for pw in penalty_weights[:4]:
            for smt in soft_max_temps[:2]:
                attempts.append((init_pts, pw, smt))
    
    # Sort by initial score, try best first
    scored_attempts = []
    for init_pts, pw, smt in attempts:
        ls, _, _ = log_score(init_pts)
        d = diameter(init_pts)
        if d > 0:
            scaled = init_pts * (2.0 / d)
            ls_scaled, _, _ = log_score(scaled)
            scored_attempts.append((ls_scaled, scaled, pw, smt))
        else:
            scored_attempts.append((ls, init_pts, pw, smt))
    
    scored_attempts.sort(key=lambda x: -x[0])
    
    for ls_init, init_pts, pw, smt in scored_attempts:
        if time.time() > deadline - 15:
            break
        
        x0 = init_pts.flatten()
        
        try:
            def objective_and_grad(x):
                if time.time() > deadline - 5:
                    return np.inf, np.ones_like(x)
                pts = x.reshape(n, 2)
                return neg_log_score_grad(pts, pw, smt)
            
            result = minimize(
                lambda x: neg_log_score_grad(x.reshape(n, 2), pw, smt)[0],
                x0,
                jac=lambda x: neg_log_score_grad(x.reshape(n, 2), pw, smt)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if time.time() > deadline - 5:
                continue
            
            opt_pts = result.x.reshape(n, 2)
            
            # Scale to diameter 2
            d = diameter(opt_pts)
            if d > 1e-10:
                opt_pts = opt_pts * (2.0 / d)
            
            opt_ls, _, _ = log_score(opt_pts)
            
            if opt_ls > best_score:
                best_score = opt_ls
                best_points = opt_pts.copy()
                
        except Exception:
            continue
    
    # Final polish with the best point found
    if best_points is not None and time.time() < deadline - 20:
        for pw in [10.0, 50.0, 100.0]:
            for smt in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 12:
                    break
                x0 = best_points.flatten()
                try:
                    result = minimize(
                        lambda x: neg_log_score_grad(x.reshape(n, 2), pw, smt)[0],
                        x0,
                        jac=lambda x: neg_log_score_grad(x.reshape(n, 2), pw, smt)[1].flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
                    )
                    opt_pts = result.x.reshape(n, 2)
                    d = diameter(opt_pts)
                    if d > 1e-10:
                        opt_pts = opt_pts * (2.0 / d)
                    opt_ls, _, _ = log_score(opt_pts)
                    if opt_ls > best_score:
                        best_score = opt_ls
                        best_points = opt_pts.copy()
                except Exception:
                    continue

    # Nelder-Mead final polish: pins diameter to exactly 2 by rescaling each iteration
    if best_points is not None and time.time() < deadline - 8:
        def nm_objective(x):
            if time.time() > deadline - 4:
                return 0.0
            pts = x.reshape(n, 2)
            d = diameter(pts)
            if d < 1e-10:
                return 0.0
            scaled = pts * (2.0 / d)
            ls, _, _ = log_score(scaled)
            return -ls

        x0 = best_points.flatten()
        try:
            nm_result = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_pts = nm_result.x.reshape(n, 2)
            d = diameter(nm_pts)
            if d > 1e-10:
                nm_pts = nm_pts * (2.0 / d)
            nm_ls, _, _ = log_score(nm_pts)
            if nm_ls > best_score:
                best_score = nm_ls
                best_points = nm_pts.copy()
        except Exception:
            pass
    
    # Ensure best_points is valid
    if best_points is None:
        theta = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Final scaling to diameter 2
    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)
    
    # Verify all points are distinct and coordinates are bounded
    # Check for near-duplicate points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                # Nudge slightly
                best_points[j] += np.array([1e-8, 1e-8])
    
    # Ensure |coordinate| < 1e6
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points * (1e5 / max_coord)
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)
    
    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
