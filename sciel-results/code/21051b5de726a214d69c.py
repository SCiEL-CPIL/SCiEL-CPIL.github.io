"""Optimize 8 planar points to maximize product of distances at fixed diameter."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    
    def log_score(pts):
        """Compute log of Delta/n^n where Delta is product over ordered pairs of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Guard against zero distances
        d_ij_safe = np.maximum(d_ij, 1e-15)
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        log_score_val = log_delta - n * np.log(n)
        return log_score_val, dists, d_ij
    
    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)
    
    def smooth_max_penalty(pts, penalty_weight, temperature):
        """
        Compute smooth max of (d_ij - 2) using log-sum-exp in log domain.
        Returns (penalty_value, gradient).
        """
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        s = np.sum(diffs ** 2, axis=2)  # squared distances
        d_ij = np.sqrt(np.maximum(s, 1e-30))
        
        # Smooth max of (d_ij - 2) using log-sum-exp
        x = (d_ij - 2.0) * temperature
        x_clipped = np.clip(x, -500, 500)
        max_x = x_clipped.max()
        exp_x = np.exp(x_clipped - max_x)
        sum_exp = np.sum(exp_x)
        smooth_max_val = (max_x + np.log(sum_exp)) / temperature
        
        penalty_val = penalty_weight * max(smooth_max_val, 0.0)
        
        # Gradient
        weights = exp_x / sum_exp  # softmax weights
        grad = np.zeros_like(pts)
        
        if smooth_max_val > 0:
            for idx in range(num_pairs):
                i, j = iu[0][idx], iu[1][idx]
                d = d_ij[idx]
                if d < 1e-15:
                    continue
                diff = pts[i] - pts[j]
                w = penalty_weight * weights[idx]
                grad[i] += w * diff / d
                grad[j] -= w * diff / d
        
        return penalty_val, grad
    
    def objective_and_grad(pts_flat, penalty_weight, temperature):
        """Full objective: -log_score + penalty."""
        pts = pts_flat.reshape(n, 2)
        
        # Log score and its gradient
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        d_ij_safe = np.maximum(d_ij, 1e-15)
        
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        log_score_val = log_delta - n * np.log(n)
        
        # Gradient of log_score
        grad_log = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            grad_log[i] += 2.0 * diff / (d * d)
            grad_log[j] -= 2.0 * diff / (d * d)
        
        # Penalty
        pen_val, grad_pen = smooth_max_penalty(pts, penalty_weight, temperature)
        
        total_val = -log_score_val + pen_val
        total_grad = -grad_log + grad_pen
        
        return total_val, total_grad.flatten()
    
    def polish(pts, max_time=10.0):
        """Refine a point configuration using gradient optimization."""
        if time.time() > deadline - max_time:
            return pts, log_score(pts)[0]
        
        d = diameter(pts)
        if d < 1e-10:
            return pts, -np.inf
        pts = pts * (2.0 / d)
        
        best_pts = pts.copy()
        best_ls = log_score(pts)[0]
        
        # Try multiple penalty weights and temperatures
        for pw in [5.0, 20.0, 50.0, 100.0]:
            for temp in [50.0, 100.0, 200.0]:
                if time.time() > deadline - max_time:
                    break
                x0 = best_pts.flatten()
                try:
                    result = minimize(
                        lambda x: objective_and_grad(x, pw, temp)[0],
                        x0,
                        jac=lambda x: objective_and_grad(x, pw, temp)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
                    )
                    opt_pts = result.x.reshape(n, 2)
                    d_opt = diameter(opt_pts)
                    if d_opt > 1e-10:
                        opt_pts = opt_pts * (2.0 / d_opt)
                    opt_ls, _, _ = log_score(opt_pts)
                    if opt_ls > best_ls:
                        best_ls = opt_ls
                        best_pts = opt_pts.copy()
                except Exception:
                    continue
        
        return best_pts, best_ls
    
    # Generate diverse initial points
    def generate_initial_points():
        inits = []
        theta = 2 * np.pi * np.arange(n) / n
        octagon = np.column_stack([np.cos(theta), np.sin(theta)])
        inits.append(octagon.copy())
        
        rng = np.random.RandomState(12345)
        
        # Perturbed octagons with various scales
        for scale in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
            perturbed = octagon + rng.randn(n, 2) * scale
            inits.append(perturbed.copy())
        
        # Rotated octagons
        for angle in [np.pi/16, np.pi/12, np.pi/8]:
            R = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
            rotated = (octagon @ R.T).copy()
            inits.append(rotated)
        
        # Symmetric non-regular: two squares rotated
        for rot in [np.pi/8, np.pi/6, np.pi/5, np.pi/4]:
            sq1 = np.array([[1,1],[1,-1],[-1,-1],[-1,1]]) * 0.7
            R = np.array([[np.cos(rot), -np.sin(rot)],
                         [np.sin(rot), np.cos(rot)]])
            sq2 = (sq1 @ R.T).copy()
            combined = np.vstack([sq1, sq2])
            inits.append(combined)
        
        # Elliptical configurations
        for a, b in [(1.2, 0.8), (1.3, 0.7), (1.1, 0.9), (1.4, 0.6)]:
            ell = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
            inits.append(ell)
        
        # Random points in disk
        for _ in range(8):
            angles = rng.uniform(0, 2*np.pi, n)
            radii = rng.uniform(0.4, 1.0, n)
            rand_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            inits.append(rand_pts)
        
        # Square with midpoints
        square_mid = np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ])
        inits.append(square_mid)
        
        # Two clusters
        for offset in [0.3, 0.5, 0.7]:
            c1 = octagon[:4] * 0.6
            c2 = octagon[4:] * 0.6 + np.array([offset, 0])
            inits.append(np.vstack([c1, c2]))
        
        return inits
    
    best_score = -np.inf
    best_points = None
    
    # Evaluate all initial points
    for init_pts in generate_initial_points():
        if time.time() > deadline - 30:
            break
        d = diameter(init_pts)
        if d > 1e-10:
            scaled = init_pts * (2.0 / d)
            ls, _, _ = log_score(scaled)
            if ls > best_score:
                best_score = ls
                best_points = scaled.copy()
    
    # Polish the best initial point
    if best_points is not None:
        best_points, best_score = polish(best_points, max_time=15.0)
    
    # Try polishing several diverse initial points
    inits = generate_initial_points()
    # Score and sort
    scored = []
    for ip in inits:
        d = diameter(ip)
        if d > 1e-10:
            scaled = ip * (2.0 / d)
            ls, _, _ = log_score(scaled)
            scored.append((ls, scaled))
    scored.sort(key=lambda x: -x[0])
    
    for ls_init, init_pts in scored[:10]:
        if time.time() > deadline - 20:
            break
        polished, ls_polished = polish(init_pts, max_time=8.0)
        if ls_polished > best_score:
            best_score = ls_polished
            best_points = polished
    
    # Global exploration with differential evolution (if time permits)
    if time.time() < deadline - 40 and best_score < 1.3:
        try:
            bounds = [(-1.5, 1.5)] * (2 * n)
            
            def de_objective(x):
                if time.time() > deadline - 10:
                    return 0.0
                pts = x.reshape(n, 2)
                d = diameter(pts)
                if d < 1e-10:
                    return 0.0
                scaled = pts * (2.0 / d)
                ls, _, _ = log_score(scaled)
                # Penalize if diameter is not close to 2 after scaling (should be exactly 2)
                return -ls  # We want to maximize ls, so minimize -ls
            
            de_result = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=50, popsize=20,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=False, init='sohal'
            )
            
            if de_result.success and time.time() < deadline - 15:
                de_pts = de_result.x.reshape(n, 2)
                d_de = diameter(de_pts)
                if d_de > 1e-10:
                    de_pts = de_pts * (2.0 / d_de)
                    de_ls, _, _ = log_score(de_pts)
                    if de_ls > best_score:
                        best_score = de_ls
                        best_points = de_pts
                    # Polish the DE result
                    polished, ls_pol = polish(de_pts, max_time=10.0)
                    if ls_pol > best_score:
                        best_score = ls_pol
                        best_points = polished
        except Exception:
            pass
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_score = polish(best_points, max_time=5.0)
    
    # Ensure validity
    if best_points is None:
        theta = 2 * np.pi * np.arange(n) / n
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Scale to diameter 2
    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)
    
    # Check distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                best_points[j] += np.array([1e-8, 1e-8])
    
    # Bound coordinates
    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points * (1e5 / max_coord)
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)
    
    return [[float(p[0]), float(p[1])] for p in best_points]
