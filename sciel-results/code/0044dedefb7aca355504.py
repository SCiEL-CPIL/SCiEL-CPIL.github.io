import numpy as np
from scipy.optimize import minimize
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    n = 8

    def compute_score(pts):
        """
        Compute Delta/n^n where Delta is the product of all ordered pair distances
        after scaling to diameter 2.
        """
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 0.0
        if len(pts) != n:
            return 0.0
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 0.0
        
        # Compute diameter
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 0.0
        
        # Scale to diameter 2
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        # Compute log of product of all ordered pair distances
        # Ordered pairs means (i,j) and (j,i) both count, so it's 2 * sum over i<j of log(d_ij)
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 0.0
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        
        # Delta = exp(log_prod_ordered), score = Delta / n^n
        # log(score) = log_prod_ordered - n*log(n)
        log_score = log_prod_ordered - n * math.log(n)
        
        return math.exp(log_score)

    def negative_log_score(points):
        """Objective for minimization: -log(score)"""
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return 1e10
        
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return 1e10
        
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 1e10
        
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 1e10
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        log_score = log_prod_ordered - n * math.log(n)
        
        return -log_score

    # 1. Baseline: Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = compute_score(reg_pts)
    best_pts = reg_pts.copy()

    # 2. Targeted Search: Two Concentric Rotated Squares
    # Square 1 (radius r1): angles 0, pi/2, pi, 3pi/2
    # Square 2 (radius r2): angles theta, theta+pi/2, theta+pi, theta+3pi/2
    
    def two_square_points(r1, r2, theta):
        pts = np.zeros((n, 2))
        # Square 1
        for k in range(4):
            a = k * math.pi / 2
            pts[k] = [r1 * math.cos(a), r1 * math.sin(a)]
        # Square 2
        for k in range(4):
            a = theta + k * math.pi / 2
            pts[4+k] = [r2 * math.cos(a), r2 * math.sin(a)]
        return pts

    def two_square_log_score(r1, r2, theta):
        """
        Compute log(Delta/n^n) for two concentric squares with relative rotation theta.
        """
        if r1 <= 1e-10 or r2 <= 1e-10:
            return -1e10
        
        # Build points
        pts = two_square_points(r1, r2, theta)
        
        # Compute all pairwise distances
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-15:
                    return -1e10
                dists.append(d)
        
        # Log product of unordered distances
        log_prod_unordered = sum(math.log(d) for d in dists)
        
        # Diameter
        D = max(dists)
        
        # Delta = (product of ordered distances) = (product of unordered)^2
        # log(Delta) = 2 * log_prod_unordered
        # log(score) = log(Delta) - log(D^28) - 8*log(8)
        #            = 2*log_prod_unordered - 28*log(D) - 8*log(8)
        
        log_score = 2 * log_prod_unordered - 28 * math.log(D) - 8 * math.log(8)
        return log_score

    def neg_two_square_log_score(params):
        r1, r2, theta = params
        if r1 <= 0 or r2 <= 0:
            return 1e10
        # Normalize theta to [0, pi/2) for uniqueness
        theta_mod = theta % (math.pi / 2)
        return -two_square_log_score(r1, r2, theta_mod)

    # Optimize r1, r2, theta
    # Regular octagon: r1=r2=1, theta=pi/4
    best_r1 = 1.0
    best_r2 = 1.0
    best_theta = math.pi / 4
    best_sym_score = two_square_log_score(1.0, 1.0, math.pi / 4)
    
    # 3D grid search
    r_range = np.linspace(0.3, 1.5, 20)
    theta_range = np.linspace(0, math.pi/2, 20)
    
    for r1 in r_range:
        if time.time() > deadline - 15:
            break
        for r2 in r_range:
            for theta in theta_range:
                s = two_square_log_score(r1, r2, theta)
                if s > best_sym_score:
                    best_sym_score = s
                    best_r1 = r1
                    best_r2 = r2
                    best_theta = theta

    # Local refinement in 3D
    if time.time() < deadline - 10:
        try:
            res = minimize(neg_two_square_log_score, [best_r1, best_r2, best_theta], 
                           method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
            if res.success or res.fun < 1e9:
                r1_opt, r2_opt, theta_opt = res.x
                if r1_opt > 0 and r2_opt > 0:
                    theta_opt_mod = theta_opt % (math.pi / 2)
                    s_opt = two_square_log_score(r1_opt, r2_opt, theta_opt_mod)
                    if s_opt > best_sym_score:
                        best_sym_score = s_opt
                        best_r1 = r1_opt
                        best_r2 = r2_opt
                        best_theta = theta_opt_mod
        except Exception:
            pass

    # Also try from regular octagon
    try:
        res_reg = minimize(neg_two_square_log_score, [1.0, 1.0, math.pi/4], 
                           method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
        if res_reg.success or res_reg.fun < 1e9:
            r1_reg, r2_reg, theta_reg = res_reg.x
            if r1_reg > 0 and r2_reg > 0:
                theta_reg_mod = theta_reg % (math.pi / 2)
                s_reg = two_square_log_score(r1_reg, r2_reg, theta_reg_mod)
                if s_reg > best_sym_score:
                    best_sym_score = s_reg
                    best_r1 = r1_reg
                    best_r2 = r2_reg
                    best_theta = theta_reg_mod
    except Exception:
        pass

    # Generate candidate from optimized model
    cand_pts = two_square_points(best_r1, best_r2, best_theta)
    s_cand = compute_score(cand_pts)
    if s_cand > best_score:
        best_score = s_cand
        best_pts = cand_pts.copy()

    # 3. Full Space Refinement from Best Candidate
    # Use the best point found to seed a full 16D optimization.
    # This allows the points to move off the symmetric model if that yields a higher score.
    
    if time.time() < deadline - 15:
        x0 = best_pts.flatten()
        
        # Multiple restarts with varying noise
        for noise_level in [0.0, 0.02, 0.1]:
            if time.time() > deadline - 5:
                break
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                x_init = x0 + np.random.normal(0, noise_level, size=16)
                try:
                    res = minimize(negative_log_score, x_init, method='L-BFGS-B',
                                   bounds=[(-10.0, 10.0)] * 16,
                                   options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
                    
                    if np.all(np.isfinite(res.x)):
                        pts = res.x.reshape(n, 2)
                        s = compute_score(pts)
                        if s > best_score:
                            best_score = s
                            best_pts = pts.copy()
                except Exception:
                    pass

    # 4. Final Validation and Formatting
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        best_score = compute_score(reg_pts)
        
    # Check distinctness
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-6:
        best_pts = reg_pts
        best_score = compute_score(reg_pts)

    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
