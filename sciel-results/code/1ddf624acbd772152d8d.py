# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    def exact_score(points):
        """Calculate R = (d_min / d_max)^2"""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_max_score(x, beta=50.0):
        """
        Smooth approximation: maximize soft_min(dists) / soft_max(dists)
        Using log-sum-exp for stability.
        soft_min(d) = -log(sum(exp(-beta*d))) / beta
        soft_max(d) = log(sum(exp(beta*d))) / beta
        """
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        if np.any(dists <= 0) or not np.all(np.isfinite(dists)):
            return -1e10
        
        # Soft min: -log(sum(exp(-beta*d)))/beta
        neg_beta_d = -beta * dists
        max_neg = np.max(neg_beta_d)
        log_sum_exp_neg = max_neg + np.log(np.sum(np.exp(neg_beta_d - max_neg)))
        soft_min = -log_sum_exp_neg / beta
        
        # Soft max: log(sum(exp(beta*d)))/beta
        beta_d = beta * dists
        max_beta = np.max(beta_d)
        log_sum_exp_pos = max_beta + np.log(np.sum(np.exp(beta_d - max_beta)))
        soft_max = log_sum_exp_pos / beta
        
        if soft_max <= 0:
            return -1e10
        
        ratio = soft_min / soft_max
        if not np.isfinite(ratio):
            return -1e10
        return ratio ** 2

    def soft_objective(x, beta=50.0):
        """Objective for minimization (we minimize -soft_score)"""
        s = soft_min_max_score(x, beta)
        if not np.isfinite(s):
            return 1e10
        return -s

    def soft_objective_grad(x, beta=50.0):
        """
        Gradient of -soft_min_max_score.
        """
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        if np.any(dists <= 0) or not np.all(np.isfinite(dists)):
            return np.zeros_like(x)
        
        # Compute weights for soft min
        neg_beta_d = -beta * dists
        max_neg = np.max(neg_beta_d)
        exp_neg = np.exp(neg_beta_d - max_neg)
        sum_exp_neg = np.sum(exp_neg)
        w_neg = exp_neg / sum_exp_neg  # weights for soft min
        
        # Compute weights for soft max
        beta_d = beta * dists
        max_beta = np.max(beta_d)
        exp_pos = np.exp(beta_d - max_beta)
        sum_exp_pos = np.sum(exp_pos)
        w_pos = exp_pos / sum_exp_pos  # weights for soft max
        
        soft_min = -(max_neg + np.log(sum_exp_neg)) / beta
        soft_max = (max_beta + np.log(sum_exp_pos)) / beta
        
        if soft_max <= 0:
            return np.zeros_like(x)
        
        ratio = soft_min / soft_max
        if not np.isfinite(ratio) or ratio <= 0:
            return np.zeros_like(x)
        
        # Build gradient by accumulating contributions
        grad = np.zeros_like(x)
        
        # Map from pdist index to (i, k) pairs
        idx = 0
        for i in range(n - 1):
            for k in range(i + 1, n):
                d_j = dists[idx]
                if d_j < 1e-10:
                    idx += 1
                    continue
                diff = pts[i] - pts[k]
                # d(d_j)/dx_i = diff / d_j, d(d_j)/dx_k = -diff / d_j
                
                w_n = w_neg[idx]
                w_p = w_pos[idx]
                
                # Contribution to d(soft_min)/dx: (1/beta) * w_n * diff/d_j for x_i, -w_n * diff/d_j for x_k
                # Contribution to d(soft_max)/dx: (1/beta) * w_p * diff/d_j for x_i, -w_p * diff/d_j for x_k
                
                coeff_min_i = (1.0 / beta) * w_n / d_j
                coeff_min_k = -(1.0 / beta) * w_n / d_j
                coeff_max_i = (1.0 / beta) * w_p / d_j
                coeff_max_k = -(1.0 / beta) * w_p / d_j
                
                # d(score)/dx_i = 2*ratio/soft_max * (coeff_min_i * diff - ratio * coeff_max_i * diff)
                # = 2*ratio/soft_max * diff * (coeff_min_i - ratio * coeff_max_i)
                
                factor_i = 2.0 * ratio / soft_max * (coeff_min_i - ratio * coeff_max_i)
                factor_k = 2.0 * ratio / soft_max * (coeff_min_k - ratio * coeff_max_k)
                
                grad[2*i:2*i+2] += factor_i * diff
                grad[2*k:2*k+2] += factor_k * diff
                
                idx += 1
        
        # We're minimizing -score, so gradient of objective = -gradient of score
        return -grad

    # Generate diverse initial configurations
    initial_guesses = []
    
    # 1. 4x4 Grid
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    d_max_grid = np.max(pdist(grid_pts))
    if d_max_grid > 0:
        grid_pts = grid_pts / d_max_grid
    initial_guesses.append(grid_pts.flatten())

    # 2. Hexagonal-like arrangement
    # Use a hexagonal packing pattern for 16 points
    hex_pts = []
    # Center point
    hex_pts.append([0.0, 0.0])
    # Ring 1: 6 points at radius 1
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        hex_pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 9 points at radius ~2 (approximate hexagonal)
    for angle in np.linspace(0, 2*np.pi, 9, endpoint=False):
        r = 2.0
        hex_pts.append([r * np.cos(angle), r * np.sin(angle)])
    hex_pts = np.array(hex_pts)
    hex_pts = hex_pts - np.mean(hex_pts, axis=0)
    d_max_hex = np.max(pdist(hex_pts))
    if d_max_hex > 0:
        hex_pts = hex_pts / d_max_hex
    initial_guesses.append(hex_pts.flatten())

    # 3. Two concentric rings: 6 + 10
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r1 = 0.6
    ring1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    angles2 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r2 = 1.0
    ring2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    pts_rings = np.vstack((ring1, ring2))
    pts_rings = pts_rings - np.mean(pts_rings, axis=0)
    d_max_rings = np.max(pdist(pts_rings))
    if d_max_rings > 0:
        pts_rings = pts_rings / d_max_rings
    initial_guesses.append(pts_rings.flatten())

    # 4. Center + 5 + 10
    p0 = np.array([0.0, 0.0])
    angles1_5 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r1_5 = 0.5
    ring1_5 = np.column_stack((r1_5 * np.cos(angles1_5), r1_5 * np.sin(angles1_5)))
    angles2_10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    r2_10 = 1.0
    ring2_10 = np.column_stack((r2_10 * np.cos(angles2_10), r2_10 * np.sin(angles2_10)))
    pts_1_5_10 = np.vstack((p0.reshape(1,2), ring1_5, ring2_10))
    pts_1_5_10 = pts_1_5_10 - np.mean(pts_1_5_10, axis=0)
    d_max_1510 = np.max(pdist(pts_1_5_10))
    if d_max_1510 > 0:
        pts_1_5_10 = pts_1_5_10 / d_max_1510
    initial_guesses.append(pts_1_5_10.flatten())

    # 5. Random initializations
    np.random.seed(12345)
    for i in range(8):
        rand_pts = np.random.randn(n, 2)
        rand_pts = rand_pts - np.mean(rand_pts, axis=0)
        d_max_r = np.max(pdist(rand_pts))
        if d_max_r > 0:
            rand_pts = rand_pts / d_max_r
        initial_guesses.append(rand_pts.flatten())
    
    # 6. Hexagonal cluster of 19 points with 3 removed (Inspiration 1 strategy)
    # Generate a hexagonal lattice of 19 points (center + 2 rings)
    # Center: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Total 19. We need 16, so remove 3 corner points to make it more compact?
    # Or just take the first 16?
    # Let's try to construct a specific 16-point subset of a hex lattice.
    # A 4x4 grid is square. A hex cluster is better.
    # Let's generate a hex lattice with spacing 1.
    # Points (i, j) -> (i + 0.5*j, sqrt(3)/2 * j)
    # We want a compact shape.
    # Let's try a "hexagon" of side length 3? No, that's 19 points.
    # 1, 6, 12, 18...
    # 19 points is 1+6+12.
    # Removing 3 points from the outer ring (radius 2) might help.
    # Let's generate the 19 points.
    h = np.sqrt(3) / 2
    hex19 = np.array([[0.0, 0.0]])
    # Ring 1 (radius 1)
    for k in range(6):
        angle = np.pi / 3 * k
        hex19 = np.vstack([hex19, [[np.cos(angle), np.sin(angle)]]])
    # Ring 2 (radius 2)
    # Points on hexagon vertices and midpoints?
    # Actually, standard hex lattice coordinates are easier.
    # Let's just use the coordinates directly.
    # Center (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 pts
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-2,-1), (-1,-2), (0,-2), (1,-2), (2,-1) -> 12 pts
    # Total 19.
    # Let's construct these.
    hex19 = []
    # Center
    hex19.append([0, 0])
    # Ring 1
    hex19.append([1, 0])
    hex19.append([0, 1])
    hex19.append([-1, 1])
    hex19.append([-1, 0])
    hex19.append([0, -1])
    hex19.append([1, -1])
    # Ring 2
    hex19.append([2, 0])
    hex19.append([1, 1])
    hex19.append([0, 2])
    hex19.append([-1, 2])
    hex19.append([-2, 2])
    hex19.append([-2, 1])
    hex19.append([-2, 0])
    hex19.append([-2, -1])
    hex19.append([-1, -2])
    hex19.append([0, -2])
    hex19.append([1, -2])
    hex19.append([2, -1])
    
    hex19 = np.array(hex19, dtype=float)
    # Convert axial to cartesian?
    # The coordinates above are axial (q, r).
    # Cartesian: x = q + 0.5*r, y = sqrt(3)/2 * r
    cart = np.zeros((19, 2))
    for i in range(19):
        q, r = hex19[i]
        cart[i, 0] = q + 0.5 * r
        cart[i, 1] = h * r
    
    # We have 19 points. We need 16.
    # Remove 3 points from the outer ring to reduce d_max while keeping d_min high.
    # The outer ring points are indices 7 to 18.
    # Let's try removing 3 "corner" points that are far apart.
    # Indices: 7(2,0), 10(0,2), 14(-2,0), 17(0,-2) are the "axis" points.
    # Let's remove 7, 10, 14? Or maybe just take the first 16?
    # Taking the first 16 removes the last 3: 16, 17, 18 which are (-1,-2), (0,-2), (1,-2).
    # This creates an asymmetry.
    # Let's try a few subsets.
    # Subset 1: Remove 3 consecutive from the end.
    sub1 = cart[:16]
    sub1 = sub1 - np.mean(sub1, axis=0)
    d_max_s1 = np.max(pdist(sub1))
    if d_max_s1 > 0:
        sub1 = sub1 / d_max_s1
    initial_guesses.append(sub1.flatten())
    
    # Subset 2: Remove 3 spread out points (e.g., 7, 11, 15)
    # 7: (2,0), 11: (-1,2), 15: (-2,-1)
    mask2 = np.ones(19, dtype=bool)
    mask2[[7, 11, 15]] = False
    sub2 = cart[mask2]
    sub2 = sub2 - np.mean(sub2, axis=0)
    d_max_s2 = np.max(pdist(sub2))
    if d_max_s2 > 0:
        sub2 = sub2 / d_max_s2
    initial_guesses.append(sub2.flatten())
    
    # Subset 3: Remove 4 "axis" points and add 1 center? No, center is already there.
    # Just try removing 7, 10, 14, 17 (the 4 axis points of the hexagon) -> 15 points. Add one more?
    # We need 16.
    # Let's just use the 19 points and optimize? No, fixed size.
    # Let's try removing 7, 10, 14.
    mask3 = np.ones(19, dtype=bool)
    mask3[[7, 10, 14]] = False
    sub3 = cart[mask3]
    sub3 = sub3 - np.mean(sub3, axis=0)
    d_max_s3 = np.max(pdist(sub3))
    if d_max_s3 > 0:
        sub3 = sub3 / d_max_s3
    initial_guesses.append(sub3.flatten())

    best_score = -np.inf
    best_points = initial_guesses[0].reshape(-1, 2)

    # Phase 1: L-BFGS-B with soft objective from multiple starts
    for x0 in initial_guesses:
        if time.time() > deadline - 120:
            break
            
        try:
            # Use L-BFGS-B with soft objective and gradient
            res = minimize(soft_objective, x0, method='L-BFGS-B', 
                          jac=lambda x: soft_objective_grad(x, beta=50.0),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            
            if res.fun < best_score:
                pts = res.x.reshape(-1, 2)
                s = exact_score(pts)
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = pts
        except Exception:
            pts = x0.reshape(-1, 2)
            s = exact_score(pts)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_points = pts

    # Phase 2: Nelder-Mead on exact objective from best point and nearby perturbations
    if time.time() < deadline - 100:
        # Start from best found
        try:
            res = minimize(lambda x: -exact_score(x.reshape(-1, 2)), best_points.flatten(), 
                          method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            pts = res.x.reshape(-1, 2)
            s = exact_score(pts)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass

    # Phase 3: More L-BFGS-B with higher beta (sharper soft-min) from best point
    if time.time() < deadline - 80:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 60:
                break
            try:
                res = minimize(lambda x: soft_objective(x, beta), best_points.flatten(), 
                              method='L-BFGS-B',
                              jac=lambda x: soft_objective_grad(x, beta),
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                pts = res.x.reshape(-1, 2)
                s = exact_score(pts)
                if np.isfinite(s) and s > best_score:
                    best_score = s
                    best_points = pts
            except Exception:
                pass

    # Phase 4: Final Nelder-Mead polish on exact objective
    if time.time() < deadline - 40:
        try:
            res = minimize(lambda x: -exact_score(x.reshape(-1, 2)), best_points.flatten(), 
                          method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            pts = res.x.reshape(-1, 2)
            s = exact_score(pts)
            if np.isfinite(s) and s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass

    # Final sanity check
    if not np.all(np.isfinite(best_points)):
        best_points = initial_guesses[0].reshape(-1, 2)

    return best_points

# EVOLVE-BLOCK-END
