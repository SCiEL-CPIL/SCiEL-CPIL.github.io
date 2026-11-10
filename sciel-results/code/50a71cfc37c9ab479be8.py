# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def compute_ratio(params):
        """Compute the ratio d_min/d_max from point parameters."""
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
        return d_min / d_max

    def exact_objective(params):
        """Exact objective: minimize -(d_min/d_max)."""
        return -compute_ratio(params)

    def soft_objective(params, beta=50.0):
        """
        Smooth surrogate using soft-min and soft-max.
        soft_min(d) ≈ (1/beta) * log(sum(exp(-beta*d)))
        soft_max(d) ≈ -(1/beta) * log(sum(exp(-beta*d)))
        We want to maximize soft_min/soft_max, so minimize -soft_min/soft_max.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        
        # Soft min: approximate minimum using log-sum-exp trick
        # soft_min = -(1/beta) * log(sum(exp(-beta * dists)))
        max_d = np.max(dists)
        min_d = np.min(dists)
        
        if max_d < 1e-10:
            return 0.0
            
        # Use stable computation
        neg_beta_dists = -beta * dists
        neg_beta_max = np.max(neg_beta_dists)
        soft_min = -(1.0/beta) * (neg_beta_max + np.log(np.sum(np.exp(neg_beta_dists - neg_beta_max))))
        
        pos_beta_dists = beta * dists
        pos_beta_max = np.max(pos_beta_dists)
        soft_max = (1.0/beta) * (pos_beta_max + np.log(np.sum(np.exp(pos_beta_dists - pos_beta_max))))
        
        if soft_max < 1e-10:
            return 0.0
            
        ratio = soft_min / soft_max
        return -ratio

    def soft_objective_deriv(params, beta=50.0):
        """
        Gradient of the soft objective.
        """
        pts = params.reshape((n, 2))
        dists = pdist(pts, 'euclidean')
        
        max_d = np.max(dists)
        if max_d < 1e-10:
            return np.zeros_like(params)
            
        # Compute soft_min and soft_max with their derivatives
        # For soft_min: f_min = -(1/beta) * log(sum(exp(-beta*d_i)))
        # df_min/dx = (1/beta) * sum(exp(-beta*d_i) * (-beta * dd_i/dx)) / sum(exp(-beta*d_i))
        #            = -sum(exp(-beta*d_i) * dd_i/dx) / sum(exp(-beta*d_i))
        
        # For soft_max: f_max = (1/beta) * log(sum(exp(beta*d_i)))
        # df_max/dx = sum(exp(beta*d_i) * dd_i/dx) / sum(exp(beta*d_i))
        
        # We want to maximize f_min/f_max, so minimize -(f_min/f_max)
        # d(-f_min/f_max)/dx = -(f_min' * f_max - f_min * f_max') / f_max^2
        #                    = (f_min * f_max' - f_min' * f_max) / f_max^2
        
        # Compute weights
        neg_beta_dists = -beta * dists
        neg_beta_max = np.max(neg_beta_dists)
        exp_neg = np.exp(neg_beta_dists - neg_beta_max)
        sum_exp_neg = np.sum(exp_neg)
        
        pos_beta_dists = beta * dists
        pos_beta_max = np.max(pos_beta_dists)
        exp_pos = np.exp(pos_beta_dists - pos_beta_max)
        sum_exp_pos = np.sum(exp_pos)
        
        w_min = exp_neg / sum_exp_neg  # weights for soft_min
        w_max = exp_pos / sum_exp_pos  # weights for soft_max
        
        f_min = -(1.0/beta) * (neg_beta_max + np.log(sum_exp_neg))
        f_max = (1.0/beta) * (pos_beta_max + np.log(sum_exp_pos))
        
        if f_max < 1e-10:
            return np.zeros_like(params)
            
        # Need dd_i/dx for all pairs
        # dists[i] corresponds to pair (i,j) where i < j
        # We need to build the gradient with respect to each point
        
        grad = np.zeros_like(pts)
        
        # Map from pair index to (i, j)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                d = dists[idx]
                if d < 1e-12:
                    idx += 1
                    continue
                    
                # Unit vector from point i to point j
                vec = pts[j] - pts[i]
                unit_vec = vec / d
                
                # Contribution to d(dist_i/dx) for point i: -unit_vec
                # Contribution to d(dist_i/dx) for point j: +unit_vec
                
                # d(f_min)/dx contribution: -w_min[idx] * d(dist_i)/dx
                # d(f_max)/dx contribution: +w_max[idx] * d(dist_i)/dx
                
                dfmin_dx_i = -w_min[idx] * (-unit_vec)  # = w_min[idx] * unit_vec
                dfmin_dx_j = -w_min[idx] * (unit_vec)   # = -w_min[idx] * unit_vec
                
                dfmax_dx_i = w_max[idx] * (-unit_vec)   # = -w_max[idx] * unit_vec
                dfmax_dx_j = w_max[idx] * (unit_vec)    # = w_max[idx] * unit_vec
                
                # d(-f_min/f_max)/dx = (f_min * dfmax_dx - dfmin_dx * f_max) / f_max^2
                factor = (f_min * dfmax_dx_i - dfmin_dx_i * f_max) / (f_max * f_max)
                grad[i] += factor
                
                factor_j = (f_min * dfmax_dx_j - dfmin_dx_j * f_max) / (f_max * f_max)
                grad[j] += factor_j
                
                idx += 1
        
        return grad.flatten()

    # Generate diverse initial configurations
    def make_grid_4x4():
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))

    def make_hexagonal_16():
        """Create a hexagonal-like arrangement of 16 points."""
        pts = []
        # 4 rows, staggered
        for row in range(4):
            y = row * np.sqrt(3) / 2
            if row % 2 == 0:
                for col in range(4):
                    x = col * 1.0
                    pts.append([x, y])
            else:
                for col in range(4):
                    x = col * 1.0 + 0.5
                    pts.append([x, y])
        return np.array(pts)

    def make_circle_16():
        """16 points on a circle."""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def make_random(seed=42):
        rng = np.random.RandomState(seed)
        return rng.rand(n, 2) * 10 - 5

    initial_configs = [
        make_grid_4x4(),
        make_hexagonal_16(),
        make_circle_16(),
        make_random(42),
        make_random(123),
    ]

    best_points = None
    best_score = -np.inf

    # Phase 1: Use soft objective with gradient-based optimization
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
            
        try:
            # First, optimize with soft objective using L-BFGS-B
            res_soft = minimize(
                soft_objective,
                init_pts.flatten(),
                jac=soft_objective_deriv,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if time.time() > deadline - 20:
                break
                
            soft_candidate = res_soft.x.reshape((n, 2))
            
            # Phase 2: Polish with exact objective
            res_exact = minimize(
                exact_objective,
                soft_candidate.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() > deadline - 10:
                break
                
            candidate = res_exact.x.reshape((n, 2))
            score = compute_ratio(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
                
        except Exception as e:
            continue

    # If we didn't find anything, fall back to grid
    if best_points is None:
        best_points = make_grid_4x4()
        best_score = compute_ratio(best_points)

    # Additional refinement: try Powell method on the best solution if time permits
    if time.time() < deadline - 60:
        try:
            res_powell = minimize(
                exact_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12}
            )
            candidate = res_powell.x.reshape((n, 2))
            score = compute_ratio(candidate)
            if score > best_score:
                best_points = candidate
                best_score = score
        except Exception:
            pass

    # Try a few more random restarts with the two-stage approach if time allows
    if time.time() < deadline - 45:
        for seed in [777, 888, 999]:
            if time.time() > deadline - 30:
                break
            try:
                init_pts = make_random(seed)
                res_soft = minimize(
                    soft_objective,
                    init_pts.flatten(),
                    jac=soft_objective_deriv,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if time.time() > deadline - 15:
                    break
                soft_candidate = res_soft.x.reshape((n, 2))
                res_exact = minimize(
                    exact_objective,
                    soft_candidate.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                candidate = res_exact.x.reshape((n, 2))
                score = compute_ratio(candidate)
                if score > best_score:
                    best_points = candidate
                    best_score = score
            except Exception:
                continue

    return best_points

# EVOLVE-BLOCK-END
