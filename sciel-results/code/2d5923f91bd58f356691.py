import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Reframe the problem as a geometric packing problem using a smooth surrogate objective.
    2. Use L-BFGS-B with analytic gradients for fast, robust convergence.
    3. Multi-start from diverse high-quality initial configurations (hexagonal lattice, 
       circle packing, random restarts) to avoid local minima.
    4. Final polish with Nelder-Mead on the raw objective for precision.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(params):
        """
        Smooth surrogate objective: minimize -log(sum(exp(-d_i^2 / T))) + log(sum(exp(d_i^2 / T)))
        This approximates -(d_min/d_max)^2 smoothly.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_sq = dists ** 2
        
        # Temperature for soft-min/soft-max
        T = 0.01
        
        # Soft max: approximates max(d_sq)
        # Use log-sum-exp for numerical stability
        max_d_sq = np.max(d_sq)
        soft_max = max_d_sq + T * np.log(np.sum(np.exp((d_sq - max_d_sq) / T)))
        
        # Soft min: approximates min(d_sq)
        min_d_sq = np.min(d_sq)
        soft_min = min_d_sq - T * np.log(np.sum(np.exp(-(d_sq - min_d_sq) / T)))
        
        # Objective: minimize -(soft_min / soft_max)^2
        ratio_sq = soft_min / soft_max
        return -(ratio_sq ** 2)

    def smooth_gradient(params):
        """
        Analytic gradient of the smooth objective.
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_sq = dists ** 2
        
        T = 0.01
        
        # Soft max: approximates max(d_sq)
        max_d_sq = np.max(d_sq)
        exp_max = np.exp((d_sq - max_d_sq) / T)
        sum_exp_max = np.sum(exp_max)
        soft_max = max_d_sq + T * np.log(sum_exp_max)
        
        # Soft min: approximates min(d_sq)
        min_d_sq = np.min(d_sq)
        exp_min = np.exp(-(d_sq - min_d_sq) / T)
        sum_exp_min = np.sum(exp_min)
        soft_min = min_d_sq - T * np.log(sum_exp_min)
        
        # Gradient of soft_max w.r.t. d_sq
        grad_soft_max = exp_max / sum_exp_max
        
        # Gradient of soft_min w.r.t. d_sq
        grad_soft_min = exp_min / sum_exp_min
        
        # ratio_sq = soft_min / soft_max
        # d(ratio_sq)/d(soft_min) = 1/soft_max
        # d(ratio_sq)/d(soft_max) = -soft_min/soft_max^2 = -ratio_sq/soft_max
        
        ratio_sq = soft_min / soft_max
        d_ratio_d_soft_min = 1.0 / soft_max
        d_ratio_d_soft_max = -ratio_sq / soft_max
        
        # d(objective)/d(soft_min) = -2 * ratio_sq * d_ratio_d_soft_min
        # d(objective)/d(soft_max) = -2 * ratio_sq * d_ratio_d_soft_max
        
        d_obj_d_soft_min = -2 * ratio_sq * d_ratio_d_soft_min
        d_obj_d_soft_max = -2 * ratio_sq * d_ratio_d_soft_max
        
        # d(soft_min)/d(d_sq) = grad_soft_min
        # d(soft_max)/d(d_sq) = grad_soft_max
        
        d_obj_d_dsq = d_obj_d_soft_min * grad_soft_min + d_obj_d_soft_max * grad_soft_max
        
        # Now compute gradient w.r.t. points
        # d(d_sq_ij)/d(points_k) for k != i,j: 0
        # d(d_sq_ij)/d(points_i) = 2 * (points_i - points_j)
        # d(d_sq_ij)/d(points_j) = 2 * (points_j - points_i)
        
        grad_points = np.zeros((n, 2))
        
        # Map from pdist index to (i, j)
        # pdist uses upper triangular matrix
        idx = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                d_ij_sq = d_sq[idx]
                w = d_obj_d_dsq[idx]
                
                # d(d_ij_sq)/d(points_i) = 2 * (points_i - points_j)
                grad_points[i] += 2 * w * (points[i] - points[j])
                # d(d_ij_sq)/d(points_j) = 2 * (points_j - points_i)
                grad_points[j] += 2 * w * (points[j] - points[i])
                
                idx += 1
        
        return grad_points.ravel()

    def raw_objective(params):
        """Objective function for optimization: minimize -R"""
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return -(ratio ** 2)

    # Step 1: Generate diverse initial configurations
    
    def generate_hex_lattice_points(max_radius=5, spacing=1.0):
        points = []
        for q in range(-max_radius, max_radius + 1):
            for r in range(-max_radius, max_radius + 1):
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    def generate_circle_packing():
        """Generate points on concentric circles with optimal angular spacing"""
        points = []
        # Center point
        points.append([0.0, 0.0])
        
        # Ring 1: 6 points at radius 1
        for k in range(6):
            angle = 2 * np.pi * k / 6
            points.append([np.cos(angle), np.sin(angle)])
        
        # Ring 2: 9 points at radius 2 (approximate)
        for k in range(9):
            angle = 2 * np.pi * k / 9 + np.pi / 9  # Offset for better packing
            points.append([2 * np.cos(angle), 2 * np.sin(angle)])
        
        return np.array(points[:n])

    def generate_random_points(seed):
        """Generate random points in a unit disk"""
        rng = np.random.RandomState(seed)
        r = rng.uniform(0, 1, n) ** 0.5
        theta = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    # Generate initial configurations
    hex_points = generate_hex_lattice_points(max_radius=5)
    hex_points -= np.mean(hex_points, axis=0)
    dists_to_center = np.linalg.norm(hex_points, axis=1)
    sorted_indices = np.argsort(dists_to_center)
    initial_hex = hex_points[sorted_indices[:n]].copy()
    
    # Normalize
    d_max_init = np.max(pdist(initial_hex))
    if d_max_init > 1e-10:
        initial_hex = initial_hex / d_max_init

    initial_circle = generate_circle_packing()
    d_max_init = np.max(pdist(initial_circle))
    if d_max_init > 1e-10:
        initial_circle = initial_circle / d_max_init

    best_points = initial_hex.copy()
    best_score = evaluate(best_points)

    # Step 2: Multi-start L-BFGS-B optimization
    initial_configs = [initial_hex, initial_circle]
    
    # Add some random restarts
    for seed in [42, 123, 456]:
        initial_configs.append(generate_random_points(seed))

    for init_idx, init_config in enumerate(initial_configs):
        if time_exceeded():
            break
        
        try:
            res = minimize(
                smooth_objective,
                init_config.ravel(),
                jac=smooth_gradient,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                },
                callback=lambda xk: time_exceeded()
            )
            
            if not time_exceeded():
                candidate = res.x.reshape((n, 2))
                score = evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            pass

    # Step 3: Final polish with Nelder-Mead on the raw objective
    if not time_exceeded():
        try:
            res = minimize(
                raw_objective,
                best_points.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                },
                callback=lambda xk: time_exceeded()
            )
            if not time_exceeded():
                candidate = res.x.reshape((n, 2))
                score = evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            pass

    # Step 4: If time remains, try one more polish from a slightly perturbed best point
    if not time_exceeded():
        try:
            np.random.seed(42)
            perturbed = best_points + np.random.normal(0, 1e-5, (n, 2))
            res = minimize(
                raw_objective,
                perturbed.ravel(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                },
                callback=lambda xk: time_exceeded()
            )
            if not time_exceeded():
                candidate = res.x.reshape((n, 2))
                score = evaluate(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
        except Exception:
            pass

    return best_points
