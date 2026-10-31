"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Smooth objective using log-sum-exp to approximate max/min
    # Objective: (max_dist / min_dist)^2 = exp(2 * (log_max - log_min))
    # We minimize log_ratio = log_max - log_min
    # log_max = log(sum(exp(d^2))) / (2 * scale_factor) -> approx max d
    # log_min = -log(sum(exp(-d^2))) / (2 * scale_factor) -> approx -min d
    
    # Better: minimize (max_d / min_d)^2 directly via smooth approx
    # Let f(x) = (M(x) / m(x))^2
    # M(x) approx max d, m(x) approx min d
    # Use soft min/max on squared distances to keep it differentiable and positive
    
    def smooth_objective(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        
        # Soft max of squared distances
        # To avoid overflow, we can normalize or use a stable log-sum-exp
        # However, for L-BFGS-B, we just need a smooth scalar.
        # Let's minimize the ratio of soft max to soft min of distances.
        
        # Soft max of d: max(d) ≈ (1/k) * log(sum(exp(k*d^2))) is tricky with squares.
        # Let's work with distances d = sqrt(d_sq)
        d = np.sqrt(d_sq)
        
        # Scale parameter for smoothness
        k = 10.0
        
        # Soft max: (1/k) * log(sum(exp(k * d)))
        # Soft min: -(1/k) * log(sum(exp(-k * d)))
        
        # To prevent overflow in exp, subtract max/min before exp
        d_max_val = np.max(d)
        d_min_val = np.min(d)
        
        # Soft max
        exp_term_max = np.exp(k * (d - d_max_val))
        soft_max = d_max_val + (1.0 / k) * np.log(np.sum(exp_term_max))
        
        # Soft min
        exp_term_min = np.exp(-k * (d - d_min_val))
        soft_min = d_min_val - (1.0 / k) * np.log(np.sum(exp_term_min))
        
        if soft_min < 1e-12:
            return 1e10
        
        ratio = soft_max / soft_min
        return ratio ** 2

    # Gradient of the smooth objective
    def smooth_gradient(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        d = np.sqrt(d_sq)
        
        k = 10.0
        d_max_val = np.max(d)
        d_min_val = np.min(d)
        
        exp_term_max = np.exp(k * (d - d_max_val))
        sum_exp_max = np.sum(exp_term_max)
        soft_max = d_max_val + (1.0 / k) * np.log(sum_exp_max)
        
        exp_term_min = np.exp(-k * (d - d_min_val))
        sum_exp_min = np.sum(exp_term_min)
        soft_min = d_min_val - (1.0 / k) * np.log(sum_exp_min)
        
        if soft_min < 1e-12:
            return np.zeros_like(x)
        
        ratio = soft_max / soft_min
        obj = ratio ** 2
        
        # Chain rule: d(obj)/d(x) = 2 * ratio * (d(soft_max)/d(x) / soft_min - soft_max * d(soft_min)/d(x) / soft_min^2)
        # = 2 * ratio / soft_min * (d(soft_max)/d(x) - (soft_max/soft_min) * d(soft_min)/d(x))
        # = 2 * ratio / soft_min * (d(soft_max)/d(x) - ratio * d(soft_min)/d(x))
        
        # Gradient of soft_max wrt d
        # soft_max = d_max + (1/k) log(sum(exp(k(d - d_max))))
        # d(soft_max)/d(d_i) = (1/k) * (k * exp(k(d_i - d_max)) / sum(exp(k(d - d_max))))
        # = exp_term_max[i] / sum_exp_max
        grad_soft_max_d = exp_term_max / sum_exp_max
        
        # Gradient of soft_min wrt d
        # soft_min = d_min - (1/k) log(sum(exp(-k(d - d_min))))
        # d(soft_min)/d(d_i) = - (1/k) * ( -k * exp(-k(d_i - d_min)) / sum(exp(-k(d - d_min))))
        # = exp_term_min[i] / sum_exp_min
        grad_soft_min_d = exp_term_min / sum_exp_min
        
        # Gradient of d wrt x
        # d_i = ||p_a - p_b||
        # d(d_i)/d(p_j) = (p_j - p_k) / d_i for the pair (j, k) corresponding to i
        
        # We need to map the gradient from d-space to x-space
        # grad_x = sum_i (grad_obj_d[i] * d(d_i)/dx)
        
        grad_x = np.zeros_like(x)
        
        # Efficiently compute using matrix operations
        # d_sq is (N, N), d is (N, N)
        # We only care about upper triangle i < j
        
        # Let's use a vectorized approach for the gradient accumulation
        # This is O(N^2) per point, which is fine for N=16.
        
        # grad_d is the gradient of the objective with respect to each distance d_i
        # grad_obj_d[i] = 2 * ratio / soft_min * (grad_soft_max_d[i] - ratio * grad_soft_min_d[i])
        
        grad_obj_d = 2.0 * ratio / soft_min * (grad_soft_max_d - ratio * grad_soft_min_d)
        
        # Now accumulate to points
        # For each pair (i, j) with i < j:
        # d_ij = ||p_i - p_j||
        # d(d_ij)/dp_i = (p_i - p_j) / d_ij
        # d(d_ij)/dp_j = (p_j - p_i) / d_ij
        
        # We can use the full matrix d and d_sq
        # Note: d[i,j] = d[j,i]
        
        # Vectorized gradient calculation
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
        # d_full = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
        
        # We have d_sq and d from get_sq_dists (upper triangle only)
        # Let's recompute full matrices for gradient ease, or map back.
        # Recomputing is cheap for N=16.
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_full = np.sqrt(np.sum(diff**2, axis=2))
        
        # Avoid division by zero
        d_full_safe = np.where(d_full > 1e-12, d_full, 1e-12)
        
        # We need grad_obj_d_full, which is symmetric
        # grad_obj_d from get_sq_dists is for upper triangle
        # Let's create a full matrix of gradients
        grad_obj_d_full = np.zeros((N, N))
        iu = np.triu_indices(N, 1)
        grad_obj_d_full[iu] = grad_obj_d
        grad_obj_d_full[iu[1], iu[0]] = grad_obj_d
        
        # grad_x[k] = sum_{m != k} grad_obj_d_full[k, m] * (p_k - p_m) / d_full[k, m]
        # = sum_{m != k} (grad_obj_d_full[k, m] / d_full_safe[k, m]) * (p_k - p_m)
        
        # coeff[k, m] = grad_obj_d_full[k, m] / d_full_safe[k, m]
        coeff = grad_obj_d_full / d_full_safe
        # Zero out diagonal
        np.fill_diagonal(coeff, 0.0)
        
        # grad_x = coeff @ (-pts) + (row_sums * pts)
        # grad_x[k] = sum_m coeff[k, m] * (p_k - p_m) = p_k * sum_m coeff[k, m] - sum_m coeff[k, m] p_m
        row_sums = np.sum(coeff, axis=1)
        grad_x = row_sums[:, np.newaxis] * pts - (coeff @ pts)
        
        return grad_x.ravel()

    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(5):
        pts = np.random.uniform(-1, 1, (N, 2))
        inits.append(normalize(pts).ravel())
        
    # 2. Grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    inits.append(normalize(base_grid).ravel())
    
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(normalize(circle_pts).ravel())
    
    # 4. Perturbed Grid
    for _ in range(5):
        noise = np.random.normal(0, 0.1, (N, 2))
        perturbed = base_grid + noise
        inits.append(normalize(perturbed).ravel())
        
    # 5. Two clusters
    for _ in range(5):
        c1 = np.random.uniform(-0.5, 0.5, (8, 2))
        c2 = np.random.uniform(0.5, 1.5, (8, 2))
        pts = np.vstack([c1, c2])
        inits.append(normalize(pts).ravel())

    best_x = None
    best_score = 1e10
    
    for x0 in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                smooth_objective,
                x0,
                jac=smooth_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate raw score at the result
            final_pts = res.x.reshape(N, 2)
            raw_score = score_raw(final_pts)
            
            if raw_score < best_score:
                best_score = raw_score
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        # Fallback to first init if all optimizations failed
        best_x = inits[0]
        best_score = score_raw(best_x.reshape(N, 2))
        
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
