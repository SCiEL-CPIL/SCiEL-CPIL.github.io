# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x, beta):
        """
        Smooth objective: maximize soft_min(d) / d_max
        where soft_min uses log-sum-exp trick.
        """
        points = x.reshape(n, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-10:
            return 0.0, np.zeros_like(x)
        
        # Soft minimum of distances
        # soft_min = -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability, subtract min first
        d_min_val = np.min(d)
        d_shifted = d - d_min_val
        log_sum_exp = np.log(np.sum(np.exp(-beta * d_shifted)))
        soft_min = d_min_val - log_sum_exp / beta
        
        score = soft_min / d_max
        return -score  # Negate for minimization

    def objective_grad(x, beta):
        points = x.reshape(n, 2)
        d = pdist(points)
        d_max = np.max(d)
        if d_max < 1e-10:
            return 0.0, np.zeros_like(x)
        
        d_min_val = np.min(d)
        d_shifted = d - d_min_val
        exp_neg_beta_d = np.exp(-beta * d_shifted)
        sum_exp = np.sum(exp_neg_beta_d)
        weights = exp_neg_beta_d / sum_exp
        
        soft_min = d_min_val - np.log(sum_exp) / beta
        score = soft_min / d_max
        
        # Gradient of soft_min w.r.t. points
        # d(soft_min)/d(points_k) = sum_i weights_i * d(d_i)/d(points_k)
        # For pair (i,j), d(d_ij)/d(points_k) = (points_k - points_j)/d_ij if k==i, 
        #                                     (points_k - points_i)/d_ij if k==j,
        #                                     0 otherwise
        
        grad_soft_min = np.zeros_like(points)
        idx = np.triu_indices(n, k=1)
        for k in range(len(idx[0])):
            i, j = idx[0][k], idx[1][k]
            diff = points[i] - points[j]
            dist = d[k]
            if dist < 1e-12:
                continue
            w = weights[k]
            grad_soft_min[i] += w * diff / dist
            grad_soft_min[j] -= w * diff / dist
        
        # Gradient of d_max: the pair with max distance
        max_idx = np.argmax(d)
        i_max, j_max = idx[0][max_idx], idx[1][max_idx]
        diff_max = points[i_max] - points[j_max]
        d_max_val = d[max_idx]
        if d_max_val < 1e-12:
            grad_d_max = np.zeros_like(points)
        else:
            grad_d_max = np.zeros_like(points)
            grad_d_max[i_max] += diff_max / d_max_val
            grad_d_max[j_max] -= diff_max / d_max_val
        
        # d(score)/d(x) = (d(soft_min)/dx * d_max - soft_min * d(d_max)/dx) / d_max^2
        grad_score = (grad_soft_min * d_max - soft_min * grad_d_max) / (d_max ** 2)
        
        return -score, -grad_score  # Negate for minimization

    def optimize_from(points_init, betas=None):
        x0 = points_init.flatten()
        
        if betas is None:
            betas = [5.0, 20.0, 100.0, 500.0, 2000.0]
        
        best_x = x0.copy()
        best_score = compute_score(points_init)
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            res = minimize(
                lambda x: objective_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 150, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            
            x0 = res.x
            points = x0.reshape(n, 2)
            score = compute_score(points)
            
            if score > best_score:
                best_score = score
                best_x = x0.copy()
        
        return best_x.reshape(n, 2), best_score

    def regular_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def random_points(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def grid_based(seed):
        np.random.seed(seed)
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append([i, j])
        pts = np.array(pts[:n], dtype=float)
        pts += np.random.randn(n, 2) * 0.5
        return pts

    # Generate diverse initializations
    initializations = []
    initializations.append(regular_polygon())
    
    for seed in range(30):
        initializations.append(random_points(seed))
    
    for seed in range(10):
        initializations.append(grid_based(seed))
    
    # Two concentric circles
    n1, n2 = 8, 8
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
    pts1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    pts2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    initializations.append(np.vstack([pts1, pts2]))
    
    # Four clusters of 4
    cluster_centers = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    pts = []
    for c in cluster_centers:
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        local = 0.3 * np.column_stack([np.cos(angles), np.sin(angles)])
        pts.append(local + c)
    initializations.append(np.vstack(pts))
    
    # 4x4 grid perturbed
    np.random.seed(999)
    grid_pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid_pts += np.random.randn(16, 2) * 0.3
    initializations.append(grid_pts)
    
    best_points = None
    best_score = -1.0
    
    for i, init in enumerate(initializations):
        if time.time() > deadline:
            break
        
        points, score = optimize_from(init)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        return regular_polygon()
    
    return best_points


# EVOLVE-BLOCK-END
