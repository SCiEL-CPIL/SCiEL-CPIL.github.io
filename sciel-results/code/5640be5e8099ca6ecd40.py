# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def objective(x):
        """
        x is a flattened array of shape (n*d,).
        Minimizes the sum of inverse squared distances to push points apart.
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        # Avoid division by zero, though unlikely in optimization
        dists = np.maximum(dists, 1e-10)
        # Sum of inverse squared distances
        return np.sum(1.0 / (dists ** 2))

    def gradient(x):
        """
        Gradient of the objective function.
        """
        pts = x.reshape(n, d)
        # Compute pairwise differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Avoid division by zero
        mask = dists > 1e-10
        inv_dists_sq = np.zeros_like(dists_sq)
        inv_dists_sq[mask] = 1.0 / (dists_sq[mask] ** 2) # d/dx (1/d^2) = -2/d^4 * dx/dd = -2/d^4 * (diff/d) = -2 diff / d^5
        # Wait, let's derive carefully.
        # f = sum_{i<j} d_{ij}^{-2}
        # df/dx_k = sum_{i<j, k in {i,j}} -2 d_{ij}^{-4} (dd_{ij}/dx_k)
        # dd_{ij}/dx_k = (x_i - x_j) / d_{ij} if k=i, (x_j - x_i)/d_{ij} if k=j, 0 otherwise.
        
        # Vectorized gradient computation
        # grad_{ij} = -2 * d_{ij}^{-4} * (x_i - x_j) / d_{ij} = -2 * (x_i - x_j) / d_{ij}^5
        # Let's compute the force on each point i from all j.
        # F_i = sum_{j != i} -2 * (x_i - x_j) / d_{ij}^5
        # Note: pdist gives upper triangle. We need full matrix for easy summing.
        
        # Re-compute with full matrix for clarity and vectorization
        diff_full = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # (n, n, d)
        dists_full = np.sqrt(np.sum(diff_full ** 2, axis=2)) # (n, n)
        np.fill_diagonal(dists_full, 1.0) # Avoid div by 0 on diagonal, will be masked out
        
        # Gradient contribution from pair (i, j) to point i is -2 * (x_i - x_j) / d_{ij}^5
        # Gradient contribution from pair (i, j) to point j is -2 * (x_j - x_i) / d_{ij}^5 = 2 * (x_i - x_j) / d_{ij}^5
        
        # Let's sum forces.
        # Force on i from j: F_{ij} = -2 * (x_i - x_j) / d_{ij}^5
        # Total Force on i: F_i = sum_{j != i} F_{ij}
        
        # Create a matrix of coefficients c_{ij} = -2 / d_{ij}^5 for i != j
        coeffs = np.zeros_like(dists_full)
        mask = ~np.eye(n, dtype=bool)
        coeffs[mask] = -2.0 / (dists_full[mask] ** 5)
        
        # F_i = sum_j c_{ij} * (x_i - x_j)
        # F_i = (sum_j c_{ij}) * x_i - sum_j c_{ij} * x_j
        sum_coeffs = np.sum(coeffs, axis=1, keepdims=True) # (n, 1)
        # sum_j c_{ij} x_j is matrix multiplication of coeffs (n,n) and pts (n,d)
        weighted_pts = coeffs @ pts # (n, d)
        
        forces = sum_coeffs * pts - weighted_pts # (n, d)
        
        # The objective is sum over i<j. The force derived is for sum over i != j?
        # Let's check.
        # Objective = 0.5 * sum_{i != j} d_{ij}^{-2}
        # Gradient = 0.5 * sum_{i != j} -2 d_{ij}^{-4} (x_i - x_j)/d_{ij}
        #           = sum_{i != j} - d_{ij}^{-5} (x_i - x_j)
        # My coeff was -2. So I need to divide forces by 2?
        # Let's re-evaluate.
        # f = sum_{i<j} d_{ij}^{-2}
        # df/dx_i = sum_{j>i} -2 d_{ij}^{-4} (x_i - x_j)/d_{ij} + sum_{j<i} -2 d_{ji}^{-4} (x_j - x_i)/d_{ji}
        # Note d_{ji} = d_{ij}.
        # Term 1 (j>i): -2 (x_i - x_j) / d_{ij}^5
        # Term 2 (j<i): -2 (x_j - x_i) / d_{ij}^5 = 2 (x_i - x_j) / d_{ij}^5
        # This is getting confusing. Let's stick to the potential energy analogy.
        # Potential V = sum_{i<j} 1/d_{ij}^2.
        # Force on i is -grad_i V.
        # grad_i V = sum_{j != i} grad_i (1/d_{ij}^2)  (since each pair appears once, but for a fixed i, it interacts with all j)
        # Wait, if i < j, term is 1/d_{ij}^2. If i > j, term is 1/d_{ji}^2 = 1/d_{ij}^2.
        # So for fixed i, the sum is over all j != i of 1/d_{ij}^2.
        # grad_i (1/d_{ij}^2) = -2 d_{ij}^{-3} * (1/d_{ij}^2) * (x_i - x_j) / d_{ij} ? No.
        # d/dx (d^{-2}) = -2 d^{-3} (dd/dx). dd/dx = (x_i - x_j)/d.
        # So grad = -2 d^{-3} * d^{-1} * (x_i - x_j) = -2 (x_i - x_j) / d^4.
        # So grad_i V = sum_{j != i} -2 (x_i - x_j) / d_{ij}^4.
        # My previous coeff was 1/d^5. Let's re-verify powers.
        # f = d^{-2}. f' = -2 d^{-3}. d' = (x_i-x_j)/d.
        # Chain rule: f' d' = -2 d^{-3} * (x_i-x_j)/d = -2 (x_i-x_j) / d^4.
        # So the gradient of the objective is sum_{j != i} -2 (x_i - x_j) / d_{ij}^4.
        
        # Let's re-calculate with power 4.
        coeffs2 = np.zeros_like(dists_full)
        coeffs2[mask] = -2.0 / (dists_full[mask] ** 4)
        
        sum_coeffs2 = np.sum(coeffs2, axis=1, keepdims=True)
        weighted_pts2 = coeffs2 @ pts
        forces2 = sum_coeffs2 * pts - weighted_pts2
        
        return forces2.flatten()

    def constraints(x):
        """
        Constraints: points must be on the unit sphere.
        """
        pts = x.reshape(n, d)
        cons = []
        for i in range(n):
            cons.append({
                'type': 'eq',
                'fun': lambda x, i=i: np.sum(x[i*d:(i+1)*d] ** 2) - 1.0,
                'jac': lambda x, i=i: np.concatenate([
                    np.zeros(i*d),
                    2.0 * x[i*d:(i+1)*d],
                    np.zeros((n - i - 1) * d)
                ])
            })
        return cons

    best_score = -np.inf
    best_points = None

    # Multi-start optimization
    for seed in range(10):
        np.random.seed(seed)
        # Initialize random points on sphere
        init_pts = np.random.randn(n, d)
        init_pts /= np.linalg.norm(init_pts, axis=1, keepdims=True)
        x0 = init_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='SLSQP', jac=gradient, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                pts = res.x.reshape(n, d)
                # Calculate actual metric
                dists = pdist(pts)
                if len(dists) == 0:
                    continue
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max == 0:
                    score = 0.0
                else:
                    score = (d_min / d_max) ** 2
                
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            pass

    if best_points is None:
        # Fallback to random if optimization failed completely
        np.random.seed(0)
        best_points = np.random.randn(n, d)
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
