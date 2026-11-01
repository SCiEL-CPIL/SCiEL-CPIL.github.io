import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles inside a unit circle
    to maximize the common radius.
    """
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin for safety
    
    # 1. Initialization: Hexagonal Lattice
    # A hex lattice is a good starting point as it's close to optimal density.
    # We generate a grid and select points closest to the center.
    spacing = 0.2
    max_iter_spacing = 100
    points = None
    for _ in range(max_iter_spacing):
        k = int(1.0 / spacing) + 2
        # Generate hex grid points
        xs = []
        ys = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                xs.append(x)
                ys.append(y)
        pts = np.column_stack((xs, ys))
        r = np.hypot(pts[:, 0], pts[:, 1])
        # Filter points inside unit circle (with some margin)
        mask = r < 1.0
        pts_in = pts[mask]
        r_in = r[mask]
        if len(pts_in) >= N:
            # Sort by distance to center to pick the "densest" center cluster
            idx = np.argsort(r_in)
            points = pts_in[idx[:N]]
            break
        spacing *= 0.95
        
    if points is None:
        # Fallback to random if lattice fails (unlikely)
        points = np.random.randn(N, 2) * 0.5

    best_score = -np.inf
    best_points = points.copy()
    
    # 2. Optimization using Soft-Min Gradient Descent
    
    def get_distances(pts):
        # pts: (N, 2)
        # Returns array of distances: 
        # 1 - |c_i| for each i
        # |c_i - c_j| / 2 for each i < j
        r_c = np.hypot(pts[:, 0], pts[:, 1])
        d_boundary = 1.0 - r_c
        
        # Pairwise distances
        # Use broadcasting to compute all pair distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_pair = np.linalg.norm(diff, axis=2) / 2.0
        # Only take upper triangle
        iu = np.triu_indices(N, 1)
        d_pair_min = d_pair[iu]
        
        return np.concatenate([d_boundary, d_pair_min])

    def objective_and_grad(x, beta):
        pts = x.reshape(N, 2)
        
        # Compute distances
        r_c = np.hypot(pts[:, 0], pts[:, 1])
        d_boundary = 1.0 - r_c
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_pair = np.linalg.norm(diff, axis=2) / 2.0
        iu = np.triu_indices(N, 1)
        d_pair_min = d_pair[iu]
        
        # All distances
        d_all = np.concatenate([d_boundary, d_pair_min])
        
        # Soft-min calculation
        # f(x) = (1/beta) * log(sum(exp(-beta * d)))
        # We want to MINIMIZE this (which maximizes the min distance)
        
        # Numerical stability: subtract max d (which is min -d)
        # exp(-beta * d)
        # If beta is large, exp(-beta * d_small) is huge.
        # Let M = min(d_all). Then exp(-beta * (d - M)) = exp(-beta * d) * exp(beta * M).
        # log(sum(exp(-beta * d))) = beta * M + log(sum(exp(-beta * (d - M))))
        # This is more stable.
        
        M = np.min(d_all)
        shifted = d_all - M
        # If d is very close to M, shifted is 0, exp(0)=1.
        # If d >> M, shifted is large, exp(-large) ~ 0.
        weights_unnorm = np.exp(-beta * shifted)
        S = np.sum(weights_unnorm)
        
        # Objective value
        obj = (M + np.log(S)) / beta
        
        # Gradient
        # d/dx [ (1/beta) log S ] = (1/beta) * (1/S) * dS/dx
        # dS/dx = sum_k (d/dx exp(-beta * d_k)) = sum_k exp(-beta * d_k) * (-beta) * (d d_k / dx)
        # So grad = (1/beta) * (1/S) * (-beta) * sum_k (exp(-beta * d_k) * grad_d_k)
        # grad = - (1/S) * sum_k (exp(-beta * d_k) * grad_d_k)
        
        # Normalized weights w_k = exp(-beta * d_k) / S
        # Note: using shifted d for stability in exp, but w_k needs to be consistent.
        # exp(-beta * d_k) = exp(-beta * (d_k - M)) * exp(-beta * M)
        # The exp(-beta * M) term cancels out in the ratio w_k = e_k / S.
        # So we can use w_k = weights_unnorm[k] / S.
        
        w = weights_unnorm / S
        
        grad = np.zeros_like(pts)
        
        # 1. Boundary contributions
        # d_i = 1 - |c_i|
        # grad_{c_i} d_i = - c_i / |c_i|
        # Contribution to grad: - w_i * grad_d_i = - w_i * (-c_i / |c_i|) = w_i * c_i / |c_i|
        
        # Avoid division by zero
        r_c_safe = np.where(r_c > 1e-8, r_c, 1e-8)
        # w_boundary is the first N weights
        w_b = w[:N]
        grad_boundary = w_b[:, np.newaxis] * (pts / r_c_safe[:, np.newaxis])
        grad += grad_boundary
        
        # 2. Pair contributions
        # d_ij = |c_i - c_j| / 2
        # grad_{c_i} d_ij = (c_i - c_j) / (2 |c_i - c_j|)
        # grad_{c_j} d_ij = (c_j - c_i) / (2 |c_i - c_j|)
        # Contribution to grad_i: - w_ij * grad_{c_i} d_ij
        # Contribution to grad_j: - w_ij * grad_{c_j} d_ij
        
        # w_pair are the remaining weights
        w_p = w[N:]
        
        # We need to accumulate gradients for each point based on all pairs it participates in.
        # Efficient way: use the full difference matrix and weight matrix.
        
        # Full weight matrix W_full (N x N)
        # W_full[i, j] = w_pair[k] if (i,j) is the k-th pair in iu, else 0.
        # Symmetric.
        
        # Let's construct the full weight matrix for pairs
        W_full = np.zeros((N, N))
        W_full[iu] = w_p
        W_full = W_full + W_full.T
        
        # Distance matrix D_full (N x N)
        # D_full[i, j] = |c_i - c_j|
        D_full = np.linalg.norm(diff, axis=2)
        
        # Safe distance
        D_full_safe = np.where(D_full > 1e-8, D_full, 1e-8)
        
        # grad_pair[i] = - sum_j W_full[i, j] * (c_i - c_j) / (2 * D_full_safe[i, j])
        # Note: The factor 1/2 is in d_ij definition.
        # grad_{c_i} d_ij = (c_i - c_j) / (2 * |c_i - c_j|)
        
        # Vectorized calculation:
        # grad_pair = - 0.5 * sum_j (W_full[i, j] / D_full_safe[i, j]) * (c_i - c_j)
        
        # Let's compute term by term or use broadcasting.
        # coeff[i, j] = W_full[i, j] / (2 * D_full_safe[i, j])
        coeff = W_full / (2.0 * D_full_safe)
        
        # grad_pair[i] = - sum_j coeff[i, j] * (pts[i] - pts[j])
        # = - (sum_j coeff[i, j] * pts[i]) + (sum_j coeff[i, j] * pts[j])
        # = - pts[i] * sum_j coeff[i, j] + sum_j coeff[i, j] * pts[j]
        
        sum_coeff = np.sum(coeff, axis=1)
        # sum_j coeff[i, j] * pts[j] is matrix multiplication coeff @ pts
        weighted_pts = coeff @ pts
        
        grad_pair = - (pts * sum_coeff[:, np.newaxis] + weighted_pts)
        
        grad += grad_pair
        
        return obj, grad

    # Optimization Loop
    # We increase beta to sharpen the minimum.
    # Start with a moderate beta.
    betas = [10, 50, 100, 200, 500, 1000]
    
    current_points = points
    for beta in betas:
        if time.time() - start_time > max_time - 10:
            break
            
        x0 = current_points.flatten()
        
        # Use L-BFGS-B
        # We minimize the soft-min.
        # Note: The objective is smooth.
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < np.inf:
                new_points = res.x.reshape(N, 2)
                # Check validity
                r_c = np.hypot(new_points[:, 0], new_points[:, 1])
                if np.all(r_c < 1.0):
                    # Calculate actual score
                    d_b = 1.0 - r_c
                    diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
                    d_p = np.linalg.norm(diff, axis=2) / 2.0
                    iu = np.triu_indices(N, 1)
                    d_p_min = d_p[iu]
                    score = np.min(np.concatenate([d_b, d_p_min]))
                    
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        current_points = new_points
        except Exception:
            pass

    # Final check and return
    # Ensure the returned points are the best ones found
    # Calculate final score to be sure
    r_c = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(r_c >= 1.0):
        # Fallback to a safe configuration if optimization failed badly
        # Scale down to fit
        max_r = np.max(r_c)
        if max_r > 0:
            best_points = best_points * (0.99 / max_r)
            
    return best_points
