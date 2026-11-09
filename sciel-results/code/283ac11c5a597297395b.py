import numpy as np
from scipy.optimize import minimize
import time

N = 300
START_TIME = time.time()
TIME_LIMIT = 140.0 # 10s buffer for 150s limit

def compute_score(points):
    """Calculate the packing radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Boundary constraint
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    
    # Pairwise distances
    # Vectorized calculation
    # diff[i,j] = points[i] - points[j]
    # This is O(N^2) memory, 300x300 is fine.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and lower triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pairwise_dists = dists[mask] / 2.0
    
    if len(pairwise_dists) == 0:
        return np.min(dists_to_boundary)
        
    min_pair = np.min(pairwise_dists)
    min_boundary = np.min(dists_to_boundary)
    
    return np.min([min_pair, min_boundary])

def soft_min_objective(x, beta):
    """
    Objective function and gradient for L-BFGS-B.
    x: (N*2,) array of centers.
    beta: softness parameter.
    """
    points = x.reshape(N, 2)
    
    # 1. Boundary terms
    # d_i = 1 - ||c_i||
    norms = np.linalg.norm(points, axis=1)
    # Avoid division by zero
    norms_safe = np.where(norms < 1e-12, 1e-12, norms)
    d_boundary = 1.0 - norms
    grad_boundary = -points / norms_safe[:, np.newaxis]
    
    # 2. Pairwise terms
    # d_ij = ||c_i - c_j|| / 2
    # Vectorized
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Get pairwise distances and gradients
    # We need to sum e^{-beta * d_k} * grad(d_k)
    
    # To save memory and time, we can iterate or use sparse-like structures, 
    # but N=300 is small enough for dense if we are careful.
    # However, 300x300 matrix is 90k elements. 300x300x2 for gradients is 180k.
    # This is very fast.
    
    d_pairwise = dists[mask] / 2.0
    
    # Gradients for pairwise distances
    # grad_i(d_ij) = (c_i - c_j) / (2 * ||c_i - c_j||)
    # grad_j(d_ij) = (c_j - c_i) / (2 * ||c_i - c_j||) = -grad_i
    
    # We need to accumulate gradients for each point.
    # grad_x = sum_k w_k * grad_k
    
    # Let's compute weights w_k = e^{-beta * d_k}
    # To avoid underflow/overflow, we can shift by max? 
    # Actually, softmin is: (1/beta) * ln(sum(exp(-beta*d)))
    # If d is large, exp(-beta*d) is small.
    # If d is small, exp is close to 1.
    
    # Let's compute the sum of exponentials for the objective value.
    # And the weighted sum of gradients.
    
    # Objective value part:
    # log_sum_exp(-beta * d)
    # For numerical stability: max_d = max(d)
    # ln(sum(exp(-beta*(d - max_d)))) - beta * max_d
    
    # But we also need the weights for the gradient.
    # w_k = exp(-beta * d_k) / sum(exp(-beta * d_j))
    # grad_obj = -1/beta * sum(w_k * grad(d_k))  ... wait.
    
    # f(x) = (1/beta) * ln( sum_k exp(-beta * d_k) )
    # df/dx = (1/beta) * (1 / sum_k exp(-beta * d_k)) * sum_k (-beta * exp(-beta * d_k) * grad(d_k))
    # df/dx = - (sum_k exp(-beta * d_k) * grad(d_k)) / (sum_k exp(-beta * d_k))
    
    # So we need S = sum_k exp(-beta * d_k)
    # and V = sum_k exp(-beta * d_k) * grad(d_k)
    
    # Let's compute S and V.
    
    # Boundary part
    exp_boundary = np.exp(-beta * d_boundary)
    S_boundary = np.sum(exp_boundary)
    V_boundary = np.sum(exp_boundary[:, np.newaxis] * grad_boundary, axis=0)
    
    # Pairwise part
    # We only need i < j
    # d_pairwise is a 1D array of length M = N(N-1)/2
    exp_pairwise = np.exp(-beta * d_pairwise)
    S_pairwise = np.sum(exp_pairwise)
    
    # For gradients, we need to accumulate contributions to each point.
    # grad_pairwise[i,j] = (c_i - c_j) / (2 * dists[i,j])
    # We have the full diff matrix.
    # grad_diff = diff / (2 * dists) ... but dists is 0 on diagonal.
    # We only use mask.
    
    # Let's create a gradient accumulation array
    V_pairwise = np.zeros((N, 2))
    
    # To avoid iterating N^2 times in Python, we can use the full matrix and masking.
    # grad_mat[i,j] = (points[i] - points[j]) / (2 * dists[i,j])
    # We divide by dists, so handle zeros.
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    grad_mat = diff / (2.0 * dists_safe[:, :, np.newaxis])
    
    # Weights matrix W[i,j] = exp(-beta * dists[i,j] / 2) for i!=j, else 0
    W = np.exp(-beta * dists_safe / 2.0)
    W[~mask] = 0.0 # Zero out diagonal and lower triangle
    
    # V_pairwise[i] = sum_j W[i,j] * grad_mat[i,j]
    # Note: grad_mat[j,i] = -grad_mat[i,j]
    # And W[j,i] = W[i,j]
    # So V_pairwise[i] is just the sum over j.
    
    V_pairwise = np.sum(W[:, :, np.newaxis] * grad_mat, axis=1)
    
    # Total S and V
    S_total = S_boundary + S_pairwise
    V_total = V_boundary + V_pairwise
    
    # Objective value
    # log_sum_exp trick for stability?
    # The values are sums of exponentials.
    # If S_total is very small, log is -inf.
    # But d_k are positive (distances), so exp(-beta*d) is between 0 and 1.
    # S_total is at least N * exp(-beta * max_d).
    # For reasonable beta and d, this is fine.
    
    obj_val = (1.0 / beta) * np.log(S_total)
    
    # Gradient
    grad = -V_total / S_total
    
    # Reshape gradient to 1D
    grad_flat = grad.flatten()
    
    return obj_val, grad_flat

def construct():
    # Initial points: Hexagonal lattice
    # Approximate radius of hex lattice for N points
    # Area of unit circle = pi. Area per point = pi/N.
    # Hex lattice point spacing s: area per point = s^2 * sqrt(3)/2.
    # s = sqrt(2*pi*N / (N*sqrt(3))) = sqrt(2*pi/sqrt(3))
    # Wait, area of hex cell is s^2 * sqrt(3)/2.
    # N * s^2 * sqrt(3)/2 = pi  => s = sqrt(2*pi / (N*sqrt(3))) ... No.
    # N points in area pi.
    # pi / N = s^2 * sqrt(3)/2  => s = sqrt(2*pi / (N*sqrt(3)))
    
    s = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate lattice points
    # k range
    k_max = int(1.0 / s) + 2
    points = []
    for i in range(-k_max, k_max + 1):
        for j in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, pick the first N (closest to center? or random?)
    # Better: sort by radius and take first N? Or just take first N generated.
    # The generation order is not radial.
    # Let's sort by radius and take the innermost N.
    if len(points) > N:
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)
        points = points[indices[:N]]
    elif len(points) < N:
        # Should not happen with k_max, but just in case
        # Add random points
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1:
                points = np.vstack([points, p])
    
    # Ensure exactly N
    points = points[:N]
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Optimization loop
    # Try different beta values and restarts
    betas = [5.0, 10.0, 20.0, 50.0]
    
    for beta in betas:
        if time.time() > TIME_LIMIT:
            break
            
        # Jitter initial points slightly to escape symmetry?
        # Or use the result of the previous beta?
        # Let's use the current best points as start.
        x0 = best_points.flatten()
        
        try:
            res = minimize(
                soft_min_objective, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success:
                new_points = res.x.reshape(N, 2)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass

    # Final polish with a very high beta?
    # Or just return best.
    
    return best_points
