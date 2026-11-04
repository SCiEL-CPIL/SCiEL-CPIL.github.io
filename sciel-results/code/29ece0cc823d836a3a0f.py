import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: 5x5x4 grid
    # x, y: 5 points, z: 4 points
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = centers + np.random.normal(0, 0.001, centers.shape) # Small noise to break symmetry
    
    best_centers = centers
    best_score = -np.inf

    def compute_score(c):
        # c is (N, 3)
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out self-distances
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        # c is in [-0.5, 0.5]
        # dist to wall is 0.5 - |c|
        wall_dists = 0.5 - np.abs(c)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    # Soft min objective
    # We want to maximize min(distances)
    # Minimize -soft_min(distances)
    # soft_min(x) = -1/alpha * log(sum(exp(-alpha * x)))
    
    alpha = 200.0
    
    def objective_and_grad(flat_c):
        c = flat_c.reshape(N, 3)
        
        # Calculate all distances
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists[dists == 0] = 1e-10
        
        # Wall distances
        # For each coordinate k, dist is 0.5 - |c_ik|
        # We need to sum exp(-alpha * s) over all s
        # s includes dists/2 (i<j) and wall_dists
        
        # It is more efficient to compute the sum and gradient in one go
        
        # Sum for soft min
        sum_exp = 0.0
        grad = np.zeros_like(c)
        
        # Pairwise terms
        # s_ij = dists[i,j] / 2
        # exp_term = exp(-alpha * s_ij)
        # We iterate over i < j
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                s = d / 2.0
                e = np.exp(-alpha * s)
                sum_exp += e
                # Gradient of -soft_min is 1/(alpha * sum_exp) * sum( e * grad_s )
                # grad_s_ij with respect to c_i is 0.5 * (c_i - c_j) / d
                # grad_s_ij with respect to c_j is 0.5 * (c_j - c_i) / d
                factor = 0.5 / d
                grad[i] += e * factor * (c[i] - c[j])
                grad[j] += e * factor * (c[j] - c[i])
                
        # Wall terms
        for i in range(N):
            for k in range(3):
                # s = 0.5 - |c[i,k]|
                # If c[i,k] > 0, s = 0.5 - c[i,k], grad = -1
                # If c[i,k] < 0, s = 0.5 + c[i,k], grad = 1
                # If c[i,k] == 0, grad = 0 (subgradient)
                ck = c[i, k]
                if ck > 0:
                    s = 0.5 - ck
                    g_s = -1.0
                elif ck < 0:
                    s = 0.5 + ck
                    g_s = 1.0
                else:
                    s = 0.5
                    g_s = 0.0
                    
                e = np.exp(-alpha * s)
                sum_exp += e
                grad[i, k] += e * g_s
                
        # Objective: -soft_min = 1/alpha * log(sum_exp)
        # Wait, soft_min = -1/alpha * log(sum_exp)
        # So -soft_min = 1/alpha * log(sum_exp)
        obj = (1.0 / alpha) * np.log(sum_exp)
        
        # Gradient of obj = 1/(alpha * sum_exp) * grad_sum
        # where grad_sum is the sum computed above (which is sum e * grad_s)
        # Note: The `grad` variable above holds sum(e * grad_s)
        grad = grad / (alpha * sum_exp)
        
        return obj, grad

    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run optimization
    # Use L-BFGS-B
    # We might run multiple iterations or restarts if time permits
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        try:
            res = minimize(
                lambda x: objective_and_grad(x)[0],
                best_centers.flatten(),
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.success:
                cand = res.x.reshape(N, 3)
                score = compute_score(cand)
                if score > best_score:
                    best_score = score
                    best_centers = cand
        except Exception:
            break
            
        # If we have time, we could perturb and restart, but L-BFGS-B on a good start is usually enough.
        # Let's just run once or twice.
        if num_restarts >= 2:
            break

    return best_centers
