import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s eval / 150s kill
    
    # 1. Initialization
    def init_points():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        # Add small noise to break symmetry
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.01, (N, 3))
        pts = pts + noise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms

    P = init_points()
    
    # 2. Samples for Optimization
    S_opt = 10000
    i = np.arange(S_opt) + 0.5
    z = 1.0 - 2.0 * i / S_opt
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples_opt = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 3. L-BFGS Optimization
    lambda_val = 10.0
    
    def covering_loss_and_grad(P_flat):
        P = P_flat.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        # Handle zero norm (should not happen)
        norms[norms == 0] = 1.0
        P = P / norms
        
        # Distances: (S, N)
        dots = samples_opt @ P.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.sqrt(np.maximum(0, 2.0 - 2.0 * dots))
        
        # Min distance for each sample
        min_dists = np.min(dists, axis=1)
        nearest_idx = np.argmin(dists, axis=1)
        
        # Loss: Sum of exp(lambda * min_dist)
        # To avoid overflow, we can subtract max or just use a reasonable lambda
        # lambda=10, dist~0.2 -> exp(2) ~ 7. Safe.
        loss = np.sum(np.exp(lambda_val * min_dists))
        
        # Gradient
        grad = np.zeros_like(P)
        
        # weights = lambda * exp(lambda * min_dists) / min_dists
        weights = np.zeros_like(min_dists)
        valid = min_dists > 1e-6
        # Prevent division by zero
        weights[valid] = lambda_val * np.exp(lambda_val * min_dists[valid]) / min_dists[valid]
        
        # Accumulate gradient: d/dP_i (dist) = (P_i - s) / dist
        # So term is weights[s] * (P_i - s)
        # Vectorized accumulation
        # P[nearest_idx] gives (S, 3)
        # samples_opt is (S, 3)
        # diff = P[nearest_idx] - samples_opt
        # grad += sum over s where nearest_idx[s]==i of weights[s] * diff[s]
        
        # np.add.at is efficient enough for S=10000
        np.add.at(grad, nearest_idx, weights[:, None] * (P[nearest_idx] - samples_opt))
        
        # Chain rule for normalization
        # P = P_raw / norm
        # d P / d P_raw = (I - P P^T) / norm
        # grad_P_raw = (grad_P - (grad_P . P) P) / norm
        
        norms_inv = 1.0 / norms[:, 0]
        # Project gradient to tangent space
        grad_dot_P = np.sum(grad * P, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_P * P
        grad_P_raw = grad_tangent * norms_inv[:, 0]
        
        return loss, grad_P_raw.flatten()
    
    # Run L-BFGS
    # maxiter should be enough to converge
    # We check time periodically? L-BFGS doesn't easily support time limits in callback without breaking.
    # We can wrap it in a try/except or just set a reasonable maxiter.
    # 390 variables. 1000 iterations is usually fast enough.
    
    try:
        res = minimize(
            covering_loss_and_grad,
            P.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        P = res.x.reshape(-1, 3)
        # Normalize again
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
    except Exception as e:
        pass # Keep previous P if it fails
    
    # 4. Greedy Polish
    # Use higher resolution samples for the final polish
    S_polish = 50000
    i = np.arange(S_polish) + 0.5
    z = 1.0 - 2.0 * i / S_polish
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples_polish = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Helper to calculate covering radius metric (max of min dist)
    def calc_metric(P, samples):
        dots = samples @ P.T
        max_dots = np.max(dots, axis=1)
        # We want to maximize min(max_dots)
        # Which is equivalent to minimizing max(arccos(max_dots))
        # For the greedy step, we just need the argmin of max_dots
        return np.min(max_dots), max_dots

    current_val, _ = calc_metric(P, samples_polish)
    
    rng = np.random.default_rng(42)
    
    # Greedy Local Search
    # Move points to cover the worst samples
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        val, max_dots = calc_metric(P, samples_polish)
        
        # Find the sample with the lowest max_dot (worst covered)
        worst_idx = np.argmin(max_dots)
        worst_sample = samples_polish[worst_idx]
        
        # Find which point is nearest to this worst sample
        dots_to_worst = P @ worst_sample
        nearest_pt_idx = np.argmax(dots_to_worst)
        
        p_old = P[nearest_pt_idx].copy()
        
        # Move towards worst sample
        # Tangent direction
        proj = np.dot(p_old, worst_sample)
        grad_dir = worst_sample - proj * p_old
        grad_norm = np.linalg.norm(grad_dir)
        if grad_norm < 1e-6:
            continue
        grad_dir = grad_dir / grad_norm
        
        # Small random perturbation to escape local minima
        random_dir = rng.standard_normal(3)
        random_dir = random_dir - np.dot(random_dir, p_old) * p_old
        random_norm = np.linalg.norm(random_dir)
        if random_norm < 1e-6:
            continue
        random_dir = random_dir / random_norm
        
        # Adaptive step size
        step_size = 0.05 
        new_p = p_old + step_size * (0.9 * grad_dir + 0.1 * random_dir)
        
        # Normalize
        new_p = new_p / np.linalg.norm(new_p)
        
        # Check improvement
        P[nearest_pt_idx] = new_p
        new_val, _ = calc_metric(P, samples_polish)
        
        if new_val > current_val + 1e-7:
            current_val = new_val
            # Keep new_p
        else:
            # Revert
            P[nearest_pt_idx] = p_old
            
    return P
