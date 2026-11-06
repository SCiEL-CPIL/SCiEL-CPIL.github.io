"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START

def _thomson_init(n, iterations=500):
    """Initialize points using Thomson problem minimization (electrostatic repulsion)."""
    # Start with Fibonacci spiral
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize to minimize electrostatic energy (Thomson problem)
    def energy_and_grad(pts):
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
        dist = np.sqrt(dist_sq)
        
        # Energy: sum of 1/dist for all pairs
        energy = np.sum(1.0 / dist)
        
        # Gradient: -sum over j != i of (pi - pj) / |pi - pj|^3
        # For each point i, grad_i = -sum_{j != i} (pi - pj) / |pi - pj|^3
        inv_dist3 = 1.0 / (dist_sq * dist)
        np.fill_diagonal(inv_dist3, 0.0)
        
        # grad[i] = -sum_j (pi - pj) * inv_dist3[i,j]
        # = -sum_j pi * inv_dist3[i,j] + sum_j pj * inv_dist3[i,j]
        # = -pi * sum_j inv_dist3[i,j] + sum_j pj * inv_dist3[i,j]
        
        sum_inv = np.sum(inv_dist3, axis=1)
        grad = -pts * sum_inv[:, np.newaxis] + pts.T @ inv_dist3
        
        # Project gradient onto tangent plane (remove radial component)
        # For unit sphere, tangent projection: grad_t = grad - (grad . pi) * pi
        radial_comp = np.sum(grad * pts, axis=1)
        grad = grad - radial_comp[:, np.newaxis] * pts
        
        return energy, grad.flatten()
    
    # Use L-BFGS-B with projection to sphere
    best_pts = pts.copy()
    best_energy = energy_and_grad(pts)[0]
    
    for _ in range(iterations):
        def objective(x):
            p = x.reshape(n, 3)
            # Normalize to unit sphere
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            e, _ = energy_and_grad(p)
            return e
        
        def gradient(x):
            p = x.reshape(n, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            p = p / norms
            _, g = energy_and_grad(p)
            return g
        
        result = minimize(objective, best_pts.flatten(), jac=gradient, method='L-BFGS-B',
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        new_pts = result.x.reshape(n, 3)
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        new_energy = result.fun
        
        if new_energy < best_energy:
            best_pts = new_pts
            best_energy = new_energy
    
    return best_pts


def _covering_radius(pts):
    """Compute the covering radius (maximum distance to nearest point) in radians."""
    # For each point on a dense grid, find nearest configuration point
    # Use a spherical grid for approximation
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    max_dist = 0.0
    for t in theta:
        for p in phi:
            x = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            dists = np.arccos(np.clip(np.sum(x * pts, axis=1), -1.0, 1.0))
            max_dist = max(max_dist, np.min(dists))
    
    return max_dist


def _smooth_max_covering(pts, temperature=0.01):
    """Compute smooth approximation of covering radius using log-sum-exp."""
    # For each point on a grid, compute smooth max of distances
    n_theta = 40
    n_phi = 80
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    total = 0.0
    count = 0
    
    for t in theta:
        for p in phi:
            x = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            dists = np.arccos(np.clip(np.sum(x * pts, axis=1), -1.0, 1.0))
            # Smooth max using log-sum-exp
            shifted = dists / temperature
            max_shifted = np.max(shifted)
            log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
            total += temperature * log_sum_exp
            count += 1
    
    return total / count


def construct():
    start_time = time.time()
    deadline = 140.0  # 10 second safety margin
    
    # Step 1: Initialize with Thomson problem
    pts = _thomson_init(N, iterations=200)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return pts
    
    # Step 2: Optimize to minimize covering radius using smooth surrogate
    # Use gradient descent on the smooth covering radius
    
    def objective_and_grad(x):
        p = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        # Compute smooth covering radius and gradient
        # Use a grid of test points
        n_theta = 30
        n_phi = 60
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        
        test_points = []
        for t in theta:
            for ph in phi:
                test_points.append(np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)]))
        test_points = np.array(test_points)
        
        # For each test point, find nearest configuration point and compute gradient
        temperature = 0.02
        obj_val = 0.0
        grad = np.zeros_like(p)
        
        for tp in test_points:
            dists = np.arccos(np.clip(np.sum(tp * p, axis=1), -1.0, 1.0))
            # Weights for smooth max
            shifted = dists / temperature
            max_shifted = np.max(shifted)
            weights = np.exp(shifted - max_shifted)
            weights /= np.sum(weights)
            
            obj_val += temperature * (max_shifted + np.log(np.sum(weights)))
            
            # Gradient contribution: for each configuration point i, 
            # grad_i += weights[i] * d(dist_i)/d(p_i)
            # dist_i = arccos(tp . p_i)
            # d(dist_i)/d(p_i) = -tp / sqrt(1 - (tp . p_i)^2)
            # But we need to project to tangent plane
            
            dot_products = np.sum(tp * p, axis=1)
            sqrt_terms = np.sqrt(np.clip(1.0 - dot_products ** 2, 1e-10, 1.0))
            
            # Gradient of arccos(u) w.r.t. u is -1/sqrt(1-u^2)
            # d(dist_i)/d(p_i) = -tp / sqrt(1 - (tp . p_i)^2)
            grad_contrib = -tp[:, np.newaxis] / sqrt_terms[:, np.newaxis]
            
            # Weighted sum
            grad += np.sum(weights[:, np.newaxis] * grad_contrib, axis=0)
        
        # Project gradient to tangent plane
        radial_comp = np.sum(grad * p, axis=1)
        grad = grad - radial_comp[:, np.newaxis] * p
        
        return obj_val / len(test_points), grad.flatten()
    
    # Gradient descent with adaptive step size
    current_pts = pts.copy()
    current_obj = objective_and_grad(current_pts.flatten())[0]
    best_pts = current_pts.copy()
    best_obj = current_obj
    
    step_size = 0.01
    for iteration in range(500):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        obj, grad = objective_and_grad(current_pts.flatten())
        
        # Take gradient step
        new_pts = current_pts - step_size * grad.reshape(N, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / norms
        
        new_obj = objective_and_grad(new_pts.flatten())[0]
        
        if new_obj < current_obj:
            current_pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_pts = current_pts.copy()
                best_obj = current_obj
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts

# EVOLVE-BLOCK-END

