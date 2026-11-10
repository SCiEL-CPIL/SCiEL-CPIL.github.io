import numpy as np
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # Time limit guard
    start_time = time.time()
    time_limit = 500  # Keep well under 600s
    
    def objective_and_grad(points):
        # points: (n, 3)
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        dists[~np.isfinite(dists)] = 1e-10
        dists[dists == 0] = 1e-10
        
        # Objective: Sum of inverse powers
        k = 10
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask] ** k
        
        energy = np.sum(inv_dists) / 2.0
        
        # Gradient
        # grad[i] = sum_j -k * (points[i] - points[j]) / dists[i,j]^(k+2)
        # Let's compute the force matrix
        # F[i,j] = -k * (points[i] - points[j]) / dists[i,j]^(k+2)
        # grad[i] = sum_j F[i,j]
        
        # dists_pow = dists ** (k+2)
        # To avoid issues with diagonal, set diagonal to 1
        dists_safe = dists.copy()
        np.fill_diagonal(dists_safe, 1.0)
        
        # Factor for each pair
        # factor[i,j] = -k / dists[i,j]^(k+2)
        factors = -k / (dists_safe ** (k + 2))
        np.fill_diagonal(factors, 0.0)
        
        # grad[i] = sum_j factors[i,j] * (points[i] - points[j])
        # This can be computed as:
        # grad = (factors * 1) @ points - (factors @ points) ? No.
        # grad_i = sum_j f_ij (p_i - p_j) = p_i sum_j f_ij - sum_j f_ij p_j
        
        sum_factors = np.sum(factors, axis=1)
        sum_f_p = factors @ points
        
        grads = points * sum_factors[:, np.newaxis] - sum_f_p
        
        return energy, grads

    def optimize(points, steps=1000):
        # Adaptive step size
        alpha = 0.01
        best_energy = np.inf
        best_points = points.copy()
        
        for i in range(steps):
            energy, grads = objective_and_grad(points)
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
                
            # Gradient descent
            points = points - alpha * grads
            
            # Normalize to unit sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
            # Decay step size
            alpha *= 0.999
            
            # Check time
            if time.time() - start_time > time_limit - 10:
                break
                
        return best_points

    best_points = None
    best_metric = -1
    
    # Restart 1: Icosahedron + 2 points
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 points at poles (0,0,1) and (0,0,-1) but they are close to existing?
    # (0,0,1) is close to (0,1,phi) normalized?
    # Let's just add random points or specific ones.
    # Actually, let's add 2 points that are far from the icosahedron vertices.
    # Or just use random restarts including one based on icosahedron.
    
    initializations = []
    
    # 1. Icosahedron + 2 random
    p_ico = ico_verts.copy()
    np.random.seed(42)
    p_extra = np.random.randn(2, 3)
    p_extra = p_extra / np.linalg.norm(p_extra, axis=1, keepdims=True)
    initializations.append(np.vstack([p_ico, p_extra]))
    
    # 2. Pure random
    for _ in range(10):
        p_rand = np.random.randn(n, 3)
        p_rand = p_rand / np.linalg.norm(p_rand, axis=1, keepdims=True)
        initializations.append(p_rand)
        
    # 3. Fibonacci sphere (good distribution)
    # Generate Fibonacci sphere points
    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    initializations.append(fibonacci_sphere(n))
    
    # Evaluate all
    for init in initializations:
        if time.time() - start_time > time_limit - 20:
            break
            
        opt_points = optimize(init, steps=2000)
        
        # Calculate metric
        diff = opt_points[:, np.newaxis, :] - opt_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        pairwise_dists = dists[mask]
        
        if len(pairwise_dists) == 0:
            continue
            
        d_min = np.min(pairwise_dists)
        d_max = np.max(pairwise_dists)
        
        if d_max == 0:
            metric = 0
        else:
            metric = (d_min / d_max) ** 2
            
        if metric > best_metric:
            best_metric = metric
            best_points = opt_points
            
    # Final polish with more steps on the best candidate
    if best_points is not None and time.time() - start_time < time_limit - 30:
        best_points = optimize(best_points, steps=5000)
        
        # Recalculate metric to be sure
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        pairwise_dists = dists[mask]
        d_min = np.min(pairwise_dists)
        d_max = np.max(pairwise_dists)
        final_metric = (d_min / d_max) ** 2
        
        # Update if better (it should be, but just in case)
        if final_metric > best_metric:
            best_metric = final_metric
            # best_points already updated
            
    return best_points
