import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    def score(points):
        """Calculate the score for a set of 8 points."""
        # points: (8, 2) array
        # Scale to diameter 2
        n = len(points)
        # Calculate all pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dists[i, j] = np.linalg.norm(points[i] - points[j])
        
        max_dist = np.max(dists)
        if max_dist == 0:
            return 0.0
        
        # Scale points
        scaled_points = points * (2.0 / max_dist)
        
        # Recalculate distances for scaled points
        # Actually, we can just scale the distances
        scaled_dists = dists * (2.0 / max_dist)
        
        # Product of ordered distances
        # Exclude diagonals (distance 0)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        ordered_dists = scaled_dists[mask]
        
        # To avoid overflow/underflow, use log
        log_prod = np.sum(np.log(ordered_dists))
        
        # Score = exp(log_prod) / 8^8
        # log_score = log_prod - 8 * log(8)
        # We return the raw score or log score?
        # The prompt says "The evaluator measures the raw value".
        # But for optimization, we want to maximize log_score.
        # For returning, we need the actual points.
        # For internal tracking, we use log_score.
        
        # Check for any zeros (should not happen with distinct points)
        if np.any(ordered_dists <= 0):
            return -np.inf
            
        log_score = log_prod - 8 * np.log(8)
        return log_score

    def gradient_ascent(points, steps=500, step_size=0.05):
        """Perform gradient ascent on the log-score."""
        n = 8
        x = points.flatten()
        
        best_log_score = -np.inf
        best_x = x.copy()
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            p = x.reshape(n, 2)
            
            # Calculate distances and gradient
            # dists[i, j]
            dists = np.zeros((n, n))
            grad = np.zeros_like(x)
            
            # Sum of log distances term
            # G(x) = 2 * sum_{i<j} ln(d_ij) - 56 * ln(D)
            # Note: The factor 2 comes from ordered pairs in the score definition?
            # Score = Prod_{i!=j} d_ij / 8^8.
            # ln Score = sum_{i!=j} ln d_ij - 8 ln 8.
            # sum_{i!=j} = 2 * sum_{i<j}.
            # So ln Score = 2 * sum_{i<j} ln d_ij - 8 ln 8.
            # When scaling to diameter 2, d_ij becomes d_ij * (2/D).
            # ln d_ij_scaled = ln d_ij + ln(2) - ln D.
            # Sum over i!=j:
            # 2 * sum_{i<j} (ln d_ij + ln 2 - ln D) - 8 ln 8
            # = 2 * sum_{i<j} ln d_ij + 2 * 28 * ln 2 - 2 * 28 * ln D - 8 ln 8
            # = 2 * sum_{i<j} ln d_ij - 56 * ln D + const.
            
            # So we maximize H(x) = 2 * sum_{i<j} ln d_ij - 56 * ln D.
            
            # Calculate D and the pair (p, q)
            max_d = 0
            p_idx = 0
            q_idx = 1
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(p[i] - p[j])
                    dists[i, j] = d
                    dists[j, i] = d
                    if d > max_d:
                        max_d = d
                        p_idx = i
                        q_idx = j
            
            if max_d < 1e-10:
                return best_x, best_log_score
            
            # Gradient of 2 * sum_{i<j} ln d_ij
            # d/dx_k (2 * sum_{i<j} ln d_ij) = 2 * sum_{j != k} (x_j - x_k) / d_jk^2
            # Note: The sum over i<j includes k as i or j.
            # If k=i, term is ln d_ik. d/dx_k = (x_i - x_k) / d_ik^2? No.
            # d/dx_k ln ||x_i - x_k|| = (x_k - x_i) / ||x_i - x_k||^2.
            # Wait, d/dx ln ||x|| = x / ||x||^2.
            # Let u = x_i - x_k. d/dx_k ln ||u|| = d/dx_k ln sqrt(u^T u) = 1/(2||u||^2) * 2 u^T (-I) = -u/||u||^2 = (x_k - x_i)/||x_k - x_i||^2.
            # So the gradient contribution from pair (i,k) to x_k is (x_k - x_i) / d_ik^2.
            
            for i in range(n):
                for j in range(i+1, n):
                    d = dists[i, j]
                    if d < 1e-10: continue
                    vec = p[i] - p[j]
                    grad_vec = vec / (d * d)
                    # Contribution to x_i: +grad_vec
                    # Contribution to x_j: -grad_vec
                    grad[2*i] += 2.0 * grad_vec[0]
                    grad[2*i+1] += 2.0 * grad_vec[1]
                    grad[2*j] -= 2.0 * grad_vec[0]
                    grad[2*j+1] -= 2.0 * grad_vec[1]
            
            # Gradient of -56 * ln D
            # D = d_pq
            # d/dx_k ln D = d/dx_k ln d_pq
            # If k == p: (x_p - x_q) / d_pq^2
            # If k == q: (x_q - x_p) / d_pq^2
            # Else: 0
            d_pq = max_d
            vec_pq = p[p_idx] - p[q_idx]
            grad_pq = vec_pq / (d_pq * d_pq)
            
            grad[2*p_idx] -= 56.0 * grad_pq[0]
            grad[2*p_idx+1] -= 56.0 * grad_pq[1]
            grad[2*q_idx] += 56.0 * grad_pq[0]
            grad[2*q_idx+1] += 56.0 * grad_pq[1]
            
            # Update
            x = x + step_size * grad
            
            # Calculate current log score for tracking
            curr_log_score = score(p)
            if curr_log_score > best_log_score:
                best_log_score = curr_log_score
                best_x = x.copy()
            
            # Adaptive step size?
            # If score decreased, reduce step size?
            # For simplicity, keep fixed or decay.
            if step % 50 == 0 and step > 0:
                step_size *= 0.95
                
        return best_x, best_log_score

    # Initial guess: Regular octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_global_log_score = -np.inf
    best_global_points = init_points.copy()
    
    # Multiple restarts with perturbation
    np.random.seed(42)
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb the initial points
        noise = np.random.normal(0, 0.1, (8, 2))
        perturbed_points = init_points + noise
        
        # Run gradient ascent
        res_x, res_log_score = gradient_ascent(perturbed_points, steps=800, step_size=0.1)
        
        if res_log_score > best_global_log_score:
            best_global_log_score = res_log_score
            best_global_points = res_x.reshape(8, 2)
            
    # Final scaling to diameter 2
    p = best_global_points
    dists = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            dists[i, j] = np.linalg.norm(p[i] - p[j])
    max_d = np.max(dists)
    if max_d > 0:
        p = p * (2.0 / max_d)
        
    # Convert to list of lists of floats
    return [[float(p[i][0]), float(p[i][1])] for i in range(8)]
