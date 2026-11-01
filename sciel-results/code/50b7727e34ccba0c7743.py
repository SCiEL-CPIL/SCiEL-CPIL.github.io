import numpy as np
import scipy.optimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s buffer for safety, limit is 120s (or 150s? Contract says 120s in text, 150s in directive. Let's assume 120s is the hard kill or 150s. The prompt says "evaluator KILLS ... after 150s". But the task contract says "finish within 120 s". I will use 110s to be safe against both).
    
    # Precompute pair indices for vectorized penalty calculation
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        # Boundary violations
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                        np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # Overlap violations (Vectorized)
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dist_sq = dx**2 + dy**2
        min_dist_sq = (r[i_indices] + r[j_indices])**2
        
        # We need dist >= r_i + r_j
        # Violation is (r_i + r_j - dist)^2 if dist < r_i + r_j
        # Using squared distances to avoid sqrt? 
        # If we use squared distance, the condition is dist_sq < min_dist_sq.
        # The violation magnitude in terms of distance is sqrt(min_dist_sq) - sqrt(dist_sq).
        # This is non-smooth.
        # Let's stick to the previous scalar loop for robustness of the gradient 
        # or use a smooth approximation.
        # Actually, for L-BFGS-B, numerical gradients are used if not provided.
        # The non-smoothness of max(0, ...) is a problem for numerical gradients.
        # Better to use a smooth penalty: 
        # penalty = sum( (max(0, r_i + r_j - dist))^2 )
        # This is smooth except where dist = r_i + r_j.
        
        # Let's calculate dist properly.
        dist = np.sqrt(dist_sq)
        min_dist = r[i_indices] + r[j_indices]
        overlap_viol = np.sum(np.maximum(0, min_dist - dist)**2)
        
        penalty = boundary_viol + overlap_viol
        lambda_val = 500.0
        return -sum_r + lambda_val * penalty

    def is_valid(params, tol=1e-6):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Check bounds
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
            
        # Check overlaps
        dx = x[i_indices] - x[j_indices]
        dy = y[i_indices] - y[j_indices]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[i_indices] + r[j_indices]
        if np.any(dist < min_dist - tol):
            return False
            
        return True

    def get_score(params):
        if is_valid(params, tol=1e-5):
            return np.sum(params[2::3])
        return -np.inf

    # Initializations
    def init_grid():
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.5 / side - 1e-3)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        params = np.zeros(3*N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        return params

    def init_hex():
        # Hexagonal packing approximation
        # Try to fit 32 circles in a hex lattice
        # Rows: 6, 5, 6, 5, 6, 4? Total 32.
        # Or 5, 6, 5, 6, 5, 5?
        # Let's try a rectangular grid with hex offset.
        # 6 columns, 6 rows.
        # Offset odd rows by 0.5 * dx.
        dx = 1.0 / 6
        dy = 1.0 / 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.full(N, 0.5 / dx - 1e-3) # 0.5 * dx is the radius if touching.
        
        idx = 0
        for i in range(6):
            for j in range(6):
                if idx >= N:
                    break
                x[idx] = (j + 0.5) * dx + (0.5 * dx if i % 2 == 1 else 0)
                y[idx] = (i + 0.5) * dy
                idx += 1
        # Adjust x to fit in [0,1]
        # Max x is (5.5 + 0.5)*dx = 6*dx = 1.0. Min x is 0.5*dx.
        # If offset, max x is (5.5+0.5)*dx = 6dx = 1.0.
        # So it fits.
        # However, r = 0.5*dx = 1/12 ~ 0.0833.
        # Distance between centers in same row is dx = 2r.
        # Distance between centers in adjacent rows: sqrt((0.5dx)^2 + dy^2).
        # If dx=dy, dist = sqrt(0.25+1)dx = sqrt(1.25)dx ~ 1.118 dx > 2r = 2*0.5dx = dx.
        # So they don't overlap.
        params = np.zeros(3*N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        return params

    def init_random(seed):
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.08)
        params = np.zeros(3*N)
        params[0::3] = x
        params[1::3] = y
        params[2::3] = r
        return params

    best_result = None
    best_score = -np.inf

    initializations = [
        init_grid(),
        init_hex(),
        init_random(1),
        init_random(2),
        init_random(3)
    ]
    
    # Add perturbed versions
    for i in range(2):
        base = init_grid()
        noise = np.random.uniform(-0.02, 0.02, 3*N)
        perturbed = base + noise
        perturbed = np.clip(perturbed, 0, 1)
        perturbed[2::3] = np.maximum(perturbed[2::3], 0.01)
        initializations.append(perturbed)

    for z0 in initializations:
        if time.time() > deadline:
            break
            
        try:
            res = scipy.optimize.minimize(objective, z0, method='L-BFGS-B', 
                                          options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if is_valid(res.x, tol=1e-6):
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_result = res.x.copy()
                    
        except Exception:
            pass

    # Final polish
    if best_result is not None and time.time() < deadline - 5:
        try:
            def objective_polish(params):
                x = params[0::3]
                y = params[1::3]
                r = params[2::3]
                sum_r = np.sum(r)
                
                viol_x_low = np.maximum(0, r - x)
                viol_x_high = np.maximum(0, r - (1 - x))
                viol_y_low = np.maximum(0, r - y)
                viol_y_high = np.maximum(0, r - (1 - y))
                
                boundary_viol = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                                np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
                
                dx = x[i_indices] - x[j_indices]
                dy = y[i_indices] - y[j_indices]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i_indices] + r[j_indices]
                overlap_viol = np.sum(np.maximum(0, min_dist - dist)**2)
                
                penalty = boundary_viol + overlap_viol
                lambda_val = 5000.0 # Higher penalty for polish
                return -sum_r + lambda_val * penalty

            res = scipy.optimize.minimize(objective_polish, best_result, method='L-BFGS-B', 
                                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if is_valid(res.x, tol=1e-8):
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_result = res.x.copy()
                    
        except Exception:
            pass

    # If no valid result found, return a valid grid (fallback)
    if best_result is None:
        best_result = init_grid()

    result = np.zeros((N, 3))
    result[:, 0] = best_result[0::3]
    result[:, 1] = best_result[1::3]
    result[:, 2] = best_result[2::3]
    
    return result
