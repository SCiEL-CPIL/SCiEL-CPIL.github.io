"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    def compute_radius(c):
        """Compute the actual radius for a configuration."""
        # c shape (N, 3)
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d[i, j] = np.linalg.norm(c[i] - c[j])
        min_pair_dist = np.min(d[np.triu_indices(N, k=1)])
        
        # Wall distance
        # min_i min_k (0.5 - |c_ik|)
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def soft_min_radius(c, beta=100.0):
        """Differentiable approximation of the radius."""
        c = np.clip(c, -0.499999, 0.499999) # Prevent going outside during gradient steps? 
        # Actually, better to let it go and penalize, or just compute distances.
        # If we use soft-min, we don't need to clip strictly, but it helps stability.
        
        # Pairwise distances
        d = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d[i, j] = np.linalg.norm(c[i] - c[j]) / 2.0
        pair_dists = d[np.triu_indices(N, k=1)]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        # Soft min
        m = np.min(all_dists)
        # Avoid overflow in exp
        exp_terms = np.exp(-beta * (all_dists - m))
        soft_min_val = m - (1.0/beta) * np.log(np.sum(exp_terms))
        
        return -soft_min_val # We want to minimize negative radius

    def gradient_soft_min_radius(c, beta=100.0):
        """Gradient of the soft-min radius (negative for minimization)."""
        c = c.copy()
        
        # Pairwise distances
        d = np.zeros((N, N))
        diff = np.zeros((N, N, 3))
        for i in range(N):
            for j in range(i+1, N):
                diff[i, j] = c[i] - c[j]
                d[i, j] = np.linalg.norm(diff[i, j])
        
        pair_dists = d[np.triu_indices(N, k=1)] / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        wall_dists_flat = wall_dists.flatten()
        
        all_dists = np.concatenate([pair_dists, wall_dists_flat])
        
        m = np.min(all_dists)
        exp_terms = np.exp(-beta * (all_dists - m))
        weights = exp_terms / np.sum(exp_terms)
        
        grad = np.zeros_like(c)
        
        # Gradient of pairwise part
        # d/dx_i ( -1/beta * ln(sum exp) ) = -1/beta * sum (w_j * d/dx_i (-beta * dist_j))
        # = sum (w_j * d(dist_j)/dx_i)
        
        # Let's compute d(dist)/dx for pairs
        # dist_ij = 0.5 * ||c_i - c_j||
        # d(dist_ij)/d(c_i) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        # d(dist_ij)/d(c_j) = 0.5 * (c_j - c_i) / ||c_i - c_j||
        
        pair_weights = weights[:N*(N-1)//2]
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = pair_weights[idx]
                if d[i, j] > 1e-8:
                    vec = (c[i] - c[j]) / d[i, j]
                    grad[i] += 0.5 * w * vec
                    grad[j] -= 0.5 * w * vec
                idx += 1
        
        # Gradient of wall part
        # dist_ik = 0.5 - |c_ik|
        # d(dist_ik)/d(c_ik) = -sign(c_ik)
        
        wall_weights = weights[N*(N-1)//2:]
        wall_weights_reshaped = wall_weights.reshape((N, 3))
        signs = np.sign(c)
        # d(-softmin)/dc = sum(w * d(dist)/dc)
        # d(dist)/dc = -sign(c)
        # So grad_wall = - sum(w * sign(c))
        grad_wall = -np.sum(wall_weights_reshaped * signs, axis=1)
        grad += grad_wall
        
        return grad # This is gradient of -softmin? No.
        # Let's re-verify signs.
        # f = -softmin.
        # d(f)/dx = - d(softmin)/dx.
        # softmin = m - 1/beta ln(S).
        # d(softmin)/dx = -1/beta * (1/S) * dS/dx = -1/beta * (1/S) * sum(exp * d(-beta*dist)/dx)
        # = -1/beta * (1/S) * sum(exp * (-beta) * d(dist)/dx)
        # = (1/S) * sum(exp * d(dist)/dx)
        # = sum(w * d(dist)/dx).
        # So d(softmin)/dx = sum(w * d(dist)/dx).
        # d(-softmin)/dx = - sum(w * d(dist)/dx).
        
        # My calculation above for grad was:
        # grad[i] += 0.5 * w * vec. This is w * d(dist_ij)/dc_i.
        # So I computed sum(w * d(dist)/dc).
        # This is d(softmin)/dc.
        # I want to minimize -softmin, so I need -d(softmin)/dc.
        return -grad

    # 1. Initial Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c0 = grid[:N].copy()
    
    best_c = c0.copy()
    best_r = compute_radius(best_c)
    
    # 2. Gradient Ascent (Minimize -softmin)
    # Use simple gradient descent with adaptive step size
    current_c = c0.copy()
    step_size = 0.01
    prev_loss = soft_min_radius(current_c, beta=50.0)
    
    max_iter = 1000
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        grad = gradient_soft_min_radius(current_c, beta=50.0)
        
        # Check if gradient is zero
        if np.linalg.norm(grad) < 1e-8:
            break
            
        # Take step
        new_c = current_c - step_size * grad
        
        # Clip to box to keep valid? 
        # If we clip, we might get stuck on walls. 
        # But the objective handles walls. 
        # However, if a sphere goes outside, dist becomes negative? 
        # 0.5 - |c| becomes negative. 
        # The soft-min will drive it back.
        
        new_loss = soft_min_radius(new_c, beta=50.0)
        
        if new_loss < prev_loss:
            current_c = new_c
            prev_loss = new_loss
            # Increase step slightly if improving
            # step_size *= 1.01 
        else:
            # Decrease step
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
        # Update best
        r_curr = compute_radius(current_c)
        if r_curr > best_r:
            best_r = r_curr
            best_c = current_c.copy()
            
    # 3. Local Polish with Nelder-Mead on exact objective
    # Flatten the best configuration
    x0 = best_c.flatten()
    
    def exact_obj(x):
        c = x.reshape((N, 3))
        # Ensure finite
        if not np.all(np.isfinite(c)):
            return 1e10
        # Check bounds strictly
        if np.any(c > 0.5) or np.any(c < -0.5):
            # Heavy penalty
            return 1e10 + np.sum(np.abs(c - np.clip(c, -0.5, 0.5)))
        
        r = compute_radius(c)
        return -r

    # Run Nelder-Mead
    # We need to be careful with time
    if time.time() - start_time < timeout - 10:
        try:
            res = minimize(exact_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < exact_obj(x0):
                c_nm = res.x.reshape((N, 3))
                # Validate
                if np.all(np.isfinite(c_nm)) and np.all(c_nm <= 0.5) and np.all(c_nm >= -0.5):
                    r_nm = compute_radius(c_nm)
                    if r_nm > best_r:
                        best_r = r_nm
                        best_c = c_nm
        except Exception:
            pass

    # Final check
    if time.time() - start_time > timeout:
         return best_c
         
    return best_c

# EVOLVE-BLOCK-END
