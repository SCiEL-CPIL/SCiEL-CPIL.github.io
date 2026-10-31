"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check for final result."""
    n = len(radii)
    # Check boundaries
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    # Check overlaps
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # dist_sq < min_dist^2  =>  dist < min_dist
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
        
    return True

def _post_process(pos, radii):
    """
    Shrink radii slightly to ensure strict validity if the optimizer
    left them on the boundary or slightly overlapping due to numerical noise.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        # 1. Boundary fix
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0-x, y, 1.0-y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        # 2. Overlap fix
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Scale both radii to make them just touch
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """
    Generate a tight initial packing using a greedy best-fit algorithm.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(3000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0-x, y, 1.0-y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_lattice():
    """
    Generate a hexagonal lattice-like initial state.
    """
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        if row == 5:
            x = 0.5
            y = row * dy + r
            if k < N:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
            continue
            
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            y = row * dy + r
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        placed = False
        rng = np.random.RandomState(k)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1-x, y, 1-y)
            for j in range(k):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                placed = True
                break
        if not placed:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _vectorized_optimize(pos_init, radii_init, time_limit, seed):
    """
    Vectorized gradient descent with adaptive penalty for circle packing.
    This is the structural core: uses NumPy broadcasting for speed.
    """
    rng = np.random.RandomState(seed)
    pos = pos_init.copy()
    radii = radii_init.copy()
    
    # Start with slightly smaller radii to allow movement
    radii = radii * 0.9
    
    start_time = time.time()
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Hyperparameters
    lr = 0.01
    lr_decay = 0.9999
    mu = 10.0  # Penalty weight
    mu_growth = 1.001
    
    # Precompute mask for upper triangle (i < j)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    idx_i, idx_j = np.where(mask)
    
    while (time.time() - start_time) < time_limit:
        
        # 1. Compute Forces/Gradients using Vectorization
        
        # --- Pairwise Overlap Penalty ---
        # dist_sq[i,j] = (xi-xj)^2 + (yi-yj)^2
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        
        # We want to penalize dist < ri + rj
        # Let d = sqrt(dist_sq). Penalty is max(0, (ri+rj) - d)^2
        # Gradient w.r.t pos: -2 * max(0, (ri+rj) - d) * (pos_i - pos_j) / d
        
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Only calculate for upper triangle to avoid double counting and division by zero on diagonal
        # But for vectorization, it's easier to do all and mask.
        
        # Avoid division by zero
        safe_dist = np.where(dist_sq > 1e-12, np.sqrt(dist_sq), 1e-12)
        
        # Overlap amount: positive if overlapping
        overlap = min_dist - safe_dist
        # Mask to only count actual overlaps (and only i<j to keep gradient direction consistent)
        # Note: The force on i is proportional to (xi-xj). If we sum over all j, 
        # force on i from j is k*(xi-xj). Force on j from i is k*(xj-xi).
        
        # We compute the penalty term P = sum max(0, overlap)^2
        # dP/dxi = sum_j 2 * max(0, overlap_ij) * (xi - xj) / d_ij
        # Note: overlap_ij is symmetric, but the vector (xi-xj) is antisymmetric.
        
        # Let's compute the term T_ij = 2 * max(0, overlap_ij) / d_ij
        # Then grad_xi = sum_j T_ij * (xi - xj)
        
        T = 2.0 * np.maximum(0.0, overlap) / safe_dist
        # Zero out diagonal and lower triangle? 
        # Actually, if we sum over all j, the forces balance out correctly for the system.
        # But we must ensure we don't double count the penalty energy.
        # For gradient descent, we just need the correct direction.
        # The force on i is sum_j k_ij (xi - xj).
        # If we use T from full matrix, force on i = sum_j T_ij (xi - xj).
        # This is correct.
        
        # Zero out diagonal to avoid self-force
        np.fill_diagonal(T, 0.0)
        
        # Force on i from x: sum_j T_ij * (xi - xj)
        # dx[i,j] = xi - xj
        # force_x[i] = sum_j T[i,j] * dx[i,j]
        force_x = np.sum(T * dx, axis=1)
        force_y = np.sum(T * dy, axis=1)
        
        # --- Boundary Penalty ---
        # Penalty = sum (max(0, r_i - x_i)^2 + max(0, x_i - (1-r_i))^2 + ...)
        # Gradient w.r.t x_i:
        # if x_i < r_i: 2*(r_i - x_i) -> push right
        # if x_i > 1-r_i: 2*(x_i - (1-r_i)) -> push left
        
        left_viol = np.maximum(0.0, radii - pos[:, 0])
        right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
        bottom_viol = np.maximum(0.0, radii - pos[:, 1])
        top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
        
        b_force_x = 2.0 * (left_viol - right_viol)
        b_force_y = 2.0 * (bottom_viol - top_viol)
        
        # --- Radius Gradient ---
        # Objective: maximize sum(r) - mu * Penalty
        # d(sum r)/dr = 1
        # d(Penalty)/dr:
        # From overlap: 2 * max(0, overlap) * (1)  (since d(ri+rj)/dri = 1)
        # From boundary: 2 * max(0, r_i - x_i) + 2 * max(0, (1-r_i) - x_i) ... wait
        # Boundary penalty: max(0, r - x)^2. d/dr = 2 max(0, r-x) * 1.
        # max(0, (1-r) - x)^2 is max(0, 1-r-x)^2. d/dr = 2 max(0, 1-r-x) * (-1).
        
        r_grad_overlap = 2.0 * np.maximum(0.0, overlap).sum(axis=1) 
        # Note: overlap is symmetric. Sum over j gives total pressure from overlaps.
        
        r_grad_boundary = 2.0 * (left_viol + right_viol + bottom_viol + top_viol)
        
        # Total gradient for radius: 1 - mu * (r_grad_overlap + r_grad_boundary)
        # We want to ASCEND on radius (maximize), so we add lr * grad
        # grad_r = 1 - mu * (r_grad_overlap + r_grad_boundary)
        
        # --- Update ---
        
        # Add small noise to escape local minima
        noise_x = rng.normal(0, 0.01 * (1.0 - mu/1000.0), N)
        noise_y = rng.normal(0, 0.01 * (1.0 - mu/1000.0), N)
        
        pos[:, 0] += lr * (force_x + mu * b_force_x) + noise_x
        pos[:, 1] += lr * (force_y + mu * b_force_y) + noise_y
        
        # Clamp positions to [0, 1] roughly to prevent divergence
        pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
        pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        # Update radii
        # If gradient is positive, grow. If negative, shrink.
        # We clip radii to [1e-5, 0.5]
        radii += lr * (1.0 - mu * (r_grad_overlap + r_grad_boundary))
        radii = np.clip(radii, 1e-5, 0.5)
        
        # Adaptive Penalty: Increase mu to enforce constraints
        mu *= mu_growth
        lr *= lr_decay
        
        # Check validity and score
        # We only update best if strictly valid or close to it?
        # Post-processing will fix small errors. Let's check a relaxed validity.
        if _is_valid(pos, radii, tol=1e-5):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()
                
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Initialize with lattice and best fit
    pos_lat, rad_lat = _initial_lattice()
    pos_bf, rad_bf = _initial_best_fit()
    
    best_pos = pos_lat.copy()
    best_radii = rad_lat.copy()
    best_score = np.sum(best_radii)
    
    # Run Vectorized Optimization on multiple starts
    # We have ~145 seconds. Vectorized steps are very fast (~1000s/sec).
    # We can run many short optimizations or few long ones.
    # Let's try 10 runs of 10s each? Or 5 runs of 20s?
    # Since it's gradient based, longer runs usually refine better.
    # Let's do 5 runs of 25s each.
    
    num_runs = 5
    time_per_run = 25.0
    
    for k in range(num_runs):
        if time.time() > deadline - 10:
            break
            
        # Alternate between lattice and best fit starts
        if k % 2 == 0:
            p_init, r_init = pos_lat, rad_lat
        else:
            p_init, r_init = pos_bf, rad_bf
            
        # Add random perturbation to escape local minima
        rng = np.random.RandomState(1000 + k)
        p_init = p_init + rng.uniform(-0.02, 0.02, (N, 2))
        r_init = r_init * rng.uniform(0.95, 1.05, N)
        
        # Ensure initial state is not wildly invalid
        p_init, r_init = _post_process(p_init, r_init)
        
        # Run Vectorized Anneal
        p_out, r_out = _vectorized_optimize(p_init, r_init, time_limit=time_per_run, seed=1000+k)
        
        # Post-process result
        p_out, r_out = _post_process(p_out, r_out)
        
        if _is_valid(p_out, r_out):
            score = np.sum(r_out)
            if score > best_score:
                best_score = score
                best_pos = p_out.copy()
                best_radii = r_out.copy()
                
        # Final Polish with L-BFGS on the best found so far
        # This uses the high-quality start from the vectorized method
        if time.time() < deadline - 5:
            
            def _obj(x):
                pos = x[:2*N].reshape(N, 2)
                radii = x[2*N:]
                radii = np.maximum(radii, 1e-6)
                
                # Boundary
                left_viol = np.maximum(0.0, radii - pos[:, 0])
                right_viol = np.maximum(0.0, pos[:, 0] - (1.0 - radii))
                bottom_viol = np.maximum(0.0, radii - pos[:, 1])
                top_viol = np.maximum(0.0, pos[:, 1] - (1.0 - radii))
                boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                                   np.sum(bottom_viol**2) + np.sum(top_viol**2)
                
                # Overlap
                dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
                dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
                min_dist_sq = min_dist * min_dist
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                violation = np.maximum(0.0, min_dist_sq - dist_sq)
                denom = min_dist_sq + 1e-8
                overlap_penalty = np.sum((violation ** 2) / denom[mask])
                
                return -np.sum(radii) + 500.0 * (boundary_penalty + overlap_penalty)
            
            x0 = np.concatenate([best_pos.flatten(), best_radii])
            bounds = []
            for i in range(N):
                bounds.append((0.01, 0.99))
                bounds.append((0.01, 0.99))
            for i in range(N):
                bounds.append((0.01, 0.5))
                
            try:
                res = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.success:
                    pos_opt = res.x[:2*N].reshape(N, 2)
                    radii_opt = np.maximum(res.x[2*N:], 1e-4)
                    pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                    if _is_valid(pos_pp, radii_pp):
                        score = np.sum(radii_pp)
                        if score > best_score:
                            best_score = score
                            best_pos = pos_pp.copy()
                            best_radii = radii_pp.copy()
            except Exception:
                pass

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    if not _is_valid(best_pos, best_radii):
        # Fallback
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N-1:
                    pos[k] = [(i+0.5)/5.0, (j+0.5)/5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
