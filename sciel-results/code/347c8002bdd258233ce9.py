import numpy as np
import time
from scipy.optimize import minimize

N = 300
MAX_TIME = 115.0  # Safety margin for 120s limit (evaluator kills at 150s, but let's be safe)

def construct():
    start_time = time.time()
    
    # Strategy:
    # 1. Initialize with concentric rings.
    #    Estimate number of rings and points per ring based on area.
    #    For N=300, typical optimal packings have rings like 1, 6, 12, 18... 
    #    Sum of first k hexagonal numbers is ~ 3k^2. 3k^2 = 300 => k=10.
    #    So roughly 10 rings.
    # 2. Define a parameter vector:
    #    - r_i: radius of ring i (i=0..K-1)
    #    - theta_i: angular offset of ring i
    #    Points in ring i are placed at angles: theta_i + j * (2*pi / n_i)
    # 3. Use Nelder-Mead to maximize the minimum distance (score) in this reduced space.
    # 4. Finally, take the best symmetric configuration and run a few steps of 
    #    full-coordinate local refinement (force-directed) to break symmetry if beneficial.

    # --- Initialization: Determine ring structure ---
    # Heuristic for ring counts:
    # n_i = 6*i for i=1..K-1, n_0 = 1
    # Total points S(K) = 1 + 6 * (1+2+...+K-1) = 1 + 3*K*(K-1)
    # We need S(K) >= N.
    # 3K^2 - 3K + 1 >= 300 => K^2 - K - 99.67 >= 0 => K ~ 10.
    # Let's try K=10 rings.
    # Points: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54. Sum = 1+6*(1+..+9) = 1+6*45 = 271.
    # Need 29 more. Add to outer rings or make an 11th ring?
    # Let's just define a list of ring sizes that sums to 300.
    # Start with hexagonal numbers until we exceed or approach N.
    ring_sizes = [1]
    total = 1
    i = 1
    while total < N:
        next_size = 6 * i
        # If adding full ring exceeds N, adjust the last ring
        if total + next_size > N:
            ring_sizes.append(N - total)
            break
        else:
            ring_sizes.append(next_size)
            total += next_size
            i += 1
    
    K = len(ring_sizes)
    
    # Initial radii: distribute points roughly evenly by area
    # Area of unit circle = pi.
    # Area per ring approx proportional to index? No, annulus area ~ 2*pi*r*dr.
    # Simple init: r_i = i / K
    initial_radii = np.linspace(0.1, 0.95, K)
    initial_angles = np.zeros(K)
    
    # Parameter vector: [r_0, ..., r_{K-1}, theta_0, ..., theta_{K-1}]
    # Note: r_0 is usually 0 for center point. We can fix r_0=0 if ring_sizes[0]==1.
    # To keep it simple, let's optimize all r_i but penalize if r_i < r_{i-1}.
    # Actually, better to just let the optimizer find them.
    # Constraints: 0 <= r_0 <= r_1 <= ... <= r_{K-1} <= 1 - r_max
    # We don't know r_max yet, but it's ~0.05. So r_max_radius < 1.
    
    x0 = np.concatenate([initial_radii, initial_angles])
    
    def unpack_params(params):
        radii = params[:K]
        angles = params[K:]
        return radii, angles

    def get_points_from_params(params):
        radii, angles = unpack_params(params)
        points = []
        for i in range(K):
            n_i = ring_sizes[i]
            r_i = radii[i]
            # Ensure radii are non-negative and ordered? 
            # Nelder-Mead doesn't handle bounds well directly without transformation.
            # We can use a soft penalty in the objective or transform params.
            # Let's just clip radii to [0, 1] in the evaluation.
            if r_i < 0:
                r_i = 0.0
            
            if n_i == 1:
                # Center point? If it's the first ring, usually at 0.
                # But if we allow it to move, it might be better.
                # However, symmetry suggests center is optimal for 1 point.
                # Let's force r_0=0 if n_0==1 to reduce search space?
                # No, let's keep it flexible but initialize near 0.
                pass 
            
            theta_start = angles[i]
            for j in range(n_i):
                theta = theta_start + j * (2.0 * np.pi / n_i)
                x = r_i * np.cos(theta)
                y = r_i * np.sin(theta)
                points.append([x, y])
        
        return np.array(points)

    def objective(params):
        # Maximize min distance -> Minimize negative min distance
        points = get_points_from_params(params)
        
        # Check validity: inside unit circle
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms > 1.0):
            # Penalty for being outside
            return 1e6 + np.sum(norms[norms > 1.0] - 1.0)
        
        # Pairwise distances
        # For K=10, N=300, O(N^2) is 90k ops. Fast enough for Nelder-Mead?
        # Nelder-Mead might take ~1000-5000 evals. 5000 * 90k = 450M ops.
        # In Python/NumPy, 450M simple ops might take 1-2 seconds.
        # It's tight but feasible.
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        min_bound = np.min(1.0 - norms)
        
        r = min(min_pair, min_bound)
        
        # Add a small penalty for non-monotonic radii to help convergence?
        # If r_i > r_{i+1}, it's a valid geometry but harder to interpret.
        # The geometry is what matters.
        
        return -r

    # Run Nelder-Mead
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Limit function evaluations to ensure we finish in time
    # 150s limit. 
    # If we use 3000 evals, and each takes ~0.5ms, that's 1.5s.
    # If each takes 5ms, that's 15s.
    # Let's set maxiter and maxfev.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'maxfev': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME - 20:
        # Not enough time for refinement, return symmetric solution
        final_points = get_points_from_params(best_params)
        # Project to unit circle just in case
        norms = np.linalg.norm(final_points, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            final_points[out_mask] /= norms[out_mask, np.newaxis]
        return final_points

    # --- Phase 2: Full Coordinate Refinement ---
    # Break symmetry and optimize all 300 points locally.
    # This can fix "bottleneck" pairs that are constrained by the rigid ring structure.
    
    x = get_points_from_params(best_params)
    
    # Simple force-directed refinement
    # We want to maximize min(d_ij/2, 1 - |c_i|)
    # Gradient ascent on the soft-min or just push apart close pairs.
    
    alpha = 0.005
    decay = 0.999
    iterations = 0
    
    while time.time() - start_time < MAX_TIME - 5:
        # Compute distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        bound_dists = 1.0 - np.linalg.norm(x, axis=1)
        
        min_pair_val = np.min(dists) / 2.0
        min_bound_val = np.min(bound_dists)
        current_r = min(min_pair_val, min_bound_val)
        
        # Find active constraints
        # Pairs with distance < 2 * current_r + tol
        tol = 1e-4 + 0.1 * current_r
        target_d = 2 * current_r
        
        grad = np.zeros_like(x)
        
        # Vectorized force calculation for close pairs
        # Mask of close pairs
        close_mask = (dists < target_d + tol) & (dists > 1e-8)
        # Exclude diagonal
        close_mask[np.diag_indices_from(close_mask)] = False
        
        if np.any(close_mask):
            # For each close pair (i,j), add force
            # We can vectorize this:
            # diff is x_i - x_j
            # We want to push i away from j and j away from i.
            # Force on i: (x_i - x_j) / dist
            # Force on j: (x_j - x_i) / dist
            
            # Sum of forces on each point
            # F_i = sum_j (x_i - x_j)/d_ij * w_ij
            # We can use matrix multiplication with weighted inverse distances.
            
            inv_dists = np.zeros_like(dists)
            inv_dists[close_mask] = 1.0 / dists[close_mask]
            
            # Weighted difference
            # diff[i,j] = x_i - x_j
            # Weighted diff: diff * inv_dists
            w_diff = diff * inv_dists
            
            # Sum over j for each i -> Force on i
            # F_i = sum_j w_diff[i,j]
            forces = np.sum(w_diff, axis=1)
            
            grad += forces
        
        # Boundary forces
        # Push inward if close to boundary
        bound_close_mask = bound_dists < current_r + tol
        if np.any(bound_close_mask):
            # Force direction: -x_i / |x_i|
            norms = np.linalg.norm(x, axis=1)
            norms_safe = np.where(norms < 1e-8, 1e-8, norms)
            inward_force = -x / norms_safe[:, np.newaxis]
            
            # Only apply to close points
            grad[bound_close_mask] += inward_force[bound_close_mask]
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            grad = grad / grad_norm
        
        x += alpha * grad
        
        # Project to unit circle
        norms = np.linalg.norm(x, axis=1)
        out_mask = norms > 1.0
        if np.any(out_mask):
            x[out_mask] /= norms[out_mask, np.newaxis]
            
        alpha *= decay
        if alpha < 1e-5:
            alpha = 1e-4 # Reset to continue polishing
        
        iterations += 1
        # Safety break if too many iterations
        if iterations > 5000:
            break

    # Final cleanup
    x = np.nan_to_num(x)
    return x
