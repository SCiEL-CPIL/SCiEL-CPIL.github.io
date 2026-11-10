"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r

def _md_optimize(pts, t_start, deadline, seed=42):
    """
    Perform Molecular Dynamics relaxation to find a good local configuration.
    Uses repulsive forces to push circles apart and wall forces to keep them inside.
    """
    rng = np.random.RandomState(seed)
    current = pts.copy()
    
    # Parameters
    k_rep = 10.0       # Repulsion strength
    k_wall = 50.0      # Wall repulsion strength
    margin = 0.0       # We will clip later, but force keeps them away
    dt = 0.01
    n_steps = 5000
    damping = 0.95
    
    velocities = np.zeros_like(current)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
            
        # Compute forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        # O(N^2) is 40k ops, very fast. We can vectorize if needed, but loops are fine for 200.
        # Actually, let's vectorize for speed.
        dists = squareform(pdist(current, 'euclidean'))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        # Force magnitude ~ 1/r^2 or similar. Let's use 1/r^3 * r_vec = r_vec / r^3? 
        # Standard soft repulsion: F = k * (1 - r) if r < 1? 
        # Let's use a simple inverse square law but capped.
        # To be stable, use F = k / r^2 * unit_vec.
        # Vectorized:
        # diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        # dists_sq = np.sum(diff**2, axis=-1)
        # dists_sq[dist_sq == 0] = 1e-10
        # inv_dists_sq = 1.0 / dists_sq
        # forces_vec = k_rep * inv_dists_sq[:, :, np.newaxis] * diff
        # forces = np.sum(forces_vec, axis=1)
        
        # Simpler and faster vectorized version for N=200:
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists_sq[dists_sq == 0] = 1e-10
        # Force is proportional to 1/r^2 in direction of separation
        # F_ij = k * diff_ij / dists_sq_ij^1.5 ? No, that's 1/r^3 * r = 1/r^2 magnitude.
        # Let's use F = k * diff / (dists_sq * sqrt(dists_sq)) = k * diff / dists^3
        # This is strong at close range.
        inv_dists_cubed = 1.0 / (dists_sq * np.sqrt(dists_sq))
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        inv_dists_cubed[~mask] = inv_dists_cubed[~mask]
        inv_dists_cubed[mask] = 0.0
        
        rep_forces = np.sum(k_rep * inv_dists_cubed[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces
        # If x > 0.5, push left. If x < -0.5, push right.
        # Use a linear spring outside bounds? Or exponential?
        # Let's use a strong linear force that kicks in as they approach the boundary.
        # Actually, since we clip, we just need to prevent them from sticking to the wall.
        # Force = k_wall * (0.5 - |x|) if |x| > 0.4? 
        # Let's use a penalty that increases as they get closer to the edge.
        # F_wall_x = k_wall * (0.5 - x) for x > 0.45? 
        # Simpler: F_wall = k_wall * (0.5 - np.abs(x)) * sign(x) ... no.
        # Let's just use a hard constraint in the optimization phase, and for MD,
        # a soft repulsion from the walls.
        
        # Wall repulsion:
        # For each coordinate, if pos > 0.4, force = -k_wall * (pos - 0.4)
        # if pos < -0.4, force = -k_wall * (pos + 0.4)
        # This creates a "cushion" of size 0.1.
        wall_forces = np.zeros_like(current)
        # X-axis
        mask_pos_x = current[:, 0] > 0.4
        wall_forces[mask_pos_x, 0] = -k_wall * (current[mask_pos_x, 0] - 0.4)
        mask_neg_x = current[:, 0] < -0.4
        wall_forces[mask_neg_x, 0] = -k_wall * (current[mask_neg_x, 0] + 0.4)
        
        # Y-axis
        mask_pos_y = current[:, 1] > 0.4
        wall_forces[mask_pos_y, 1] = -k_wall * (current[mask_pos_y, 1] - 0.4)
        mask_neg_y = current[:, 1] < -0.4
        wall_forces[mask_neg_y, 1] = -k_wall * (current[mask_neg_y, 1] + 0.4)
        
        total_forces = rep_forces + wall_forces
        
        # Update velocities and positions
        velocities = damping * velocities + dt * total_forces
        current += dt * velocities
        
        # Clip to box to prevent flying out
        current = np.clip(current, -0.499, 0.499)
        
        # Optional: Anneal dt or k_rep? Keep simple for now.
        
    return current

def _gradient_optimize(pts, t_start, deadline):
    """
    Optimize the minimum pairwise distance using L-BFGS-B.
    Objective: Maximize min(d_ij/2, min_dist_to_wall)
    Equivalent to: Minimize -min(...)
    """
    # Flatten points
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        p = x.reshape(-1, 2)
        # Compute min pair distance
        d = pdist(p, 'euclidean')
        min_pair = d.min()
        # Compute min dist to wall
        bounds = 0.5 - np.abs(p).max(axis=1)
        min_bound = bounds.min()
        r = min(min_pair / 2.0, min_bound)
        return -r
    
    # Gradient is tricky to compute analytically for min function.
    # We can use finite differences or just let L-BFGS-B estimate it.
    # Given N=200, x has 400 dims. L-BFGS-B with numerical gradient might be slow.
    # Let's try it. If it's too slow, we can switch to a simpler coordinate ascent.
    
    # Actually, for 400 dimensions, numerical gradient takes 400 function evaluations per step.
    # Each function eval is O(N^2) ~ 40k ops. 400 * 40k = 16M ops per step.
    # L-BFGS-B might take 100-500 steps. 16M * 500 = 8B ops. Too slow in Python.
    
    # Alternative: Use a custom gradient.
    # The function f(x) = min( d_ij/2 for all i<j, 0.5 - |x_i| for all i,k )
    # The gradient of the minimum is the gradient of the active constraint(s).
    # We can approximate this by finding the active set.
    
    def objective_and_grad(x):
        if time.time() > deadline:
            return 0.0, np.zeros_like(x)
        p = x.reshape(-1, 2)
        
        # Compute all pair distances
        # This is expensive to do every time, but necessary for gradient.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.inf
        min_pair_idx = (-1, -1)
        min_bound_val = np.inf
        min_bound_idx = (-1, -1)
        
        # Find min pair
        # Flatten dists to find min
        flat_dists = dists.flatten()
        min_idx = np.argmin(flat_dists)
        i, j = np.unravel_index(min_idx, (N, N))
        min_pair_val = dists[i, j]
        
        # Find min bound
        bounds = 0.5 - np.abs(p).max(axis=1)
        min_bound_idx_i = np.argmin(bounds)
        min_bound_val = bounds[min_bound_idx_i]
        
        r = min(min_pair_val / 2.0, min_bound_val)
        
        # Gradient
        grad = np.zeros_like(x)
        
        if min_pair_val / 2.0 < min_bound_val:
            # Active constraint is pair (i, j)
            # f = dists[i, j] / 2
            # grad f = 0.5 * (p[i] - p[j]) / dists[i, j]
            # grad wrt p[i] = 0.5 * (p[i] - p[j]) / d
            # grad wrt p[j] = 0.5 * (p[j] - p[i]) / d
            d_vec = p[i] - p[j]
            d_mag = min_pair_val
            if d_mag > 1e-10:
                grad_factor = 0.5 / d_mag
                grad[2*i:2*i+2] = grad_factor * d_vec
                grad[2*j:2*j+2] = -grad_factor * d_vec
        else:
            # Active constraint is bound for point i
            # f = 0.5 - max(|x_i|, |y_i|)
            # Let k be the axis (0 or 1) that gives the max
            i = min_bound_idx_i
            if abs(p[i, 0]) >= abs(p[i, 1]):
                # f = 0.5 - |x_i|
                # grad f = -sign(x_i)
                grad[2*i] = -np.sign(p[i, 0])
            else:
                # f = 0.5 - |y_i|
                # grad f = -sign(y_i)
                grad[2*i+1] = -np.sign(p[i, 1])
        
        return -r, -grad # Minimize -r, so gradient is -grad(r)
        
    # Use L-BFGS-B
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'maxls': 20, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x.reshape(-1, 2)
    except Exception as e:
        print(f"Optimization error: {e}")
        return pts

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # 20s safety margin before 150s kill

    # 1. Initialization
    # Use a hexagonal grid initialization for better starting density
    # Or just a random grid with jitter.
    # Let's use a grid that fits N points roughly.
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    pts += np.random.randn(N, 2) * 0.001
    pts = np.clip(pts, -0.499, 0.499)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Molecular Dynamics Phase
    # Run MD to relax the system
    if time.time() < deadline:
        md_pts = _md_optimize(pts, t_start, deadline, seed=42)
        md_r = _compute_radius(md_pts)
        if md_r > best_r:
            best_r = md_r
            best_pts = md_pts.copy()
            
    # 3. Gradient Descent Phase (L-BFGS-B)
    # Run from the MD result
    if time.time() < deadline:
        grad_pts = _gradient_optimize(best_pts, t_start, deadline)
        grad_r = _compute_radius(grad_pts)
        if grad_r > best_r:
            best_r = grad_r
            best_pts = grad_pts.copy()
            
    # 4. Local Search Refinement (if time remains)
    # Use a simple coordinate ascent to polish
    if time.time() < deadline - 10:
        current_pts = best_pts.copy()
        current_r = best_r
        
        # Coordinate ascent: move one point at a time to maximize local min distance
        # This is expensive if done naively, but we can limit iterations.
        
        # Let's do a few rounds of "push away" for the closest pairs
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            # Find closest pair
            d = pdist(current_pts, 'euclidean')
            min_idx = np.argmin(d)
            # Convert flat index to pair
            # Number of pairs is N*(N-1)/2
            # i = floor((-1 + sqrt(1 + 8*idx*2)) / 2) ? 
            # Simpler: just find min in matrix
            dists = squareform(d)
            np.fill_diagonal(dists, np.inf)
            i, j = np.unravel_index(np.argmin(dists), (N, N))
            
            # Move i and j apart slightly
            vec = current_pts[i] - current_pts[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
                step = 0.001
                new_i = current_pts[i] + vec * step
                new_j = current_pts[j] - vec * step
                new_i = np.clip(new_i, -0.499, 0.499)
                new_j = np.clip(new_j, -0.499, 0.499)
                
                temp_pts = current_pts.copy()
                temp_pts[i] = new_i
                temp_pts[j] = new_j
                temp_r = _compute_radius(temp_pts)
                
                if temp_r > current_r:
                    current_pts = temp_pts
                    current_r = temp_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pts = current_pts.copy()
            
    # Final check
    best_pts = np.clip(best_pts, -0.499, 0.499)
    return best_pts
# EVOLVE-BLOCK-END
