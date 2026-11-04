import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_radius(centers):
    """
    Compute the maximum radius for the given centers.
    Returns the radius value.
    """
    # Distance between pairs
    # Use broadcasting for vectorization
    # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=-1))
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # More efficient:
    # We can use scipy.spatial.distance.pdist but it returns condensed form.
    # Let's stick to simple vectorization, N=100 is small.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # centers are in [-0.5, 0.5]
    # dist to wall = 0.5 - abs(coord)
    dists_wall = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_wall)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def violation_energy(centers, r):
    """
    Calculate the violation energy for a given radius r.
    Lower is better. 0 means valid packing.
    """
    centers = np.asarray(centers).reshape(-1, 3)
    N = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Pairwise violations: max(0, 2r - dist)^2
    # Only sum over upper triangle to avoid double counting? 
    # Actually, summing all is fine, just scales the energy.
    # But we must be careful with diagonal (0 dist). We set diagonal to inf, so max(0, 2r - inf) = 0.
    violations_pair = np.maximum(0, 2*r - dists)
    energy_pair = np.sum(violations_pair**2)
    
    # Wall violations
    # For each point and each dimension, check if |coord| > 0.5 - r
    # max(0, |coord| - (0.5 - r))^2
    wall_limits = 0.5 - r
    # abs_centers = np.abs(centers)
    # violations_wall = np.maximum(0, abs_centers - wall_limits)
    # energy_wall = np.sum(violations_wall**2)
    
    # Vectorized wall check
    # centers shape (N, 3)
    # We want sum over i, k of max(0, |c_ik| - limit)^2
    # This is equivalent to sum of squares of the positive parts.
    violations_wall = np.maximum(0, np.abs(centers) - wall_limits)
    energy_wall = np.sum(violations_wall**2)
    
    return energy_pair + energy_wall

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Number of restarts
    # We will try a few different initializations
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate initial configuration
        if i == 0:
            # BCC-like lattice initialization
            # 4x4x4 grid of BCC cells -> 128 points. We take 100.
            # Cell size: 1/4 = 0.25
            # Points in cell: (0,0,0) and (0.5,0.5,0.5) relative to cell origin
            # Cell origins: -0.5 + 0.25 * k
            n_x, n_y, n_z = 4, 4, 4
            cell_size = 1.0 / 4.0
            centers = []
            for ix in range(n_x):
                for iy in range(n_y):
                    for iz in range(n_z):
                        cx = -0.5 + ix * cell_size
                        cy = -0.5 + iy * cell_size
                        cz = -0.5 + iz * cell_size
                        # Corner point
                        centers.append([cx, cy, cz])
                        # Center point
                        centers.append([cx + cell_size/2, cy + cell_size/2, cz + cell_size/2])
            centers = np.array(centers)
            # Shuffle and take first 100 to break symmetry
            np.random.shuffle(centers)
            centers = centers[:N]
        elif i == 1:
            # Simple cubic grid 5x5x4
            ticks_x = np.linspace(-0.5, 0.5, 5)
            ticks_y = np.linspace(-0.5, 0.5, 5)
            ticks_z = np.linspace(-0.5, 0.5, 4)
            grid = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
            centers = grid[:N]
        else:
            # Random start
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
        # Optimization loop
        # Start with a small radius and expand
        r_current = 0.05
        step_size = 0.005
        
        # Optional: Pre-optimize at current r to get a valid state?
        # For random starts, they are likely invalid.
        # We minimize violation energy at r_current first.
        
        for _ in range(500): # Max iterations for expansion
            if time.time() > deadline - 5.0:
                break
                
            r_target = r_current + step_size
            
            # Define objective for L-BFGS-B
            # We flatten the centers array
            def obj(x_flat):
                c = x_flat.reshape(N, 3)
                return violation_energy(c, r_target)
            
            def grad(x_flat):
                c = x_flat.reshape(N, 3)
                # Numerical gradient or analytical?
                # Analytical is faster.
                # dE/dc_ik
                # Pair part: sum_{j != i} 2 * max(0, 2r - d_ij) * (-1) * (c_i - c_j)_k / d_ij
                # Wall part: 2 * max(0, |c_ik| - limit) * sign(c_ik)
                
                # This is complex to vectorize correctly for all pairs at once in a simple way without loops.
                # Given N=100, numerical gradient might be slow for L-BFGS-B which calls it many times.
                # Let's use numerical gradient with a small step, or just use BFGS which approximates Hessian but still needs grad.
                # Actually, scipy.optimize.minimize with L-BFGS-B uses numerical gradients if not provided.
                # For N=300 variables, numerical gradient is 300 evaluations per step.
                # Each evaluation is ~4950 distance calcs.
                # 300 * 5000 = 1.5M ops per step.
                # 100 steps = 150M ops.
                # This might be too slow in Python if not vectorized well.
                
                # Let's try to implement analytical gradient.
                g = np.zeros_like(c)
                
                # Pairwise
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                np.fill_diagonal(dists, np.inf)
                
                # v_ij = max(0, 2r - d_ij)
                v = np.maximum(0, 2*r_target - dists)
                
                # dE_pair/dc_i = sum_{j != i} 2 * v_ij * (-diff_ij / d_ij)
                # Note: diff_ij = c_i - c_j
                # term = 2 * v * (-diff / dists)
                # We need to sum over j for each i.
                # term shape (N, N, 3)
                # dists shape (N, N)
                # Avoid division by zero (diagonal is inf, so v is 0, but 0/inf is 0. Off-diagonal dists > 0).
                # However, if dists is 0 (coincident points), it's a problem.
                # We assume no coincident points in valid states, but during optimization they might happen.
                # Add epsilon to dists for safety in division?
                # If dist < eps, force is undefined.
                
                # Safe division
                dists_safe = np.where(dists < 1e-8, 1e-8, dists)
                
                # force_ij = 2 * v_ij * (c_j - c_i) / d_ij
                # Note: gradient of (2r - d)^2 is 2(2r-d)(-1) * dd/dc.
                # dd/dc_i = (c_i - c_j)/d.
                # So grad = 2(2r-d) * (-(c_i-c_j)/d) = 2(2r-d) * (c_j-c_i)/d.
                
                # Vectorized:
                # We want sum_j (2 * v_ij * (c_j - c_i) / d_ij)
                # Let's compute factor = 2 * v / dists_safe
                # factor shape (N, N)
                factor = 2 * v / dists_safe
                # factor[i, j] is the coefficient for (c_j - c_i)
                
                # g[i] = sum_j factor[i, j] * (c_j - c_i)
                # g[i] = sum_j factor[i, j] * c_j - c_i * sum_j factor[i, j]
                
                # sum_j factor[i, j] * c_j
                # factor @ c
                # factor shape (N, N), c shape (N, 3)
                # result shape (N, 3)
                sum_fc = factor @ c
                sum_f = factor.sum(axis=1, keepdims=True)
                
                g_pair = sum_fc - c * sum_f
                
                # Wall part
                # E_wall = sum max(0, |c_ik| - L)^2
                # dE/dc_ik = 2 * max(0, |c_ik| - L) * sign(c_ik)
                L = 0.5 - r_target
                v_wall = np.maximum(0, np.abs(c) - L)
                g_wall = 2 * v_wall * np.sign(c)
                
                g = g_pair + g_wall
                return g.flatten()
            
            x0 = centers.flatten()
            
            # Use L-BFGS-B
            # Bounds: [-0.5, 0.5] for each coordinate
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            res = minimize(obj, x0, method='L-BFGS-B', jac=grad, bounds=bounds, options={'maxiter': 100, 'gtol': 1e-6})
            
            new_centers = res.x.reshape(N, 3)
            new_energy = res.fun
            
            if new_energy < 1e-5:
                # Success
                r_current = r_target
                centers = new_centers
            else:
                # Failed to expand
                # Try smaller step? Or just break.
                # If we break, we keep the current valid (or best) state.
                # But if the initial state was invalid, we might have 0 radius.
                # Let's try to shrink step and retry once?
                # For simplicity, break.
                break
        
        # After expansion loop, compute actual radius
        # It's possible that the last step failed, so r_current might be slightly optimistic if the state drifted?
        # No, we only update r_current if energy is low.
        # So centers correspond to a valid packing of radius r_current (approximately).
        # Let's verify and compute exact radius.
        r_actual = compute_radius(centers)
        
        if r_actual > best_radius:
            best_radius = r_actual
            best_centers = centers.copy()
            
    # If no valid center found (shouldn't happen), return random
    if best_centers is None:
        best_centers = np.random.uniform(-0.45, 0.45, (N, 3))
        
    return best_centers

if __name__ == "__main__":
    # Quick test
    c = construct()
    print("Radius:", compute_radius(c))
