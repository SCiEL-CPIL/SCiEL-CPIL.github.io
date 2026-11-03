"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def objective_smooth(positions, beta=10.0):
        """
        Smooth objective: negative soft-min of distances.
        Returns (value, gradient).
        """
        positions = positions.reshape(-1, 3)
        
        # Compute all pairwise distances
        dists = pdist(positions, 'euclidean')
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        wall_flat = walls.flatten()
        
        # Combine all constraint values
        # We want to maximize the minimum, so we minimize the negative soft-min
        # soft_min = -1/beta * log(sum(exp(-beta * x)))
        # We define x as the set of all distances (pairwise/2 and wall)
        
        pair_vals = dists / 2.0
        all_vals = np.concatenate([pair_vals, wall_flat])
        
        # Soft min calculation
        neg_beta_x = -beta * all_vals
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(positions)
        
        # Weights for gradient
        # p_j = exp(-beta * x_j) / sum(exp(-beta * x_k))
        p_vals = exp_vals / sum_exp
        
        # Value
        val = -(max_val + np.log(sum_exp)) / beta
        
        # Gradient calculation
        # dE/dx_i = - sum_j p_j * (d(-x_j)/dx_i) = sum_j p_j * (dx_j/dx_i)
        # Wait, E = -soft_min. soft_min approx min(x).
        # Let S = sum exp(-beta x_j). E = -1/beta log S.
        # dE/dx_i = -1/beta * (1/S) * dS/dx_i = -1/beta * (1/S) * sum_j (-beta exp(-beta x_j)) (dx_j/dx_i)
        # dE/dx_i = sum_j (exp(-beta x_j)/S) (dx_j/dx_i) = sum_j p_j (dx_j/dx_i)
        
        grad = np.zeros_like(positions)
        
        # Process pairwise distances
        # p_vals[:len(dists)] correspond to pairs
        pair_p = p_vals[:len(dists)]
        
        # We need to map pair indices back to (i, j)
        # pdist returns condensed matrix.
        # For N=50, number of pairs is 1225.
        # We can iterate through pairs. This is O(N^2) per gradient call.
        # 1225 iterations is fine.
        
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                p = pair_p[idx]
                if p < 1e-10:
                    idx += 1
                    continue
                diff = positions[i] - positions[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    d = 1e-10
                    # grad contribution is undefined, skip or random
                else:
                    # dx_ij/dx_i = diff/d
                    # dx_ij/dx_j = -diff/d
                    # Note: x_j in all_vals is dists[idx]/2
                    # d(dist/2)/dx_i = 0.5 * (diff/d)
                    grad[i] += p * 0.5 * (diff / d)
                    grad[j] -= p * 0.5 * (diff / d)
                idx += 1
        
        # Process wall distances
        # p_vals[len(dists):] correspond to walls
        wall_p = p_vals[len(dists):]
        
        # wall_flat is flattened (N, 3) array
        # index k corresponds to position[k//3, k%3]
        # x_wall = 0.5 - |x|
        # dx_wall/dx = -sign(x)
        for k in range(N * 3):
            p = wall_p[k]
            if p < 1e-10:
                continue
            i = k // 3
            dim = k % 3
            x_val = positions[i, dim]
            # derivative of (0.5 - |x|) w.r.t x is -sign(x)
            # if x == 0, sign is 0 (or undefined, we use 0)
            s = np.sign(x_val)
            grad[i, dim] += p * (-s)
            
        return val, grad

    def local_optimize(positions, maxiter=50):
        """Run local optimization using L-BFGS-B with analytic gradient."""
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def obj(x):
            return objective_smooth(x, beta=10.0)[0]
        
        def grad(x):
            return objective_smooth(x, beta=10.0)[1]
        
        try:
            res = minimize(obj, positions.flatten(), jac=grad, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            pos_opt = res.x.reshape(-1, 3)
            # Clip to ensure within bounds strictly
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def make_initial_grid():
        """Create a reasonable initial grid configuration."""
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    # Time limit management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_positions = make_initial_grid()
    best_radius = compute_radius(best_positions)
    
    # Basin Hopping Parameters
    kick_sigma = 0.1
    min_kick_sigma = 0.001
    max_kick_sigma = 0.5
    cooling_rate = 0.99
    no_improve_count = 0
    max_no_improve = 20
    
    np.random.seed(42)
    
    # Initial local optimization
    best_positions = local_optimize(best_positions, maxiter=100)
    best_radius = compute_radius(best_positions)
    
    while time.time() - start_time < time_limit:
        # Generate kick
        kick = np.random.normal(0, kick_sigma, size=best_positions.shape)
        perturbed = best_positions + kick
        perturbed = np.clip(perturbed, -0.499999, 0.499999)
        
        # Local optimization
        new_positions = local_optimize(perturbed, maxiter=50)
        new_radius = compute_radius(new_positions)
        
        if new_radius > best_radius:
            best_positions = new_positions
            best_radius = new_radius
            no_improve_count = 0
            # Reduce kick size on success (refine)
            kick_sigma = max(kick_sigma * 0.9, min_kick_sigma)
        else:
            no_improve_count += 1
            # Increase kick size on failure (explore)
            kick_sigma = min(kick_sigma * 1.1, max_kick_sigma)
            
            # If stuck for too long, try a larger random restart or swap
            if no_improve_count > max_no_improve:
                # Random restart for a subset of spheres or full random
                # Let's try swapping the worst sphere with a random one
                # Or just a bigger kick
                kick_sigma = max(kick_sigma * 1.5, 0.1)
                no_improve_count = 0
                
    # Final polish
    best_positions = local_optimize(best_positions, maxiter=100)
    best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions
# EVOLVE-BLOCK-END
