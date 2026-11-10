import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def calculate_radius_full(points):
        """Calculate radius for a full set of points."""
        if len(points) < 2:
            return 1.0
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        return min(min_pair_dist / 2.0, min_bound_dist)

    def generate_hex_lattice(N_points):
        """Generate a hexagonal lattice initialization."""
        s = np.sqrt(2.0 * np.pi / (3.0 * np.sqrt(3.0) * N_points))
        k = int(1.0 / s) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = s * (i + 0.5 * (j % 2))
                y = s * j * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        
        points = np.array(rows)
        dists = np.linalg.norm(points, axis=1)
        points = points[dists <= 1.0 - s/2]
        
        if len(points) < N_points:
            s *= 0.9
            k = int(1.0 / s) + 2
            rows = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = s * (i + 0.5 * (j % 2))
                    y = s * j * np.sqrt(3.0) / 2.0
                    rows.append([x, y])
            points = np.array(rows)
            dists = np.linalg.norm(points, axis=1)
            points = points[dists <= 1.0 - s/2]
            
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        points = points[sorted_idx[:N_points]]
        return points

    # 1. Initialization
    best_points = generate_hex_lattice(N)
    best_radius = calculate_radius_full(best_points)

    # 2. Simulated Annealing
    current_points = best_points.copy()
    current_radius = best_radius
    
    # Precompute distance matrix for fast updates
    D = cdist(current_points, current_points)
    np.fill_diagonal(D, np.inf)
    
    T_start = 0.005
    T_end = 1e-5
    num_sa_iters = 100000
    
    for i_iter in range(num_sa_iters):
        if time.time() > deadline - 20.0: # Leave time for L-BFGS
            break
            
        T = T_start * (T_end / T_start) ** (i_iter / num_sa_iters)
        
        idx = np.random.randint(N)
        
        # Step size: scale with T, but keep a minimum
        step = 0.01 * (0.5 + T / T_start)
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        
        new_point = current_points[idx].copy()
        new_point[0] += dx
        new_point[1] += dy
        
        # Boundary check
        if np.linalg.norm(new_point) > 1.0:
            continue
        
        # Apply move
        old_point = current_points[idx].copy()
        current_points[idx] = new_point
        
        # Update distance matrix
        new_dists = np.linalg.norm(new_point - current_points, axis=1)
        new_dists[idx] = np.inf
        D[idx, :] = new_dists
        D[:, idx] = new_dists
        
        # Calculate new radius
        min_pair = np.min(D)
        r_i = np.linalg.norm(current_points, axis=1)
        min_bound = np.min(1.0 - r_i)
        new_radius = min(min_pair / 2.0, min_bound)
        
        delta_r = new_radius - current_radius
        
        if delta_r > 0:
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
        else:
            # Accept worse move with probability exp(delta_r / T)
            # delta_r is negative
            if np.random.random() < np.exp(delta_r / T):
                current_radius = new_radius
            else:
                # Revert
                current_points[idx] = old_point
                # Revert D
                old_dists = np.linalg.norm(old_point - current_points, axis=1)
                old_dists[idx] = np.inf
                D[idx, :] = old_dists
                D[:, idx] = old_dists

    # 3. Local Refinement with L-BFGS-B on the best SA result
    if time.time() < deadline - 5.0:
        def energy_and_grad(flat_x):
            x = flat_x.reshape((N, 2))
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq_safe = dist_sq + 1e-9
            np.fill_diagonal(dist_sq_safe, 1.0)
            
            inv_dist_sq = 1.0 / dist_sq_safe
            np.fill_diagonal(inv_dist_sq, 0.0)
            E_pair = np.sum(inv_dist_sq)
            
            grad_factor = -2.0 / (dist_sq_safe**2)
            np.fill_diagonal(grad_factor, 0.0)
            grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
            
            r_i = np.linalg.norm(x, axis=1)
            r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
            d_b = 1.0 - r_i
            d_b_safe = np.where(d_b < 1e-9, 1e-9, d_b)
            
            E_bound = np.sum(1.0 / d_b_safe**2)
            grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
            
            total_energy = E_pair + E_bound
            total_grad = grad_pair + grad_bound
            return total_energy, total_grad

        bounds = [(-1.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                energy_and_grad, 
                best_points.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_points = res.x.reshape((N, 2))
            if np.all(np.isfinite(opt_points)):
                r_i = np.linalg.norm(opt_points, axis=1)
                if np.all(r_i <= 1.0):
                    score = calculate_radius_full(opt_points)
                    if score > best_radius:
                        best_points = opt_points
                        best_radius = score
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
