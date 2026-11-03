"""
Combined approach: Vectorized Smooth Surrogate Optimization for Sphere Packing.
Combines the structural search of the current program with the vectorized gradient
computation and superior lattice initialization from the inspiration.
"""
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(positions):
        """Compute the exact minimum radius for given positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (N, 3) or not np.all(np.isfinite(positions)):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Wall distances
        dist_to_walls = np.min(0.5 - np.abs(positions), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    # Pre-compute indices for vectorization
    tri_i, tri_j = np.triu_indices(N, k=1)

    def smooth_objective_and_grad(c_flat, beta):
        """
        Computes the smooth surrogate objective (-soft_min) and its gradient.
        c_flat: (3N,)
        beta: temperature parameter
        Returns: (value, grad_flat)
        """
        c = c_flat.reshape(N, 3)
        
        # 1. Calculate all "radius candidates" g_k
        # Pairwise: d_ij / 2
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0
        
        # Walls: 6 distances per point
        # d_x_pos = 0.5 - c[:, 0]  (distance to x=+0.5)
        # d_x_neg = c[:, 0] + 0.5  (distance to x=-0.5)
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
        
        # Combine all g_k
        g_all = np.concatenate([d_pairs, wall_dists])
        
        # 2. Soft Min Calculation
        # soft_min(g) = -1/beta * log(sum(exp(-beta * g)))
        # We want to maximize soft_min, so we minimize -soft_min.
        # objective = -soft_min = 1/beta * log(sum(exp(-beta * g)))
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = log_sum_exp / beta
        
        # 3. Gradient Calculation
        # d/dg [1/beta * logsumexp(-beta * g)] = -softmax(-beta * g)
        # softmax(x) = exp(x) / sum(exp(x))
        # Here x = -beta * g_all
        # weights = exp(-beta * g_all) / sum(exp(-beta * g_all))
        # To avoid overflow/underflow, we use the log_sum_exp already computed.
        # exp(-beta * g_all - log_sum_exp) is the stable softmax.
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        grad_c = np.zeros_like(c)
        
        # Pairwise gradient
        # d/dc_i (d_ij/2) = (c_i - c_j) / (2 * d_ij)
        # d/dc_j (d_ij/2) = (c_j - c_i) / (2 * d_ij) = - (c_i - c_j) / (2 * d_ij)
        # Objective term for pair k is w_k * (d_k/2).
        # d(obj)/dc_i = w_k * (c_i - c_j) / (2 * d_k)
        # d(obj)/dc_j = w_k * (c_j - c_i) / (2 * d_k)
        
        n_pairs = len(d_pairs)
        # safe_d = np.maximum(d_pairs, 1e-12)
        # factor_k = weight_k / (2 * d_k)
        # Wait, d_pairs is already d/2.
        # d_pairs = ||diff|| / 2.
        # d(d_pairs)/dc_i = diff / (2 * ||diff||) = diff / (4 * d_pairs)
        # So grad contribution is w_k * diff / (4 * d_pairs)
        
        safe_d_pairs = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d_pairs)
        # Zero out factors where distance is effectively 0 to avoid numerical issues, 
        # though weight should be small there anyway if beta is not infinite.
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        
        weighted_diffs = factors[:, np.newaxis] * diff
        
        # Scatter: add to index i, subtract from index j
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)
        
        # Wall part
        # w_start = n_pairs
        # d_x_pos = 0.5 - c[:, 0] -> d/dc[:,0] = -1
        # d_x_neg = c[:, 0] + 0.5 -> d/dc[:,0] = +1
        # ... and so on
        
        w_start = n_pairs
        grad_c[:, 0] -= weights[w_start : w_start+N]       # d_x_pos
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]   # d_x_neg
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N] # d_y_pos
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N] # d_y_neg
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N] # d_z_pos
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N] # d_z_neg
        
        # We want to MINIMIZE val (which is -soft_min).
        # The gradient calculated above is for val.
        # So we return val and grad_c.
        return val, grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget):
        """Run L-BFGS-B with smooth objective."""
        if remaining() < time_budget:
            return c_init
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            val, _ = smooth_objective_and_grad(x, beta)
            return val
            
        def grad_flat(x):
            _, grad = smooth_objective_and_grad(x, beta)
            return grad
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init
            
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_radius(x.reshape(N, 3))
            
        try:
            res = opt.minimize(
                obj_flat, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            return res.x.reshape(N, 3)
        except Exception:
            return c_init

    def make_initial_grid():
        """4x4x4 grid, take first 50."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        """Body-centered cubic inspired."""
        positions = []
        a = 0.5
        spacing = (2*a) / 3.0
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * spacing
                    y = -a + (iy + 0.5) * spacing
                    z = -a + (iz + 0.5) * spacing
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + spacing/2, y + spacing/2, z + spacing/2])
        return np.array(positions[:N])

    def make_fcc_like():
        """FCC-inspired: generate dense FCC lattice, select best N by local density."""
        # Generate FCC lattice points in a larger region
        a = 0.35  # smaller lattice constant for denser packing
        positions = []
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    x = (ix - 1.5) * a
                    y = (iy - 1.5) * a
                    z = (iz - 1.5) * a
                    positions.append([x, y, z])
                    # Face centers of the cubic cell
                    if ix < 3:
                        positions.append([x + a/2, y, z])
                    if iy < 3:
                        positions.append([x, y + a/2, z])
                    if iz < 3:
                        positions.append([x, y, z + a/2])
                    if ix < 3 and iy < 3:
                        positions.append([x + a/2, y + a/2, z])
                    if ix < 3 and iz < 3:
                        positions.append([x + a/2, y, z + a/2])
                    if iy < 3 and iz < 3:
                        positions.append([x, y + a/2, z + a/2])
        
        positions = np.array(positions, dtype=np.float64)
        
        # Keep only points that fit well within the cube
        mask = np.all(np.abs(positions) <= 0.48, axis=1)
        positions = positions[mask]
        
        # If we have more than N, select the N with best local spacing
        if len(positions) > N:
            # Compute pairwise distances and prefer well-separated points
            # Simple heuristic: maximize minimum distance to other selected points
            # For speed, we just take a random subset or the first N if it's close.
            # Better: Greedy selection based on distance to already selected.
            selected_indices = []
            remaining_indices = np.arange(len(positions))
            
            # Start with a point near center
            center_dist = np.linalg.norm(positions, axis=1)
            first_idx = np.argmin(center_dist)
            selected_indices.append(first_idx)
            remaining_indices = np.delete(remaining_indices, first_idx)
            
            while len(selected_indices) < N and len(remaining_indices) > 0:
                # For each candidate, compute min distance to selected
                # This is O(N^2 * N) which might be slow, but N=50 is small.
                # We can vectorize this step.
                selected_pos = positions[selected_indices]
                cand_pos = positions[remaining_indices]
                
                # Compute distances between each candidate and all selected
                # dists shape: (num_cands, num_selected)
                dists = np.linalg.norm(cand_pos[:, np.newaxis, :] - selected_pos[np.newaxis, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                
                # Pick the candidate with the largest minimum distance
                best_local_idx = np.argmax(min_dists)
                global_idx = remaining_indices[best_local_idx]
                selected_indices.append(global_idx)
                remaining_indices = np.delete(remaining_indices, best_local_idx)
            
            positions = positions[selected_indices]
        elif len(positions) < N:
            # Pad with random points if we didn't generate enough
            # This shouldn't happen with the parameters above
            extra = np.random.uniform(-0.45, 0.45, size=(N - len(positions), 3))
            positions = np.vstack([positions, extra])
            
        return positions

    def make_random(seed=None):
        """Create random initial positions."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))

    def make_perturbed(base_positions, seed, sigma=0.02):
        """Create perturbed version of a base configuration."""
        np.random.seed(seed)
        perturbed = base_positions + np.random.normal(0, sigma, size=base_positions.shape)
        return np.clip(perturbed, -0.499999, 0.499999)

    best_positions = None
    best_radius = -1.0
    
    # Initializations to try
    # Prioritize the high-quality FCC start
    initializations = [
        ("fcc", make_fcc_like),
        ("bcc", make_bcc_lattice),
        ("grid", make_initial_grid),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
    ]
    
    # Phase 1: Main optimization loop
    for name, init_func in initializations:
        if remaining() < 20:
            break
        
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            
            # Smooth optimization with annealing
            # Start with lower beta for global exploration, increase for local refinement
            beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0]
            
            current_pos = positions
            for beta in beta_schedule:
                if remaining() < 10:
                    break
                # Allocate more iterations to higher beta stages
                maxiter = 100 if beta < 100 else 50
                current_pos = local_optimize_smooth(current_pos, beta=beta, maxiter=maxiter, time_budget=5)
            
            # Final direct polish
            if remaining() > 10:
                current_pos = local_optimize_direct(current_pos, maxiter=100, time_budget=5)
            
            r = compute_radius(current_pos)
            if r > best_radius:
                best_radius = r
                best_positions = current_pos.copy()
                
        except Exception:
            continue

    # Phase 2: Perturbation search around the best solution found
    if best_positions is not None and remaining() > 30:
        for perturb_seed in range(10):
            if remaining() < 15:
                break
            
            sigma = 0.01 + 0.005 * perturb_seed
            perturbed = make_perturbed(best_positions, seed=1000 + perturb_seed, sigma=sigma)
            
            # Quick smooth optimization
            pos_opt = local_optimize_smooth(perturbed, beta=50.0, maxiter=100, time_budget=5)
            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=50, time_budget=5)
            
            r = compute_radius(pos_opt)
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt.copy()

    # Phase 3: High-precision polish of the absolute best
    if best_positions is not None and remaining() > 20:
        # Use very high beta for a tight approximation of the min
        polished = local_optimize_smooth(best_positions, beta=2000.0, maxiter=200, time_budget=10)
        if remaining() > 10:
            polished = local_optimize_direct(polished, maxiter=200, time_budget=10)
        
        r_polished = compute_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished.copy()

    # Final clip to ensure validity
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    # Ensure we return a valid array
    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions

if __name__ == "__main__":
    res = construct()
    print(f"Radius: {compute_radius(res)}")
