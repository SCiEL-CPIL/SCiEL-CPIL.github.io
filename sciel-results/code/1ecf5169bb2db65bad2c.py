import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_radius(c):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    def velocity_verlet(centers, velocities, dt, target_r, damping=0.99, iterations=5000, time_limit=None):
        """
        Velocity-Verlet integration with damping to maximize minimum radius.
        Forces: repulsion when spheres are too close, attraction to center when hitting walls.
        """
        centers = centers.copy().astype(np.float64)
        velocities = velocities.copy().astype(np.float64)
        n = len(centers)
        
        # Precompute pair indices for efficiency
        iu = np.triu_indices(n, k=1)
        pairs_i, pairs_j = iu[0], iu[1]
        n_pairs = len(pairs_i)
        
        def compute_forces(c):
            """Compute forces on all particles."""
            forces = np.zeros_like(c)
            
            # Wall forces: push away from walls
            for k in range(3):
                d = 0.5 - np.abs(c[:, k])
                wall_margin = target_r
                mask = d < wall_margin
                if np.any(mask):
                    # Strong repulsion from walls
                    viol = (wall_margin - d[mask]) ** 2
                    dirs = -np.sign(c[mask, k])
                    dirs[dirs == 0] = 1.0
                    forces[mask, k] += 10.0 * viol * dirs
            
            # Pair forces: repulsion when too close
            diff = c[pairs_i] - c[pairs_j]
            dists_sq = np.sum(diff ** 2, axis=1)
            dists = np.sqrt(dists_sq + 1e-20)
            min_dist = 2 * target_r
            
            # Only apply force when closer than min_dist
            mask = dists < min_dist
            if np.any(mask):
                # Spring force: F = k * (min_dist - d)
                k_spring = 50.0
                f_mags = k_spring * (min_dist - dists[mask])
                # Normalize direction
                dir_vecs = diff[mask] / (dists[mask] + 1e-20)[:, np.newaxis]
                f_vecs = f_mags[:, np.newaxis] * dir_vecs
                
                # Accumulate forces
                np.add.at(forces, pairs_i[mask], f_vecs)
                np.add.at(forces, pairs_j[mask], -f_vecs)
            
            return forces
        
        # Initial forces
        forces = compute_forces(centers)
        
        best_r = compute_radius(centers)
        best_centers = centers.copy()
        
        for it in range(iterations):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Velocity Verlet step
            # v_half = v + 0.5 * dt * F
            velocities += 0.5 * dt * forces
            
            # x_new = x + dt * v_half
            centers += dt * velocities
            
            # Clip to cube (with small margin)
            centers = np.clip(centers, -0.4999, 0.4999)
            
            # Compute new forces
            forces_new = compute_forces(centers)
            
            # v_new = v_half + 0.5 * dt * F_new
            velocities += 0.5 * dt * forces_new
            
            # Apply damping
            velocities *= damping
            
            # Update forces
            forces = forces_new
            
            # Check improvement every 100 iterations
            if (it + 1) % 100 == 0:
                r = compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.05, cooling=0.999):
        """
        Hybrid Simulated Annealing and Force-Directed Refinement.
        Uses SA to jump between basins and force-directed method to settle.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        it = 0
        
        while time.time() < time_limit and temp > 1e-5:
            it += 1
            # Random Perturbation (SA Move)
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.5
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            # Metropolis Criterion
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            # Local Refinement every 5 iterations
            if it % 5 == 0 and time.time() < time_limit - 1:
                # Use velocity-Verlet for local refinement
                target_r = current_r + 0.002
                vel = np.zeros_like(centers)
                dt = 0.01
                iters = 2000
                refine_limit = min(time.time() + 0.5, time_limit)
                refined_centers, refined_r = velocity_verlet(centers, vel, dt, target_r, damping=0.98, iterations=iters, time_limit=refine_limit)
                
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def lbfgs_polish(centers, time_limit, max_iter=300):
        """
        L-BFGS-B polish using a smooth surrogate objective.
        Maximizes a soft-min of all constraints.
        """
        centers = centers.copy()
        x0 = centers.ravel()
        
        def objective_and_grad(x):
            if time.time() > time_limit:
                return 1e10, np.zeros_like(x)
            c = x.reshape(N, 3)
            
            # Wall distances
            wall_dists = 0.5 - np.abs(c)
            
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            pair_dists = dists / 2.0
            
            # Soft-min via log-sum-exp
            beta = 80.0
            neg_wall = -wall_dists.ravel()
            iu = np.triu_indices(N, k=1)
            neg_pair = -pair_dists[iu]
            all_neg = np.concatenate([neg_wall, neg_pair])
            
            max_neg = np.max(all_neg)
            logsum = max_neg + beta * np.log(np.sum(np.exp((all_neg - max_neg) / beta)))
            soft_min_val = -logsum / beta
            
            # Gradient
            weights = np.exp((all_neg - max_neg) / beta)
            weights /= np.sum(weights)
            
            grad = np.zeros_like(c)
            
            # Wall gradient
            w_wall = weights[:300].reshape(N, 3)
            grad -= w_wall * np.sign(c)
            
            # Pair gradient
            w_pair = weights[300:]
            for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
                d = dists[i, j]
                if d > 1e-12:
                    direction = (c[i] - c[j]) / d
                    grad[i] += w_pair[idx] * direction / 2.0
                    grad[j] -= w_pair[idx] * direction / 2.0
            
            obj_val = -soft_min_val
            grad_obj = -grad
            
            return obj_val, grad_obj.ravel()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < 0:
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > compute_radius(centers):
                    return new_centers, new_r
        except Exception:
            pass
        
        return centers, compute_radius(centers)

    # --- Main Execution Loop ---
    
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # Generate Initial Configurations
    initializations = []
    
    # Config 1: 5x5x4 Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    for jitter in [0.0, 0.01, 0.02, 0.05]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
        
    # Config 2: FCC-like Lattice
    a = 0.35
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.49, axis=1)
    fcc_array = fcc_array[mask]
    
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
            
    # Config 3: Random Uniform
    for _ in range(4):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)

    # Optimization Loop
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
            
        # Dynamic time allocation
        r_init = compute_radius(init_centers)
        if r_init > 0.10:
            t_alloc = min(remaining * 0.4, 15.0)
        elif r_init > 0.08:
            t_alloc = min(remaining * 0.3, 12.0)
        else:
            t_alloc = min(remaining * 0.2, 8.0)
            
        t_limit = time.time() + t_alloc
        
        # Phase A: Annealing with velocity-Verlet refinement
        anneal_time = t_alloc * 0.6
        anneal_limit = time.time() + anneal_time
        
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.06, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
        
        # Phase B: L-BFGS polish on the best found so far
        if time.time() < deadline - 15 and best_centers_global is not None:
            polish_limit = min(time.time() + 5.0, deadline - 15)
            pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, max_iter=200)
            
            if pol_r > best_radius_global:
                best_radius_global = pol_r
                best_centers_global = pol_centers

    # Final intensive polish with remaining time
    if best_centers_global is not None and time.time() < deadline - 5:
        # Run L-BFGS polish with more iterations
        polish_limit = deadline - 5
        pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, max_iter=500)
        
        if pol_r > best_radius_global:
            best_radius_global = pol_r
            best_centers_global = pol_centers
        
        # One more velocity-Verlet pass if time permits
        if time.time() < deadline - 3:
            target_r = best_radius_global + 0.001
            vel = np.zeros_like(best_centers_global)
            dt = 0.005
            iters = 10000
            refine_limit = deadline - 3
            fd_centers, fd_r = velocity_verlet(best_centers_global, vel, dt, target_r, damping=0.995, iterations=iters, time_limit=refine_limit)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
    
    # Final Safety Check
    if best_centers_global is None:
        return grid_centers
        
    # Ensure validity
    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    
    return best_centers_global

if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
