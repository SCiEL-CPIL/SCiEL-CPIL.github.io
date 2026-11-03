"""Bootstrap spawn: model-and-optimize approach for 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# Pre-compute pair indices for efficient gradient computation
_PAIR_I = np.array([i for i in range(N) for j in range(i+1, N)])
_PAIR_J = np.array([j for i in range(N) for j in range(i+1, N)])
_PAIR_COUNT = len(_PAIR_I)

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE_SECONDS - elapsed()
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape[0] != N or positions.shape[1] != 3:
            return 0.0
        # Check bounds
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        # Pairwise distances
        try:
            from scipy.spatial.distance import pdist
            dists = pdist(positions, 'euclidean')
        except ImportError:
            dists = []
            for i in range(positions.shape[0]):
                for j in range(i+1, positions.shape[0]):
                    d = np.linalg.norm(positions[i] - positions[j])
                    dists.append(d)
            dists = np.array(dists)
        
        if len(dists) > 0:
            min_dist = np.min(dists)
        else:
            min_dist = 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)
    
    def _pairwise_and_wall(positions):
        """Return (pairwise_dist_matrix_diag_excluded, wall_distances) for smooth objectives."""
        from scipy.spatial.distance import pdist
        dists = pdist(positions, 'euclidean')
        walls = 0.5 - np.abs(positions)
        return dists, walls
    
    def smooth_surrogate(x_flat, beta):
        """
        Smooth surrogate: -soft_min of (pairwise_dists/2, wall_dists).
        soft_min of values v_i = -1/beta * log(sum(exp(-beta * v_i)))
        We minimize the negative, i.e. maximize soft_min.
        So objective = -soft_min = 1/beta * log(sum(exp(-beta * v_i)))
        """
        positions = x_flat.reshape(-1, 3)
        dists, walls = _pairwise_and_wall(positions)
        # All "radius candidates": pairwise/2 and wall distances
        pair_r = dists / 2.0
        all_vals = np.concatenate([pair_r, walls.ravel()])
        # soft_min = -1/beta * logsumexp(-beta * all_vals)
        # objective = -soft_min = 1/beta * logsumexp(-beta * all_vals)
        # Use stable logsumexp
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        logsumexp = max_val + np.log(np.sum(np.exp(neg_beta_vals - max_val)))
        return logsumexp / beta
    
    def smooth_grad(x_flat, beta):
        """Gradient of the smooth surrogate (vectorized)."""
        positions = x_flat.reshape(-1, 3)
        dists, walls = _pairwise_and_wall(positions)
        pair_r = dists / 2.0
        all_vals = np.concatenate([pair_r, walls.ravel()])
        
        neg_beta_vals = -beta * all_vals
        max_val = np.max(neg_beta_vals)
        exp_vals = np.exp(neg_beta_vals - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # softmax weights
        
        # d/dv_i [1/beta * logsumexp(-beta * v)] = -weights_i
        grad_all = -weights
        
        # Now compute gradient w.r.t. positions
        grad_pos = np.zeros_like(positions)
        
        n_pairs = len(pair_r)
        
        # Compute differences and distances for all pairs using pre-computed indices
        diffs = positions[_PAIR_I] - positions[_PAIR_J]  # shape (n_pairs, 3)
        d_safe = np.maximum(dists, 1e-12)
        
        # For each pair, contribution to grad: w * diff / (2*d) for pos[i], -w * diff / (2*d) for pos[j]
        # grad_all[:n_pairs] are the weights for pairwise terms
        pair_weights = grad_all[:n_pairs]  # shape (n_pairs,)
        
        # Compute per-pair gradient factor: w / (2*d)
        pair_factors = pair_weights / (2.0 * d_safe)  # shape (n_pairs,)
        
        # For each pair, add factor * diff to pos[i] and subtract from pos[j]
        # Use np.add.at for accumulation
        pair_grads = pair_factors[:, np.newaxis] * diffs  # shape (n_pairs, 3)
        
        np.add.at(grad_pos, _PAIR_I, pair_grads)
        np.add.at(grad_pos, _PAIR_J, -pair_grads)
        
        # Wall part: wall_k = 0.5 - |pos[k//3, k%3]|
        # d(wall_k)/d(pos) = -sign(pos[k//3, k%3]) for the relevant coordinate
        wall_grad = grad_all[n_pairs:]
        n_walls = len(wall_grad)
        if n_walls > 0:
            wall_indices = np.arange(n_walls)
            wall_point_idx = wall_indices // 3
            wall_coord_idx = wall_indices % 3
            pos_vals = positions[wall_point_idx, wall_coord_idx]
            signs = np.sign(pos_vals)  # 0 at exactly 0
            wall_contrib = wall_grad * (-signs)
            np.add.at(grad_pos, (wall_point_idx, wall_coord_idx), wall_contrib)
        
        return grad_pos.ravel()
    
    def make_initial_grid():
        """Create a reasonable initial grid configuration."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def make_bcc_lattice():
        """Create a body-centered cubic inspired arrangement."""
        positions = []
        a = 0.5
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * (2*a/3)
                    y = -a + (iy + 0.5) * (2*a/3)
                    z = -a + (iz + 0.5) * (2*a/3)
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + (2*a/3)/2, y + (2*a/3)/2, z + (2*a/3)/2])
        
        positions = np.array(positions[:N])
        return positions
    
    def make_random(seed=None):
        """Create random initial positions."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    def local_optimize_smooth(positions, beta=50.0, maxiter=500, ftol=1e-14):
        """Run local optimization using smooth surrogate with L-BFGS-B."""
        from scipy.optimize import minimize
        
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return smooth_surrogate(x, beta)
        
        def grad_flat(x):
            return smooth_grad(x, beta)
        
        try:
            res = minimize(obj_flat, positions.flatten(), jac=grad_flat, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions
    
    def local_optimize_direct(positions, maxiter=300, ftol=1e-13):
        """Run local optimization on the raw min-radius objective with L-BFGS-B."""
        from scipy.optimize import minimize
        
        if remaining() < 5:
            return positions
        
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        def obj_flat(x):
            return -compute_min_radius(x.reshape(-1, 3))
        
        try:
            res = minimize(obj_flat, positions.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': maxiter, 'ftol': ftol})
            pos_opt = res.x.reshape(-1, 3)
            pos_opt = np.clip(pos_opt, -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions
    
    def polish_pass(positions, beta_list=(100.0, 200.0, 500.0, 1000.0), maxiter_each=200):
        """
        High-precision polish: run smooth optimization with increasing beta,
        then a final direct min-radius optimization.
        """
        pos = positions.copy()
        
        for beta in beta_list:
            if remaining() < 5:
                break
            pos = local_optimize_smooth(pos, beta=beta, maxiter=maxiter_each)
        
        # Final direct optimization for exact min
        if remaining() > 5:
            pos = local_optimize_direct(pos, maxiter=200)
        
        return pos
    
    def dynamics_pass(positions, dt=0.0005, n_steps=5000, damping=0.98, repulsion_strength=0.05):
        """
        Velocity-Verlet dynamics with repulsive forces to shake out of local minima.
        Uses inverse-square repulsion between spheres and wall repulsion.
        """
        if remaining() < 10:
            return positions
        
        pos = positions.copy()
        vel = np.zeros_like(pos)
        
        for step in range(n_steps):
            if remaining() < 5:
                break
            
            # Compute forces
            forces = np.zeros_like(pos)
            
            # Pairwise repulsion: F = strength / d^2 * direction
            diffs = pos[_PAIR_I] - pos[_PAIR_J]
            dists = np.linalg.norm(diffs, axis=1)
            dists_safe = np.maximum(dists, 1e-6)
            
            # Force magnitude: strength / d^2
            force_mag = repulsion_strength / (dists_safe ** 2)
            # Force direction: normalized diff
            force_dir = diffs / dists_safe[:, np.newaxis]
            
            pair_forces = force_mag[:, np.newaxis] * force_dir
            np.add.at(forces, _PAIR_I, pair_forces)
            np.add.at(forces, _PAIR_J, -pair_forces)
            
            # Wall repulsion: push away from walls
            for dim in range(3):
                # Left wall: x < 0
                near_left = pos[:, dim] < 0
                if np.any(near_left):
                    forces[near_left, dim] += repulsion_strength * 10.0
                # Right wall: x > 0
                near_right = pos[:, dim] > 0
                if np.any(near_right):
                    forces[near_right, dim] -= repulsion_strength * 10.0
            
            # Velocity-Verlet integration
            vel = vel * damping + forces * dt
            pos = pos + vel * dt
            
            # Clip to cube
            pos = np.clip(pos, -0.499999, 0.499999)
            
            # Zero velocity for spheres at walls to prevent sticking
            for dim in range(3):
                at_left = np.abs(pos[:, dim] + 0.499999) < 1e-6
                at_right = np.abs(pos[:, dim] - 0.499999) < 1e-6
                vel[at_left, dim] = 0.0
                vel[at_right, dim] = 0.0
        
        return pos
    
    best_positions = None
    best_radius = -1.0
    
    # Try multiple initializations
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]
    
    for name, init_func in initializations:
        if remaining() < 15:
            break
        
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)
            
            # Smooth optimization with moderate beta
            if remaining() > 25:
                pos_opt = local_optimize_smooth(positions, beta=30.0, maxiter=300)
            else:
                pos_opt = positions
            
            # Quick direct refinement
            if remaining() > 15:
                pos_opt = local_optimize_direct(pos_opt, maxiter=100)
            
            r = compute_min_radius(pos_opt)
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt.copy()
        except Exception:
            continue
    
    # Perturbations of the best solution
    if best_positions is not None and remaining() > 20:
        for perturb_seed in range(6):
            if remaining() < 10:
                break
            
            np.random.seed(1000 + perturb_seed)
            perturbed = best_positions + np.random.normal(0, 0.012, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)
            
            pos_opt = local_optimize_smooth(perturbed, beta=40.0, maxiter=200)
            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=100)
            
            r = compute_min_radius(pos_opt)
            if r > best_radius:
                best_radius = r
                best_positions = pos_opt.copy()
    
    # Dynamics-based escape from local minima
    if best_positions is not None and remaining() > 30:
        for dyn_trial in range(3):
            if remaining() < 15:
                break
            
            # Add small random perturbation before dynamics
            np.random.seed(2000 + dyn_trial)
            dyn_init = best_positions + np.random.normal(0, 0.008, size=best_positions.shape)
            dyn_init = np.clip(dyn_init, -0.499999, 0.499999)
            
            # Run dynamics to shake out
            dyn_pos = dynamics_pass(dyn_init, dt=0.0005, n_steps=3000, damping=0.97, repulsion_strength=0.03)
            
            # Polish the dynamics result
            if remaining() > 10:
                dyn_pos = local_optimize_smooth(dyn_pos, beta=50.0, maxiter=150)
                if remaining() > 5:
                    dyn_pos = local_optimize_direct(dyn_pos, maxiter=100)
            
            r_dyn = compute_min_radius(dyn_pos)
            if r_dyn > best_radius:
                best_radius = r_dyn
                best_positions = dyn_pos.copy()
    
    # Final high-precision polish of the best solution
    if best_positions is not None and remaining() > 15:
        polished = polish_pass(best_positions, beta_list=(100.0, 300.0, 800.0, 1500.0), maxiter_each=150)
        r_polished = compute_min_radius(polished)
        if r_polished > best_radius:
            best_radius = r_polished
            best_positions = polished.copy()
    
    # One more dynamics + polish cycle if time permits
    if best_positions is not None and remaining() > 20:
        np.random.seed(3000)
        final_dyn_init = best_positions + np.random.normal(0, 0.005, size=best_positions.shape)
        final_dyn_init = np.clip(final_dyn_init, -0.499999, 0.499999)
        
        final_dyn_pos = dynamics_pass(final_dyn_init, dt=0.0004, n_steps=2000, damping=0.96, repulsion_strength=0.02)
        
        if remaining() > 10:
            final_dyn_pos = local_optimize_smooth(final_dyn_pos, beta=100.0, maxiter=100)
            if remaining() > 5:
                final_dyn_pos = local_optimize_direct(final_dyn_pos, maxiter=80)
        
        r_final = compute_min_radius(final_dyn_pos)
        if r_final > best_radius:
            best_radius = r_final
            best_positions = final_dyn_pos.copy()
    
    # Final clip to ensure validity
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    # Ensure we return a valid array
    if best_positions is None:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    
    return best_positions
