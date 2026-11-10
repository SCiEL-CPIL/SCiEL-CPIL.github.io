import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    def calc_score(x):
        # Calculate the actual min radius
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(np.min(wall_dists, axis=1))
        
        return min(min_pair / 2.0, min_wall)
    
    def objective_and_grad(x_flat, beta):
        x = x_flat.reshape(N, 3)
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists = np.sqrt(dists_sq + 1e-15)
        
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(x)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        all_dists = np.concatenate([pair_dists, min_wall_dists])
        
        # Log-sum-exp trick
        vals = -beta * all_dists
        max_val = np.max(vals)
        log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        
        obj = -log_sum_exp / beta
        
        # Gradient
        exp_vals = np.exp(vals - max_val)
        S = np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Pair distances gradient
        pair_exp = exp_vals[:N*(N-1)]
        W = np.zeros((N, N))
        W[mask] = pair_exp / S
        
        unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-15)
        grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall distances gradient
        wall_exp = exp_vals[N*(N-1):]
        weights_wall = wall_exp / S
        
        grad_wall = np.zeros_like(x)
        min_dims = np.argmin(wall_dists, axis=1)
        for i in range(N):
            k = min_dims[i]
            grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
        
        grad = grad_pair + grad_wall
        
        return obj, grad

    def run_optimization(init_x, beta, maxiter=500):
        x0 = init_x.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        def obj_wrapper(x_flat):
            return objective_and_grad(x_flat, beta)
        
        res = minimize(
            obj_wrapper, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 3)

    def force_directed_optimize(x, r_target, max_iters=200, cooling=0.95):
        """Physics-based force-directed optimization with adaptive step sizes"""
        x = x.copy()
        step_size = 0.01
        best_x = x.copy()
        best_score = calc_score(x)
        no_improve_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            forces = np.zeros_like(x)
            
            # Pairwise repulsion (Lennard-Jones-like)
            for i in range(N):
                for j in range(i + 1, N):
                    d = x[i] - x[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-9:
                        # Add small random force if coincident
                        forces[i] += np.random.normal(0, 0.001, 3)
                        forces[j] -= np.random.normal(0, 0.001, 3)
                        continue
                    
                    # Repulsion force: strong when close, weak when far
                    # Use inverse square law for repulsion
                    if dist < 2 * r_target:
                        # Overlapping: strong repulsion
                        f = (2 * r_target - dist) / (dist * dist) * d * 10
                    elif dist < 4 * r_target:
                        # Near: moderate repulsion
                        f = (4 * r_target - dist) / (dist * dist) * d * 2
                    else:
                        # Far: weak attraction to maintain structure
                        f = -d / (dist * dist) * 0.1
                    
                    forces[i] += f
                    forces[j] -= f
            
            # Wall constraints
            for i in range(N):
                for k in range(3):
                    d_left = x[i, k] + 0.5
                    d_right = 0.5 - x[i, k]
                    
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left) * 5
                    elif d_left < 2 * r_target:
                        forces[i, k] += (2 * r_target - d_left) * 0.5
                    
                    if d_right < r_target:
                        forces[i, k] -= (r_target - d_right) * 5
                    elif d_right < 2 * r_target:
                        forces[i, k] -= (2 * r_target - d_right) * 0.5
            
            # Apply forces with adaptive step size
            x_new = x + forces * step_size
            x_new = np.clip(x_new, -0.5 + 1e-6, 0.5 - 1e-6)
            
            new_score = calc_score(x_new)
            
            if new_score > best_score:
                best_score = new_score
                best_x = x_new.copy()
                no_improve_count = 0
                step_size *= 1.1  # Increase step if improving
            else:
                no_improve_count += 1
                step_size *= 0.9  # Decrease step if not improving
            
            step_size = min(step_size, 0.05)
            step_size = max(step_size, 0.001)
            
            x = x_new
            
            # Early stopping
            if no_improve_count > 30:
                break
        
        return best_x

    def graph_based_refine(x, r_target, max_iters=100):
        """Refine using a graph of close pairs"""
        x = x.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Build graph of close pairs
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Find pairs closer than 2.5 * r_target
            close_pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    if dists[i, j] < 2.5 * r_target:
                        close_pairs.append((i, j, dists[i, j]))
            
            if not close_pairs:
                break
            
            # Compute forces only on close pairs
            forces = np.zeros_like(x)
            for i, j, dist in close_pairs:
                d = x[i] - x[j]
                if dist < 1e-9:
                    continue
                # Repulsion proportional to overlap
                if dist < 2 * r_target:
                    f = (2 * r_target - dist) / dist * d
                    forces[i] += f
                    forces[j] -= f
            
            # Wall forces
            for i in range(N):
                for k in range(3):
                    d_left = x[i, k] + 0.5
                    d_right = 0.5 - x[i, k]
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left)
                    if d_right < r_target:
                        forces[i, k] -= (r_target - d_right)
            
            # Apply with small step
            x += forces * 0.05
            x = np.clip(x, -0.5, 0.5)
        
        return x

    def polish(x, iterations=100):
        """Final high-precision polish using exact min-distance objective with subgradient"""
        x = x.copy()
        best_x = x.copy()
        best_score = calc_score(x)
        step = 0.01
        
        for it in range(iterations):
            if time.time() > deadline - 3:
                break
            
            # Compute exact min radius
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = 0.5 - np.abs(x)
            min_wall_dists = np.min(wall_dists, axis=1)
            
            # Find critical pairs and walls
            min_pair_dist = np.min(dists)
            min_wall = np.min(min_wall_dists)
            
            r = min(min_pair_dist / 2.0, min_wall)
            
            # Subgradient step
            forces = np.zeros_like(x)
            
            # If pair distance is limiting
            if min_pair_dist / 2.0 <= min_wall + 1e-12:
                for i in range(N):
                    for j in range(i + 1, N):
                        d = x[i] - x[j]
                        dist = np.linalg.norm(d)
                        if dist < min_pair_dist + 1e-8:
                            if dist > 1e-9:
                                f = d / dist * step
                                forces[i] += f
                                forces[j] -= f
            
            # If wall distance is limiting
            if min_wall <= min_pair_dist / 2.0 + 1e-12:
                for i in range(N):
                    min_dim = np.argmin(wall_dists[i])
                    k = min_dim
                    if wall_dists[i, k] < min_wall + 1e-8:
                        # Push away from wall
                        if x[i, k] > 0:
                            forces[i, k] -= step
                        else:
                            forces[i, k] += step
            
            x_new = x + forces
            x_new = np.clip(x_new, -0.5, 0.5)
            
            new_score = calc_score(x_new)
            
            if new_score > best_score:
                best_score = new_score
                best_x = x_new.copy()
                x = x_new
                step *= 1.05
            else:
                # Backtrack
                x = x_new
                step *= 0.8
            
            step = min(step, 0.02)
            step = max(step, 0.0001)
            
            # Check convergence
            if step < 0.0005 and it > 20:
                break
        
        return best_x

    def generate_fcc_init(seed):
        np.random.seed(seed)
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.25, k*0.25])
                    points.append([i*0.5+0.25, j*0.5, k*0.25])
                    points.append([i*0.5+0.25, j*0.25, k*0.5])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    def generate_bcc_init(seed):
        np.random.seed(seed)
        points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    points.append([i*0.5, j*0.5, k*0.5])
                    points.append([i*0.5+0.25, j*0.25, k*0.25])
        
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    def resolve_overlaps(x, r_target, steps=100):
        x = x.copy()
        for _ in range(steps):
            forces = np.zeros_like(x)
            # Pairwise
            for i in range(N):
                for j in range(i+1, N):
                    d = x[i] - x[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * r_target and dist > 1e-9:
                        f = (2 * r_target - dist) / dist * d
                        forces[i] += f
                        forces[j] -= f
            # Walls
            for i in range(N):
                for k in range(3):
                    d_left = x[i, k] + 0.5
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left) * (-1)
                    d_right = 0.5 - x[i, k]
                    if d_right < r_target:
                        forces[i, k] += (r_target - d_right) * (1)
            x += forces * 0.1
            x = np.clip(x, -0.5, 0.5)
        return x

    best_x = None
    best_score = -np.inf
    
    # Strategy 1: FCC Lattice + Multi-stage optimization
    for seed in range(6):
        if time.time() > deadline - 20:
            break
        
        x_init = generate_fcc_init(seed)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        # Multi-stage beta annealing
        x_opt = run_optimization(x_resolved, beta=30, maxiter=200)
        x_opt = run_optimization(x_opt, beta=80, maxiter=200)
        x_opt = run_optimization(x_opt, beta=200, maxiter=150)
        
        # Force-directed refinement
        current_r = calc_score(x_opt)
        x_opt = force_directed_optimize(x_opt, current_r, max_iters=100)
        
        # Graph-based refinement
        current_r = calc_score(x_opt)
        x_opt = graph_based_refine(x_opt, current_r, max_iters=50)
        
        # Final polish
        x_opt = polish(x_opt, iterations=80)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 2: BCC Lattice Init
    for seed in range(6, 12):
        if time.time() > deadline - 15:
            break
        
        x_init = generate_bcc_init(seed)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        # Multi-stage beta annealing
        x_opt = run_optimization(x_resolved, beta=30, maxiter=200)
        x_opt = run_optimization(x_opt, beta=80, maxiter=200)
        x_opt = run_optimization(x_opt, beta=200, maxiter=150)
        
        # Force-directed refinement
        current_r = calc_score(x_opt)
        x_opt = force_directed_optimize(x_opt, current_r, max_iters=100)
        
        # Graph-based refinement
        current_r = calc_score(x_opt)
        x_opt = graph_based_refine(x_opt, current_r, max_iters=50)
        
        # Final polish
        x_opt = polish(x_opt, iterations=80)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 3: Random + Overlap Resolution
    for seed in range(12, 18):
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        x_init = np.random.uniform(-0.4, 0.4, (N, 3))
        x_resolved = resolve_overlaps(x_init, 0.05, steps=50)
        
        # Multi-stage beta annealing
        x_opt = run_optimization(x_resolved, beta=30, maxiter=200)
        x_opt = run_optimization(x_opt, beta=80, maxiter=200)
        x_opt = run_optimization(x_opt, beta=200, maxiter=150)
        
        # Force-directed refinement
        current_r = calc_score(x_opt)
        x_opt = force_directed_optimize(x_opt, current_r, max_iters=100)
        
        # Graph-based refinement
        current_r = calc_score(x_opt)
        x_opt = graph_based_refine(x_opt, current_r, max_iters=50)
        
        # Final polish
        x_opt = polish(x_opt, iterations=80)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt

    # Strategy 4: Final polish on best solution
    if best_x is not None and time.time() < deadline - 5:
        best_x = polish(best_x, iterations=100)
        best_score = calc_score(best_x)

    if best_x is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
