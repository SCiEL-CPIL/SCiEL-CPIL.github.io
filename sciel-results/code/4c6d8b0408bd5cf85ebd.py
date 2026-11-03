"""
Optimized Sphere Packing: 50 equal spheres in a unit cube.
Strategy:
1. Diverse initializations (Grid, Jittered Grid, Greedy Random, BCC/FCC lattices).
2. Smooth surrogate optimization (log-sum-exp soft-min) with aggressive beta annealing.
3. Final polish with exact objective to maximize precision.
"""
import numpy as np
import time
import scipy.optimize as opt

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
        tri_i, tri_j = np.triu_indices(N, k=1)
        diff = c[tri_i] - c[tri_j]
        d_pairs = np.linalg.norm(diff, axis=1) / 2.0
        
        # Walls: 6 distances per point
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
        # Use stable logsumexp: -1/beta * log(sum(exp(-beta * g_k)))
        scaled = -beta * g_all
        max_scaled = np.max(scaled)
        if not np.isfinite(max_scaled):
            return 0.0, np.zeros_like(c_flat)
        
        # Shift for stability
        exp_scaled = np.exp(scaled - max_scaled)
        sum_exp = np.sum(exp_scaled)
        
        if sum_exp < 1e-15:
             return 0.0, np.zeros_like(c_flat)
             
        log_sum_exp = max_scaled + np.log(sum_exp)
        val = -log_sum_exp / beta
        
        # 3. Gradient Calculation
        # d/dx (-1/beta log sum exp(-beta g_k)) = -1/beta * (1/sum) * sum(-beta exp(-beta g_k) dg_k/dx)
        # = (1/sum) * sum(exp(-beta g_k) dg_k/dx)
        # Weights w_k = exp(-beta g_k) / sum_exp
        weights = exp_scaled / sum_exp
        
        grad_c = np.zeros_like(c)
        
        # Pairwise gradient
        n_pairs = len(d_pairs)
        safe_d = np.maximum(d_pairs, 1e-12)
        # dg/dx for pair (i,j) with d = ||c_i - c_j||/2
        # d(d/2)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) = diff / (4 * d_pairs) * 2? 
        # Let's re-derive carefully.
        # g_k = ||c_i - c_j|| / 2
        # dg_k / dc_i = (c_i - c_j) / (2 * ||c_i - c_j||)
        # Let diff = c_i - c_j. ||diff|| = 2 * g_k.
        # dg_k / dc_i = diff / (4 * g_k)
        # The objective is -soft_min = -val. We want to maximize soft_min, so minimize -soft_min.
        # grad_obj = - grad(soft_min)
        # grad(soft_min) = sum w_k * grad(g_k)
        # grad(-soft_min) = - sum w_k * grad(g_k)
        
        # Factor for pair k: w_k * (diff / (4 * g_k))
        factors = weights[:n_pairs] / (4.0 * safe_d)
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        
        weighted_diffs = factors[:, np.newaxis] * diff
        
        # Accumulate gradients
        # For pair (i, j):
        # grad_c[i] += w_k * (c_i - c_j) / (4 g_k)
        # grad_c[j] -= w_k * (c_i - c_j) / (4 g_k)
        
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)
        
        # Wall part
        # g_wall = 0.5 - c_x  => dg/dc_x = -1
        # g_wall = c_x + 0.5 => dg/dc_x = 1
        # ... and so on.
        
        # Indices in g_all:
        # 0..N-1: d_x_pos (0.5 - c_x) -> grad -1
        # N..2N-1: d_x_neg (c_x + 0.5) -> grad +1
        # 2N..3N-1: d_y_pos -> grad -1
        # 3N..4N-1: d_y_neg -> grad +1
        # 4N..5N-1: d_z_pos -> grad -1
        # 5N..6N-1: d_z_neg -> grad +1
        
        w_start = n_pairs
        # x-axis
        grad_c[:, 0] -= weights[w_start : w_start+N]
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        # y-axis
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        # z-axis
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # We want to MINIMIZE -soft_min.
        # The gradient calculated above is for soft_min.
        # So gradient for objective (-soft_min) is -grad_c.
        return -val, -grad_c.flatten()

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
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15}
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

    def make_jittered_grid(seed=0):
        """Jittered 4x4x4 grid to break symmetry."""
        np.random.seed(seed)
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        jitter = np.random.uniform(-0.02, 0.02, size=grid.shape)
        return (grid + jitter)[:N].copy()

    def make_bcc_lattice():
        """Body-centered cubic inspired."""
        positions = []
        # 3x3x3 grid of points, spacing 1/3
        # This gives 27 points. Need 50.
        # Let's use a denser grid or multiple layers.
        # A 5x5x5 grid has 125 points.
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        # Add BCC offsets
        base = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        offset = 1.0 / (2 * m)
        bcc_points = base + np.array([offset, offset, offset])
        # Combine and clip
        all_points = np.vstack([base, bcc_points])
        # Keep only those well inside
        mask = np.all(np.abs(all_points) < 0.45, axis=1)
        all_points = all_points[mask]
        if len(all_points) < N:
            # Pad with random
            np.random.seed(1)
            extra = np.random.uniform(-0.4, 0.4, size=(N - len(all_points), 3))
            all_points = np.vstack([all_points, extra])
        return all_points[:N]

    def make_random_greedy(seed=0):
        """Place points one by one to maximize minimum distance."""
        np.random.seed(seed)
        positions = np.zeros((N, 3))
        # First point at center
        positions[0] = [0, 0, 0]
        
        for i in range(1, N):
            # Generate many random candidates
            candidates = np.random.uniform(-0.45, 0.45, size=(1000, 3))
            # Calculate min distance to existing points for each candidate
            # And min distance to walls
            best_score = -1
            best_pos = candidates[0]
            
            # Vectorized calculation for speed
            # Distances to existing points
            diff = candidates[:, np.newaxis, :] - positions[:i][np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            min_d_to_pts = np.min(d, axis=1)
            
            # Distances to walls
            d_walls = np.min(0.5 - np.abs(candidates), axis=1)
            
            # Score is min(min_d_to_pts, d_walls)
            scores = np.minimum(min_d_to_pts, d_walls)
            
            best_idx = np.argmax(scores)
            best_pos = candidates[best_idx]
            best_score = scores[best_idx]
            
            positions[i] = best_pos
            
        return positions

    best_positions = None
    best_radius = -1.0
    
    initializations = [
        ("grid", make_initial_grid),
        ("jitter0", lambda: make_jittered_grid(0)),
        ("jitter1", lambda: make_jittered_grid(42)),
        ("bcc", make_bcc_lattice),
        ("greedy0", lambda: make_random_greedy(0)),
        ("greedy1", lambda: make_random_greedy(1)),
        ("rand0", lambda: np.random.uniform(-0.45, 0.45, size=(N, 3)) if (np.random.seed(0), True) else None),
        ("rand1", lambda: np.random.uniform(-0.45, 0.45, size=(N, 3)) if (np.random.seed(42), True) else None),
    ]
    
    # Phase 1: Global Optimization with Smooth Surrogate (Annealed Beta)
    # We iterate through initializations. If time runs out, we stop.
    # We use a list of betas.
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    
    for name, init_func in initializations:
        if remaining() < 10:
            break
            
        try:
            c = init_func()
            c = np.clip(c, -0.499999, 0.499999)
            
            # Check initial quality to decide if we should spend more time
            r_init = compute_radius(c)
            
            # Adaptive time budget per beta stage
            # If we are running low on time, reduce iterations
            for i, beta in enumerate(betas):
                if remaining() < 5:
                    break
                
                # Allocate more iterations to higher betas (more precise)
                # and fewer to lower betas (exploration)
                maxiter = 100 + 50 * i
                time_budget = 2.0
                
                c = local_optimize_smooth(c, beta, maxiter=maxiter, time_budget=time_budget)
                
                # Periodically check radius to update best
                if i % 3 == 0:
                    r_cur = compute_radius(c)
                    if r_cur > best_radius:
                        best_radius = r_cur
                        best_positions = c.copy()
            
            # Phase 2: Direct Polish
            if remaining() > 5:
                c = local_optimize_direct(c, maxiter=100, time_budget=4)
            
            r = compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_positions = c.copy()
                
        except Exception:
            continue

    # Phase 3: Perturbations of Best Solution
    if best_positions is not None and remaining() > 20:
        # Try a few random perturbations of the best solution found so far
        # This helps escape local minima if the best solution is stuck
        for perturb_seed in range(5):
            if remaining() < 8:
                break
            np.random.seed(1000 + perturb_seed)
            scale = 0.01
            c_pert = best_positions + np.random.normal(0, scale, size=best_positions.shape)
            c_pert = np.clip(c_pert, -0.499999, 0.499999)
            
            # Refine perturbed solution with high beta only
            for beta in [100.0, 500.0, 2000.0]:
                if remaining() < 4:
                    break
                c_pert = local_optimize_smooth(c_pert, beta, maxiter=100, time_budget=2)
                
            if remaining() > 4:
                c_pert = local_optimize_direct(c_pert, maxiter=50, time_budget=3)
                
            r = compute_radius(c_pert)
            if r > best_radius:
                best_radius = r
                best_positions = c_pert.copy()

    # Final high-precision Polish
    if best_positions is not None and remaining() > 10:
        # One last push with very high beta and direct method
        if remaining() > 5:
             best_positions = local_optimize_smooth(best_positions, beta=10000.0, maxiter=200, time_budget=4)
        best_positions = local_optimize_direct(best_positions, maxiter=200, time_budget=5)
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
        
        # Update best radius one last time
        r_final = compute_radius(best_positions)
        if r_final > best_radius:
            best_radius = r_final

    if best_positions is None:
        best_positions = make_initial_grid()
        
    return best_positions

# EVOLVE-BLOCK-END
