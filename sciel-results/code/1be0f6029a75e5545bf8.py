"""
Combined sphere packing: vectorized soft-min surrogate (from inspiration)
+ multi-start, annealing, direct/subgradient/coordinate polish (from current).
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import scipy.special as sp

N = 50
DEADLINE_SECONDS = 145.0

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def min_radius(positions):
        """Compute the largest common radius for positions of shape (N,3) in [-1/2,1/2]^3."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(positions)):
            return 0.0
        if positions.shape != (N, 3):
            return 0.0
        if np.any(np.abs(positions) > 0.5 + 1e-9):
            return 0.0
        # Pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    def objective_direct(x):
        """Negative radius for minimization."""
        return -min_radius(x.reshape(-1, 3))

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
        log_sum_exp = sp.logsumexp(-beta * g_all)
        val = -log_sum_exp / beta
        
        # 3. Gradient Calculation
        weights = np.exp(-beta * g_all - log_sum_exp)
        
        grad_c = np.zeros_like(c)
        
        # Pairwise gradient - fully vectorized using np.add.at
        n_pairs = len(d_pairs)
        safe_d = np.maximum(d_pairs, 1e-12)
        factors = weights[:n_pairs] / (4.0 * safe_d)
        factors = np.where(d_pairs > 1e-12, factors, 0.0)
        
        weighted_diffs = factors[:, np.newaxis] * diff
        
        # Scatter: add to index i, subtract from index j
        np.add.at(grad_c, tri_i, weighted_diffs)
        np.add.at(grad_c, tri_j, -weighted_diffs)
        
        # Wall part
        w_start = n_pairs
        grad_c[:, 0] -= weights[w_start : w_start+N]
        grad_c[:, 0] += weights[w_start+N : w_start+2*N]
        grad_c[:, 1] -= weights[w_start+2*N : w_start+3*N]
        grad_c[:, 1] += weights[w_start+3*N : w_start+4*N]
        grad_c[:, 2] -= weights[w_start+4*N : w_start+5*N]
        grad_c[:, 2] += weights[w_start+5*N : w_start+6*N]
        
        # We want to MINIMIZE -soft_min.
        return -val, -grad_c.flatten()

    def local_optimize_smooth(c_init, beta, maxiter, time_budget=5):
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
            res = minimize(
                obj_flat, c_init.flatten(), jac=grad_flat, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def local_optimize_direct(c_init, maxiter, time_budget=5):
        """Run L-BFGS-B on the exact non-smooth objective."""
        if remaining() < time_budget:
            return c_init
            
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        
        try:
            res = minimize(
                objective_direct, c_init.flatten(), method='L-BFGS-B',
                bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            return np.clip(res.x.reshape(N, 3), -0.499999, 0.499999)
        except Exception:
            return c_init

    def subgradient_ascent(positions, maxiter=200, lr_start=0.01, lr_end=0.0001):
        """Subgradient ascent on the min-radius function with adaptive step size."""
        if remaining() < 5:
            return positions
        pos = positions.copy()
        best_pos = pos.copy()
        best_r = min_radius(pos)
        
        for it in range(maxiter):
            if remaining() < 3:
                break
            # Adaptive learning rate
            progress = it / max(1, maxiter - 1)
            lr = lr_start * (lr_end / lr_start) ** progress
            
            # Compute subgradient
            diff = pos[:, None, :] - pos[None, :, :]
            d2 = np.einsum('ijk,ijk->ij', diff, diff)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            walls = 0.5 - np.abs(pos)
            
            min_pair_val = np.min(d) / 2.0
            min_wall_val = np.min(walls)
            
            grad = np.zeros_like(pos)
            
            if min_pair_val < min_wall_val:
                min_pair_idx = np.argmin(d)
                i, j = np.unravel_index(min_pair_idx, (N, N))
                diff_ij = pos[i] - pos[j]
                d_ij = np.linalg.norm(diff_ij)
                if d_ij > 1e-12:
                    unit_vec = diff_ij / d_ij
                    grad[i] += 0.5 * unit_vec
                    grad[j] -= 0.5 * unit_vec
            else:
                min_wall_idx = np.argmin(walls)
                i, dim = np.unravel_index(min_wall_idx, (N, 3))
                if pos[i, dim] > 0:
                    grad[i, dim] -= 1.0
                else:
                    grad[i, dim] += 1.0
            
            r = min(min_pair_val, min_wall_val)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
            
            # Update
            pos = pos + lr * grad
            pos = np.clip(pos, -0.499999, 0.499999)
            
            if it > 20 and np.linalg.norm(grad) < 1e-8:
                break
        
        return best_pos

    def coordinate_descent(positions, maxiter=80):
        """Refine via coordinate descent on the direct objective."""
        if remaining() < 3:
            return positions
        pos = positions.copy()
        step = 0.008
        for it in range(maxiter):
            if remaining() < 2:
                break
            improved = False
            for i in range(N):
                for dim in range(3):
                    old = pos[i, dim]
                    pos[i, dim] = min(0.499999, old + step)
                    r_pos = min_radius(pos)
                    pos[i, dim] = max(-0.499999, old - step)
                    r_neg = min_radius(pos)
                    pos[i, dim] = old
                    r_cur = min_radius(pos)
                    if r_pos > r_cur + 1e-13:
                        pos[i, dim] = min(0.499999, old + step)
                        improved = True
                    elif r_neg > r_cur + 1e-13:
                        pos[i, dim] = max(-0.499999, old - step)
                        improved = True
            if not improved:
                step *= 0.85
                if step < 1e-6:
                    break
        return pos

    def make_bcc_lattice():
        """Generate 50 points in a BCC-like configuration centered at origin."""
        int_pts = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    int_pts.append([x, y, z])
        body_pts = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    body_pts.append([x + 0.5, y + 0.5, z + 0.5])
        
        all_pts = np.array(int_pts + body_pts)
        center = np.array([1.5, 1.5, 1.5])
        all_pts = all_pts - center
        
        dists = np.linalg.norm(all_pts, axis=1)
        idx = np.argsort(dists)
        selected = all_pts[idx[:N]]
        
        max_norm = np.max(np.linalg.norm(selected, axis=1))
        if max_norm > 0:
            scale = 0.45 / max_norm
            selected = selected * scale
            
        return selected

    def make_grid_3d(n_per_axis=4):
        """Generate a 3D grid of points, selecting the N closest to center."""
        coords = np.linspace(-0.45, 0.45, n_per_axis)
        grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
        dists = np.linalg.norm(grid, axis=1)
        idx = np.argsort(dists)
        return grid[idx[:N]]

    def make_fcc_like():
        """FCC-inspired initialization."""
        a = 0.35
        positions = []
        for iz in range(4):
            for iy in range(4):
                for ix in range(4):
                    x = (ix - 1.5) * a
                    y = (iy - 1.5) * a
                    z = (iz - 1.5) * a
                    positions.append([x, y, z])
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
        mask = np.all(np.abs(positions) <= 0.48, axis=1)
        positions = positions[mask]
        
        if len(positions) > N:
            from scipy.spatial.distance import cdist
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            min_dists = np.min(dists, axis=1)
            idx = np.argsort(min_dists)[::-1][:N]
            positions = positions[idx]
        
        if len(positions) < N:
            while len(positions) < N:
                positions = np.vstack([positions, np.array([0.0, 0.0, 0.0])])
        
        return positions[:N]

    def make_spectral_init(seed=0):
        """Spectral initialization: start from grid, apply random rotations and perturbations."""
        rng = np.random.RandomState(seed)
        base = make_grid_3d(4)
        angle = rng.uniform(0, 2*np.pi)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([
            [c + axis[0]**2*(1-c), axis[0]*axis[1]*(1-c) - axis[2]*s, axis[0]*axis[2]*(1-c) + axis[1]*s],
            [axis[1]*axis[0]*(1-c) + axis[2]*s, c + axis[1]**2*(1-c), axis[1]*axis[2]*(1-c) - axis[0]*s],
            [axis[2]*axis[0]*(1-c) - axis[1]*s, axis[2]*axis[1]*(1-c) + axis[0]*s, c + axis[2]**2*(1-c)]
        ])
        rotated = base @ R.T
        perturbed = rotated + rng.normal(0, 0.02, size=rotated.shape)
        return np.clip(perturbed, -0.45, 0.45)

    def make_random(seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    best_positions = None
    best_radius = -1.0

    def update_best(pos):
        nonlocal best_positions, best_radius
        r = min_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

    bounds = [(-0.499999, 0.499999)] * (N * 3)

    # Phase 1: Differential evolution global search
    if remaining() > 50:
        try:
            res = differential_evolution(objective_direct, bounds, method='de',
                                         popsize=15, maxiter=25, seed=1,
                                         tol=1e-10, mutation=(0.4, 1.0), recombination=0.7,
                                         polish=False, init='latinhypercube')
            de_pos = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            update_best(de_pos)
        except Exception:
            pass

    # Phase 2: Multi-stage local optimization from multiple good starts
    initializations = [
        ("bcc", make_bcc_lattice),
        ("grid", make_grid_3d),
        ("fcc", make_fcc_like),
        ("spec0", lambda: make_spectral_init(0)),
        ("spec1", lambda: make_spectral_init(42)),
        ("rand0", lambda: make_random(0)),
    ]

    for name, init_func in initializations:
        if remaining() < 15:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)

            # Stage A: smooth surrogate with aggressive annealing
            if remaining() > 30:
                pos_opt = local_optimize_smooth(positions, beta=5.0, maxiter=200, time_budget=10)
                pos_opt = local_optimize_smooth(pos_opt, beta=20.0, maxiter=200, time_budget=10)
                pos_opt = local_optimize_smooth(pos_opt, beta=50.0, maxiter=200, time_budget=10)
            elif remaining() > 15:
                pos_opt = local_optimize_smooth(positions, beta=10.0, maxiter=150, time_budget=10)
                pos_opt = local_optimize_smooth(pos_opt, beta=40.0, maxiter=150, time_budget=10)
            else:
                pos_opt = positions

            # Stage B: direct objective for sharpness
            if remaining() > 15:
                pos_opt = local_optimize_direct(pos_opt, maxiter=300, time_budget=10)

            # Stage C: subgradient ascent for finer tuning
            if remaining() > 10:
                pos_opt = subgradient_ascent(pos_opt, maxiter=150, lr_start=0.005, lr_end=0.0001)

            # Stage D: coordinate descent polish
            if remaining() > 6:
                pos_opt = coordinate_descent(pos_opt, maxiter=30)

            # Stage E: high-beta polish if time permits
            if remaining() > 12:
                pos_opt = local_optimize_smooth(pos_opt, beta=80.0, maxiter=200, time_budget=10)

            update_best(pos_opt)
        except Exception:
            continue

    # Phase 3: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 25:
        n_perturb = 6
        for k in range(n_perturb):
            if remaining() < 8:
                break
            rng = np.random.RandomState(1000 + k)
            noise_scale = 0.015 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            if remaining() > 20:
                pos_opt = local_optimize_smooth(perturbed, beta=10.0, maxiter=200, time_budget=10)
                pos_opt = local_optimize_smooth(pos_opt, beta=60.0, maxiter=200, time_budget=10)
            else:
                pos_opt = perturbed

            if remaining() > 12:
                pos_opt = local_optimize_direct(pos_opt, maxiter=250, time_budget=10)

            if remaining() > 8:
                pos_opt = subgradient_ascent(pos_opt, maxiter=120, lr_start=0.003, lr_end=0.00005)

            if remaining() > 5:
                pos_opt = coordinate_descent(pos_opt, maxiter=25)

            if remaining() > 10:
                pos_opt = local_optimize_smooth(pos_opt, beta=100.0, maxiter=150, time_budget=10)

            update_best(pos_opt)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        best_positions = make_bcc_lattice()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    # Final high-precision polish if time remains
    if remaining() > 20:
        try:
            polished = local_optimize_smooth(best_positions, beta=150.0, maxiter=400, time_budget=15)
            polished = local_optimize_direct(polished, maxiter=300, time_budget=10)
            polished = subgradient_ascent(polished, maxiter=200, lr_start=0.002, lr_end=0.00001)
            polished = coordinate_descent(polished, maxiter=50)
            if min_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass
    elif remaining() > 10:
        try:
            polished = local_optimize_direct(best_positions, maxiter=200, time_budget=10)
            polished = coordinate_descent(polished, maxiter=30)
            if min_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass

    return best_positions
