"""
Improved sphere packing: BCC initialization + Smooth/Local Optimization + Subgradient
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE_SECONDS = 148.0  # Safety margin before 150s kill

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
        # Pairwise distances, vectorized
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        # Mask diagonal
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        # Wall distances
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        return float(min(min_pair, min_wall))

    def subgradient_radius(positions):
        """Compute subgradient of the min-radius function.
        Returns (radius, gradient) where gradient has shape (N,3).
        The subgradient points in a direction that decreases the minimum constraint."""
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # Wall distances
        walls = 0.5 - np.abs(positions)
        
        # Find the minimum constraint
        min_pair_val = np.min(d) / 2.0
        min_wall_val = np.min(walls)
        
        grad = np.zeros_like(positions)
        
        if min_pair_val < min_wall_val:
            # Pairwise constraint is active
            min_pair_idx = np.argmin(d)
            i, j = np.unravel_index(min_pair_idx, (N, N))
            if i == j:
                # Should not happen with diagonal masked
                min_pair_idx = np.argmin(np.where(np.eye(N, dtype=bool), np.inf, d))
                i, j = np.unravel_index(min_pair_idx, (N, N))
            diff_ij = positions[i] - positions[j]
            d_ij = np.linalg.norm(diff_ij)
            if d_ij > 1e-12:
                # Subgradient: move i away from j and j away from i
                unit_vec = diff_ij / d_ij
                grad[i] += 0.5 * unit_vec
                grad[j] -= 0.5 * unit_vec
            r = min_pair_val
        else:
            # Wall constraint is active
            min_wall_idx = np.argmin(walls)
            i, dim = np.unravel_index(min_wall_idx, (N, 3))
            # Subgradient: move away from the wall
            if positions[i, dim] > 0:
                grad[i, dim] -= 1.0  # Move in -dim direction (away from +wall)
            else:
                grad[i, dim] += 1.0  # Move in +dim direction (away from -wall)
            r = min_wall_val
        
        return r, grad

    def objective_direct(x):
        """Negative radius for minimization."""
        return -min_radius(x.reshape(-1, 3))

    def objective_smooth_neg_with_grad(x, beta=20.0):
        """Smooth surrogate with analytic gradient.
        Returns (value, gradient).
        Value is -soft_min(constraints). We minimize this.
        """
        positions = x.reshape(-1, 3)
        # Pairwise
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        
        pd_half = 0.5 * np.sqrt(d2)
        walls = 0.5 - np.abs(positions)
        
        # Flatten (exclude diagonal of pd_half)
        mask = ~np.eye(N, dtype=bool)
        pd_flat = pd_half[mask]
        walls_flat = walls.ravel()
        all_vals = np.concatenate([pd_flat, walls_flat])
        
        beta_safe = np.clip(beta, 1.0, 100.0)
        scaled = -beta_safe * all_vals
        m = np.max(scaled)
        exp_scaled = np.exp(scaled - m)
        sum_exp = np.sum(exp_scaled)
        
        if sum_exp == 0:
            return 0.0, np.zeros_like(x)
            
        weights = exp_scaled / sum_exp
        soft_min_val = -(m + np.log(sum_exp)) / beta_safe
        value = -soft_min_val
        
        grad = np.zeros_like(positions)
        
        # Process pairwise constraints
        # We need to map the flat index back to (i, j)
        # The flat array `pd_flat` corresponds to the upper triangle or all off-diagonals?
        # `mask` is ~eye, so it's all off-diagonals.
        # The order is row-major: (0,1), (0,2), ..., (0,N-1), (1,0), ...
        # This is 2 * (N*(N-1)/2) = N*(N-1) elements.
        
        # It is faster to compute gradients in the full matrix and then sum,
        # but we must be careful with the weights.
        
        # Let's reconstruct the weight matrix for pairs
        # The flat index k corresponds to a pair (i, j).
        # We can iterate or use vectorized ops.
        # Given N=50, N*(N-1) = 2450. Iteration is fine, but vectorized is better.
        
        # Let's compute the gradient contribution for each pair (i,j)
        # weight w_ij
        # grad_pos_i += w_ij * 0.5 * (c_i - c_j) / d_ij
        # grad_pos_j -= w_ij * 0.5 * (c_i - c_j) / d_ij
        
        # We have `weights` for `all_vals`.
        # The first N*(N-1) elements are pairwise.
        # The next N*3 elements are walls.
        
        n_pairs = N * (N - 1)
        pair_weights = weights[:n_pairs]
        wall_weights = weights[n_pairs:]
        
        # Create a matrix of weights from the flat array
        # The flat array from `pd_half[mask]` follows the order of `mask.ravel()`.
        # `mask` is a boolean array. `pd_half[mask]` extracts elements where mask is True.
        # This is equivalent to iterating i from 0 to N-1, j from 0 to N-1, if i!=j.
        
        # We can create a dense weight matrix W_pair (N, N)
        W_pair = np.zeros((N, N))
        # We need to map the flat index to (i, j).
        # This is tricky to do purely vectorized without knowing the exact mapping order.
        # However, `np.where(mask)` gives the indices.
        rows, cols = np.where(mask)
        W_pair[rows, cols] = pair_weights
        
        # Compute pairwise gradient
        # diff is (N, N, 3)
        # d is (N, N)
        # We want sum_j W_pair[i, j] * 0.5 * diff[i, j] / d[i, j]
        # And subtract for the j-th component?
        # Actually, the gradient of the objective with respect to c_i is:
        # - sum_j w_ij * grad_{c_i} (0.5 * ||c_i - c_j||)
        # grad_{c_i} (0.5 * ||c_i - c_j||) = 0.5 * (c_i - c_j) / ||c_i - c_j||
        
        # So grad_pos[i] = - sum_j W_pair[i, j] * 0.5 * diff[i, j] / d[i, j]
        
        # d is pd_half * 2? No, pd_half is 0.5 * d. So d = 2 * pd_half.
        # diff / d = diff / (2 * pd_half).
        # So 0.5 * diff / d = 0.25 * diff / pd_half.
        
        # Let's use d directly.
        d = 2.0 * pd_half
        # Avoid division by zero (diagonal is inf, so 1/inf = 0, but off-diagonal d > 0)
        # If d is very small, gradient is large.
        
        # Unit vectors
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_diff = diff / d[:, :, None]
            
        # Multiply by weights
        # grad_pair[i] = - sum_j W_pair[i, j] * 0.5 * unit_diff[i, j]
        # W_pair is (N, N). unit_diff is (N, N, 3).
        # We need to broadcast W_pair to (N, N, 1)
        weighted_unit = W_pair[:, :, None] * unit_diff
        # Sum over j (axis 1)
        grad_pair = np.sum(weighted_unit, axis=1)
        
        grad -= 0.5 * grad_pair
        
        # Process wall constraints
        # walls is (N, 3)
        # wall_weights is (N*3,)
        # We need to map wall_weights back to (N, 3)
        W_wall = wall_weights.reshape(N, 3)
        
        # grad_{c_ik} (0.5 - |c_ik|) = -sign(c_ik)
        # So contribution is - w_ik * (-sign(c_ik)) = w_ik * sign(c_ik)
        # Wait, objective is -soft_min.
        # grad f = - sum w_k grad a_k.
        # grad a_k = -sign(c_ik).
        # So grad f = - sum w_k (-sign(c_ik)) = sum w_k sign(c_ik).
        
        signs = np.sign(positions)
        # Handle 0 carefully, sign(0)=0, which is fine (subgradient 0).
        grad_wall = W_wall * signs
        
        grad += grad_wall
        
        # Flatten gradient
        return value, grad.ravel()

    def make_bcc_lattice():
        """Generate 50 points in a BCC-like configuration centered at origin."""
        # Generate a 3x3x3 BCC supercell
        # Integer points (corners)
        int_pts = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    int_pts.append([x, y, z])
        # Body center points
        body_pts = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    body_pts.append([x + 0.5, y + 0.5, z + 0.5])
        
        all_pts = np.array(int_pts + body_pts)
        center = np.array([1.5, 1.5, 1.5])
        all_pts = all_pts - center
        
        # Sort by distance to center to get the densest core
        dists = np.linalg.norm(all_pts, axis=1)
        idx = np.argsort(dists)
        selected = all_pts[idx[:N]]
        
        # Scale to fit within [-0.45, 0.45] to allow for sphere radius
        max_norm = np.max(np.linalg.norm(selected, axis=1))
        if max_norm > 0:
            scale = 0.45 / max_norm
            selected = selected * scale
            
        return selected

    def make_random(seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        return rng.uniform(-0.45, 0.45, size=(N, 3))

    def local_optimize_direct(positions, maxiter=400):
        """L-BFGS-B on the direct (nonsmooth) objective."""
        if remaining() < 5:
            return positions
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res = minimize(objective_direct, positions.flatten(), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-8})
            pos_opt = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return positions

    def local_optimize_smooth(positions, beta_start=5.0, beta_end=30.0, maxiter=300):
        """L-BFGS-B on a smooth surrogate, annealing beta for a sharper objective."""
        if remaining() < 5:
            return positions
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        n_beta = 4
        betas = np.linspace(beta_start, beta_end, n_beta)
        pos = positions.flatten()
        for bi, beta in enumerate(betas):
            if remaining() < 3:
                break
            per_iter = max(20, int(maxiter / n_beta))
            def obj_b(x):
                val, grad = objective_smooth_neg_with_grad(x, beta)
                return val, grad
            try:
                res = minimize(obj_b, pos, method='L-BFGS-B', jac=True,
                               bounds=bounds, options={'maxiter': per_iter, 'ftol': 1e-12, 'gtol': 1e-6})
                pos = res.x
            except Exception:
                pass
        pos_opt = np.clip(pos.reshape(-1, 3), -0.499999, 0.499999)
        return pos_opt

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
            
            r, grad = subgradient_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
            
            # Update
            pos = pos + lr * grad
            pos = np.clip(pos, -0.499999, 0.499999)
            
            # Check for convergence
            if it > 20 and np.linalg.norm(grad) < 1e-8:
                break
        
        return best_pos

    def differential_evolution_global(bounds, popsize, maxiter, seed):
        """Simple differential evolution on the direct objective."""
        if remaining() < 20:
            return None
        try:
            res = differential_evolution(objective_direct, bounds, method='de',
                                         popsize=popsize, maxiter=maxiter, seed=seed,
                                         tol=1e-10, mutation=(0.4, 1.0), recombination=0.7,
                                         polish=False, init='latinhypercube')
            return np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
        except Exception:
            return None

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
    if remaining() > 60:
        de_pos = differential_evolution_global(bounds, popsize=15, maxiter=30, seed=1)
        if de_pos is not None:
            update_best(de_pos)

    # Phase 2: Multi-stage local optimization from multiple good starts
    # BCC is the primary initialization now
    initializations = [
        ("bcc", make_bcc_lattice),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
    ]

    for name, init_func in initializations:
        if remaining() < 12:
            break
        try:
            positions = init_func()
            positions = np.clip(positions, -0.499999, 0.499999)

            # Stage A: smooth surrogate (helps escape flat regions)
            if remaining() > 25:
                pos_opt = local_optimize_smooth(positions, beta_start=5.0, beta_end=25.0, maxiter=200)
            else:
                pos_opt = positions

            # Stage B: direct objective for sharpness
            if remaining() > 12:
                pos_opt = local_optimize_direct(pos_opt, maxiter=300)

            # Stage C: subgradient ascent for finer tuning
            if remaining() > 8:
                pos_opt = subgradient_ascent(pos_opt, maxiter=150, lr_start=0.005, lr_end=0.0001)

            # Stage D: coordinate descent polish
            if remaining() > 5:
                pos_opt = coordinate_descent(pos_opt, maxiter=30)

            update_best(pos_opt)
        except Exception:
            continue

    # Phase 3: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 20:
        n_perturb = 6
        for k in range(n_perturb):
            if remaining() < 6:
                break
            rng = np.random.RandomState(1000 + k)
            noise_scale = 0.02 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            if remaining() > 18:
                pos_opt = local_optimize_smooth(perturbed, beta_start=8.0, beta_end=30.0, maxiter=150)
            else:
                pos_opt = perturbed

            if remaining() > 10:
                pos_opt = local_optimize_direct(pos_opt, maxiter=250)

            if remaining() > 7:
                pos_opt = subgradient_ascent(pos_opt, maxiter=120, lr_start=0.003, lr_end=0.00005)

            if remaining() > 4:
                pos_opt = coordinate_descent(pos_opt, maxiter=25)

            update_best(pos_opt)

    # Phase 4: If time remains, a second DE round from a tighter box around best
    if best_positions is not None and remaining() > 40:
        tight_bounds = []
        for i in range(N * 3):
            center = best_positions.flatten()[i]
            lo = max(-0.499999, center - 0.10)
            hi = min(0.499999, center + 0.10)
            tight_bounds.append((lo, hi))
        de_pos2 = differential_evolution_global(tight_bounds, popsize=12, maxiter=25, seed=99)
        if de_pos2 is not None:
            if remaining() > 15:
                de_pos2 = local_optimize_direct(de_pos2, maxiter=200)
            if remaining() > 8:
                de_pos2 = coordinate_descent(de_pos2, maxiter=30)
            update_best(de_pos2)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        best_positions = make_bcc_lattice()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    # Final high-precision polish if time remains
    if remaining() > 15:
        try:
            polished = local_optimize_direct(best_positions, maxiter=400)
            polished = subgradient_ascent(polished, maxiter=200, lr_start=0.002, lr_end=0.00001)
            polished = coordinate_descent(polished, maxiter=50)
            if min_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass

    return best_positions
