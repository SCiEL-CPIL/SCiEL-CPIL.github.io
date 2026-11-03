"""
Nesterov-accelerated gradient ascent on the exact min-distance objective for 50 spheres.
Uses k-NN active set updates and a final L-BFGS-B polish on a smooth surrogate.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 145.0  # Safety margin

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE_SECONDS - elapsed()

    def compute_radius(pos):
        """Compute the largest common radius for positions of shape (N,3)."""
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        if not np.all(np.isfinite(pos)):
            return 0.0
        if np.any(np.abs(pos) > 0.5 + 1e-9):
            return 0.0
        
        # Pairwise distances
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair_dist = np.sqrt(np.min(d2)) / 2.0
        
        # Wall distances
        walls = 0.5 - np.abs(pos)
        min_wall_dist = np.min(walls)
        
        return float(min(min_pair_dist, min_wall_dist))

    def get_active_set(pos, r_min, tol=1e-5):
        """
        Identify active constraints (pairs and walls) that are within tol of r_min.
        Returns list of (type, indices) and the subgradient vector.
        """
        grad = np.zeros_like(pos)
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)
        
        # Active pairs: d_ij / 2 <= r_min + tol
        active_pairs = d / 2.0 <= (r_min + tol)
        np.fill_diagonal(active_pairs, False)
        
        # Compute subgradient for active pairs
        # grad_i += 0.5 * (x_i - x_j) / d_ij
        # grad_j += 0.5 * (x_j - x_i) / d_ij
        # We can vectorize this:
        # For each active pair (i,j), add to grad[i] and grad[j]
        # Using scatter or loop for clarity first, then optimize if needed.
        # For N=50, a simple loop over active pairs is fine.
        active_idx = np.where(active_pairs)
        for i, j in zip(*active_idx):
            dist = d[i, j]
            if dist < 1e-8:
                # Coincident, random nudge
                grad[i] += np.random.normal(0, 1e-3, 3)
                grad[j] -= np.random.normal(0, 1e-3, 3)
            else:
                direction = (pos[i] - pos[j]) / dist
                grad[i] += 0.5 * direction
                grad[j] += 0.5 * (-direction)
        
        # Active walls: 0.5 - |x_ik| <= r_min + tol
        walls = 0.5 - np.abs(pos)
        active_walls = walls <= (r_min + tol)
        
        # Compute subgradient for active walls
        # grad_ik += -sign(x_ik) * 1.0 (since d(0.5-|x|)/dx = -sign(x))
        # We want to move AWAY from the wall, so if x>0, move in -x direction?
        # Wait, we are MAXIMIZING the radius.
        # r = min(...). To increase r, we need to increase the active constraints.
        # Constraint g(x) = 0.5 - |x|. Gradient is -sign(x).
        # If x > 0, g = 0.5 - x. dg/dx = -1.
        # To increase g, we must decrease x. So move in -x direction.
        # If x < 0, g = 0.5 - (-x) = 0.5 + x. dg/dx = 1.
        # To increase g, we must increase x. So move in +x direction.
        # So gradient direction is -sign(x).
        for i in range(N):
            for k in range(3):
                if active_walls[i, k]:
                    if pos[i, k] > 1e-8:
                        grad[i, k] -= 1.0
                    elif pos[i, k] < -1e-8:
                        grad[i, k] += 1.0
                    else:
                        grad[i, k] += np.random.choice([-1.0, 1.0])
        
        return grad

    def nesterov_ascent(pos, max_iter=500, step_size=0.005, momentum=0.9, tol=1e-5):
        """
        Nesterov-accelerated gradient ascent on the exact min-distance objective.
        """
        pos = pos.copy()
        v = np.zeros_like(pos)
        r_prev = compute_radius(pos)
        
        for it in range(max_iter):
            if remaining() < 5:
                break
                
            r_min = r_prev
            grad = get_active_set(pos, r_min, tol)
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            else:
                break # Stuck
            
            # Nesterov update
            # x_new = x + momentum * v + step_size * grad
            # v_new = grad + momentum * (x_new - x) ? 
            # Standard Nesterov:
            # y = x + momentum * v
            # grad_y = grad_at(y)
            # x_new = y - step_size * grad_y  (for minimization)
            # For maximization: x_new = y + step_size * grad_y
            # v_new = v + (x_new - x)
            
            # Simpler momentum update (Heavy Ball):
            v = momentum * v + grad
            pos_new = pos + step_size * v
            pos_new = np.clip(pos_new, -0.499999, 0.499999)
            
            r_new = compute_radius(pos_new)
            
            if r_new > r_prev:
                pos = pos_new
                r_prev = r_new
                # Decay step size slightly to converge
                step_size *= 0.999
            else:
                # Reject step, reduce step size
                step_size *= 0.9
                v *= 0.5 # Dampen momentum
                if step_size < 1e-6:
                    break

        return pos

    def smooth_objective_neg(x, beta=20.0):
        """Smooth surrogate for L-BFGS-B polish."""
        positions = x.reshape(-1, 3)
        diff = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        pd_half = 0.5 * np.sqrt(d2)
        walls = 0.5 - np.abs(positions)
        
        mask = ~np.eye(N, dtype=bool)
        pd_flat = pd_half[mask]
        walls_flat = walls.ravel()
        all_vals = np.concatenate([pd_flat, walls_flat])
        
        # Soft min
        scaled = -beta * all_vals
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        soft_min_val = -lse / beta
        return -soft_min_val

    def polish_lbfgsb(pos, maxiter=200):
        """L-BFGS-B polish on smooth surrogate."""
        if remaining() < 5:
            return pos
        bounds = [(-0.499999, 0.499999)] * (N * 3)
        try:
            res = minimize(smooth_objective_neg, pos.flatten(), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})
            pos_opt = np.clip(res.x.reshape(-1, 3), -0.499999, 0.499999)
            return pos_opt
        except Exception:
            return pos

    def make_initial_grid():
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def make_bcc_lattice():
        positions = []
        a = 0.5
        for iz in range(3):
            for iy in range(3):
                for ix in range(3):
                    x = -a + (ix + 0.5) * (2 * a / 3)
                    y = -a + (iy + 0.5) * (2 * a / 3)
                    z = -a + (iz + 0.5) * (2 * a / 3)
                    positions.append([x, y, z])
                    if len(positions) < N:
                        positions.append([x + (2 * a / 3) / 2, y + (2 * a / 3) / 2, z + (2 * a / 3) / 2])
        positions = np.array(positions[:N])
        return positions

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
        r = compute_radius(pos)
        if r > best_radius:
            best_radius = r
            best_positions = pos.copy()
        return r

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
            
            # Stage 1: Nesterov Ascent
            if remaining() > 20:
                pos_opt = nesterov_ascent(positions, max_iter=300, step_size=0.008, momentum=0.8)
            else:
                pos_opt = positions
            
            # Stage 2: L-BFGS-B Polish
            if remaining() > 10:
                pos_opt = polish_lbfgsb(pos_opt, maxiter=200)
            
            update_best(pos_opt)
        except Exception:
            continue

    # Phase 2: Perturbation refinement of the best solution
    if best_positions is not None and remaining() > 30:
        n_perturb = 5
        for k in range(n_perturb):
            if remaining() < 10:
                break
            rng = np.random.RandomState(1000 + k)
            noise_scale = 0.02 * (1.0 - k / (n_perturb + 1))
            perturbed = best_positions + rng.normal(0, noise_scale, size=best_positions.shape)
            perturbed = np.clip(perturbed, -0.499999, 0.499999)

            if remaining() > 20:
                pos_opt = nesterov_ascent(perturbed, max_iter=200, step_size=0.005, momentum=0.9)
            else:
                pos_opt = perturbed

            if remaining() > 10:
                pos_opt = polish_lbfgsb(pos_opt, maxiter=150)

            update_best(pos_opt)

    # Final validity clip
    if best_positions is not None:
        best_positions = np.clip(best_positions, -0.499999, 0.499999)
    else:
        best_positions = make_initial_grid()
        best_positions = np.clip(best_positions, -0.499999, 0.499999)

    # Final high-precision polish if time remains
    if remaining() > 10:
        try:
            polished = polish_lbfgsb(best_positions, maxiter=300)
            if compute_radius(polished) > best_radius:
                best_positions = np.clip(polished, -0.499999, 0.499999)
        except Exception:
            pass

    return best_positions
