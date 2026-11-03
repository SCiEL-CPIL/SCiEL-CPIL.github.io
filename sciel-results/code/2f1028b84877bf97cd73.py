"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate used in gradient polish for better convergence.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 112.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min.
        beta controls sharpness: large beta -> hard min, small beta -> smooth average.
        """
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        # Soft-min of distances: -logsumexp(-beta * dist) / beta
        # For numerical stability, use the trick: min(a,b) ≈ -(1/beta)*log(exp(-beta*a)+exp(-beta*b))/2
        # Generalized: softmin(d_i) = -(1/beta) * log(sum(exp(-beta*d_i)))
        # But we need per-row soft-min, so:
        neg_beta_dist = -beta * dist
        # Subtract max for stability
        max_neg = np.max(neg_beta_dist, axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(neg_beta_dist - max_neg), axis=1, keepdims=True))
        soft_min_dist = -(max_neg + log_sum) / beta  # shape (N, 1)
        soft_min_dist = soft_min_dist[:, 0]  # shape (N,)

        # Also need soft-min with boundary: combine dist-to-boundary as another "neighbor"
        # Boundary distances: x, 1-x, y, 1-y -> min of these 4
        # We can treat boundary as 4 additional "circles" at infinity distance, but easier:
        # soft-min of [dist_to_neighbors..., dist_to_bound]
        # Let's just do: r = softmin(0.5*dist_to_neighbors, r_bound)
        # Actually, let's compute soft-min over all constraints for each circle:
        # For circle i: constraints are 0.5*d(i,j) for all j!=i, and r_bound[i]
        # Stack them:
        # dist matrix has inf on diagonal, so 0.5*dist has inf on diagonal
        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        # Add boundary as an extra column
        # Constraints for circle i: half_dist[i, j] for j!=i, and r_bound[i]
        # Build full constraint matrix: (N, N) where row i has all constraints
        # Replace diagonal with r_bound[i]
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row
        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            # Hexagonal lattice initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=5000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        # Cooling schedule
        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        # Repulsion force parameters
        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            # Compute current energy
            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            # Compute forces (vectorized)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Boundary repulsion (vectorized)
            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            # Pairwise repulsion (vectorized)
            dx = x[:, None] - x[None, :]  # (N, N)
            dy = y[:, None] - y[None, :]  # (N, N)
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)  # Avoid division by zero on diagonal

            # Force magnitude: repulse_k / (d^2 + eps)
            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            # Force on i from j: -f * (x_i - x_j) / d = f * (x_j - x_i) / d
            # fx[i] = sum_j f_mag[i,j] * (x_j - x_i) / d[i,j]
            # = sum_j f_mag[i,j] * (-dx[i,j]) / d[i,j]
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            # Propose move: combine force direction with random perturbation
            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            # Clip to valid range
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            # Evaluate new state
            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            # Acceptance criterion
            delta = new_val - curr_val
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x = x_new
                y = y_new
                r = r_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()

        return best_x, best_y, compute_radii(best_x, best_y), best_val

    def gradient_polish(x0, y0, use_smooth=True, beta=80.0):
        """Final gradient-based refinement using L-BFGS-B.
        Uses smooth soft-min objective for better gradient signal.
        """
        params0 = np.concatenate([x0, y0])

        bounds = [(0.01, 0.99)] * (2 * N)

        def objective(params):
            x = params[:N]
            y = params[N:2*N]
            if use_smooth:
                r = compute_radii_smooth(x, y, beta=beta)
            else:
                r = compute_radii(x, y)
            return -np.sum(r)

        best_result = None
        best_val = -np.inf

        # Try smooth objective first (better gradients)
        try:
            res = minimize(objective, params0, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = compute_radii(x_opt, y_opt)  # Use hard radii for actual value

            if check_validity(x_opt, y_opt, r_opt):
                val = np.sum(r_opt)
                if val > best_val:
                    best_val = val
                    best_result = (x_opt, y_opt, r_opt, val)
            else:
                for scale in [0.99, 0.98, 0.95, 0.90]:
                    r_test = r_opt * scale
                    if check_validity(x_opt, y_opt, r_test):
                        val = np.sum(r_test)
                        if val > best_val:
                            best_val = val
                            best_result = (x_opt, y_opt, r_test, val)
                        break
        except Exception:
            pass

        # Also try with hard objective from the same starting point
        try:
            def objective_hard(params):
                x = params[:N]
                y = params[N:2*N]
                r = compute_radii(x, y)
                return -np.sum(r)

            res2 = minimize(objective_hard, params0, method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt2 = res2.x[:N]
            y_opt2 = res2.x[N:2*N]
            r_opt2 = compute_radii(x_opt2, y_opt2)

            if check_validity(x_opt2, y_opt2, r_opt2):
                val2 = np.sum(r_opt2)
                if val2 > best_val:
                    best_val = val2
                    best_result = (x_opt2, y_opt2, r_opt2, val2)
        except Exception:
            pass

        if best_result is not None:
            return best_result
        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with SA
    modes = ['grid', 'hex', 'random']
    num_starts = 30
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 15:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA with more iterations (vectorized forces are faster)
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=5000)

        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa
            best_y = y_sa
            best_r = r_sa

        # Gradient polish on SA result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_sa, y_sa, use_smooth=True, beta=80.0)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        final_polish = gradient_polish(best_x, best_y, use_smooth=True, beta=150.0)
        if final_polish is not None:
            xp, yp, rp, vp = final_polish
            if vp > best_val:
                best_val = vp
                best_x = xp
                best_y = yp
                best_r = rp

        # One more pass with even tighter smoothness
        if time.time() - start_time < DEADLINE - 2:
            final_polish2 = gradient_polish(best_x, best_y, use_smooth=True, beta=300.0)
            if final_polish2 is not None:
                xp, yp, rp, vp = final_polish2
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.90, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
