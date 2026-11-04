"""
Circle packing in unit square using Langevin dynamics + soft-penalty L-BFGS-B annealing.

Strategy:
1. Initialize with diverse configurations (hex, grid, random)
2. Use Langevin dynamics with adaptive cooling to explore configuration space
3. At each cooling step, run L-BFGS-B on smooth objective: sum(r) - lambda * penalty
4. Anneal lambda from high to low to gradually enforce constraints
5. Multi-restart with different seeds and initializations
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


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

    def project_feasible(x, y, r, tol=1e-9):
        """Project circles to satisfy all constraints by scaling down radii if needed."""
        r = np.minimum(r, x)
        r = np.minimum(r, 1.0 - x)
        r = np.minimum(r, y)
        r = np.minimum(r, 1.0 - y)
        
        for _ in range(50):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            violated = (dist < r[:, None] + r[None, :] - tol) & (dist > 0)
            if not np.any(violated):
                break
            
            violations = (r[:, None] + r[None, :] - dist) * violated
            violations[~violated] = 0
            max_viol_idx = np.unravel_index(np.argmax(violations), violations.shape)
            i, j = max_viol_idx
            
            if i == j:
                continue
            
            needed = dist[i, j]
            current = r[i] + r[j]
            scale = needed / current if current > 0 else 0.5
            r[i] *= scale
            r[j] *= scale
        
        r = np.maximum(r, 1e-8)
        return r

    def smooth_objective(params, lam):
        """
        Smooth objective: sum(r_i) - lam * sum(violations^2)
        where violations are max(0, r_i + r_j - dist_ij) for pairs,
        and max(0, r_i - dist_to_boundary) for boundaries.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Compute radii
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
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Penalty for pairwise overlaps
        # For each pair (i,j), violation = max(0, r_i + r_j - dist_ij)
        # But r depends on positions, so this is complex.
        # Instead, use a simpler penalty: for each pair, if dist < some threshold, penalize.
        # Actually, let's use the fact that r_i = min(boundaries, 0.5 * min_dist_to_others)
        # So the constraint is automatically satisfied by construction of r.
        # The real issue is that we want to maximize sum(r) subject to feasibility.
        
        # Alternative approach: define objective as sum(r) where r is computed as above,
        # and add a small penalty for circles that are too close to boundaries or each other
        # in a way that encourages spreading out.
        
        # Actually, since r is already computed to satisfy all constraints,
        # we just need to maximize sum(r). The "soft penalty" approach is:
        # Use a smooth approximation of the min function in r computation.
        
        # Let's use a different strategy: define the objective as sum of radii
        # where radii are computed with a soft min, and add penalty for any
        # residual violations.
        
        # For now, just return sum_r since r is already feasible by construction.
        # The optimization will try to maximize this.
        
        return -sum_r  # Minimize negative sum

    def smooth_objective_with_penalty(params, lam, beta=50.0):
        """
        Objective with soft-min approximation for radii and penalty for violations.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Soft min for boundary constraints
        r_bound_soft = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        half_dist = 0.5 * dist
        
        # Combine boundary and pairwise constraints with soft min
        # For each circle i, r_i = min(r_bound_i, min_j(0.5 * dist_ij))
        # Use log-sum-exp for soft min
        
        # Create matrix of all constraints for each circle
        # Row i: [r_bound_i, 0.5*dist_i0, 0.5*dist_i1, ..., 0.5*dist_i(N-1)]
        constraints = np.zeros((N, N + 1))
        constraints[:, 0] = r_bound_soft
        constraints[:, 1:] = half_dist
        
        # Soft min using log-sum-exp
        neg_beta = -beta * constraints
        max_neg = np.max(neg_beta, axis=1, keepdims=True)
        exp_vals = np.exp(neg_beta - max_neg)
        # Set inf entries to 0
        exp_vals = np.where(np.isinf(constraints), 0.0, exp_vals)
        log_sum = np.log(np.sum(exp_vals, axis=1, keepdims=True))
        soft_min = -(max_neg + log_sum) / beta
        r_soft = soft_min[:, 0]
        r_soft = np.maximum(r_soft, 1e-6)
        
        sum_r = np.sum(r_soft)
        
        # Penalty for actual violations (to encourage feasibility)
        # Compute hard radii
        min_dist = np.min(dist, axis=1)
        r_hard = np.minimum(r_bound_soft, 0.5 * min_dist)
        r_hard = np.maximum(r_hard, 1e-6)
        
        # Check pairwise violations
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                violation = r_hard[i] + r_hard[j] - dist[i, j]
                if violation > 0:
                    penalty += violation ** 2
        
        # Boundary violations
        for i in range(N):
            for bound in [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]:
                violation = r_hard[i] - bound
                if violation > 0:
                    penalty += violation ** 2
        
        return -(sum_r - lam * penalty)

    def smooth_gradient(params, lam, beta=50.0):
        """
        Gradient of smooth_objective_with_penalty.
        """
        x = params[:N]
        y = params[N:2*N]
        
        r_bound_soft = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        half_dist = 0.5 * dist
        
        constraints = np.zeros((N, N + 1))
        constraints[:, 0] = r_bound_soft
        constraints[:, 1:] = half_dist
        
        # Soft min weights
        neg_beta = -beta * constraints
        max_neg = np.max(neg_beta, axis=1, keepdims=True)
        exp_vals = np.exp(neg_beta - max_neg)
        exp_vals = np.where(np.isinf(constraints), 0.0, exp_vals)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        weights = exp_vals / sum_exp  # Shape (N, N+1)
        
        # Gradient of soft_min w.r.t constraints
        # d(soft_min)/d(constraint_k) = -weights_k / beta
        
        # Now chain rule: d(soft_min)/dx_i = sum_k weights_k * d(constraint_k)/dx_i
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        for i in range(N):
            # d(r_bound_i)/dx_i: depends on which boundary is active
            # For soft min, we use the weighted average
            w_bound = weights[i, 0]
            w_pairs = weights[i, 1:]
            
            # d(r_bound_i)/dx_i
            # r_bound_i = min(x_i, 1-x_i, y_i, 1-y_i)
            # Soft approximation: use the value closest to the minimum
            # For gradient, approximate as:
            # If x_i is the min, d/dx_i = 1
            # If 1-x_i is the min, d/dx_i = -1
            # etc.
            # For soft min, blend these.
            
            vals = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_val = np.min(vals)
            tol = 0.01
            
            dr_bound_dxi = 0.0
            if abs(x[i] - min_val) < tol:
                dr_bound_dxi += 1.0
            if abs(1.0 - x[i] - min_val) < tol:
                dr_bound_dxi -= 1.0
            
            dr_bound_dyi = 0.0
            if abs(y[i] - min_val) < tol:
                dr_bound_dyi += 1.0
            if abs(1.0 - y[i] - min_val) < tol:
                dr_bound_dyi -= 1.0
            
            # Contribution from boundary constraint
            grad_x[i] += -w_bound * dr_bound_dxi / beta
            grad_y[i] += -w_bound * dr_bound_dyi / beta
            
            # Contribution from pairwise constraints
            for j in range(N):
                if i == j:
                    continue
                w_pair = w_pairs[j]
                if w_pair < 1e-10:
                    continue
                
                # d(0.5 * dist_ij)/dx_i = 0.5 * (x_i - x_j) / dist_ij
                d_ij = dist[i, j]
                if d_ij > 1e-9:
                    d_dist_dx_i = 0.5 * (x[i] - x[j]) / d_ij
                    d_dist_dy_i = 0.5 * (y[i] - y[j]) / d_ij
                    grad_x[i] += -w_pair * d_dist_dx_i / beta
                    grad_y[i] += -w_pair * d_dist_dy_i / beta
        
        # Gradient of penalty term (simplified - use numerical if needed)
        # For now, just return the gradient of the soft-min part
        # The penalty gradient would add additional terms
        
        return np.concatenate([-grad_x, -grad_y])  # Negative because we minimize -sum_r

    def make_initial_guess(seed, mode='hex'):
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
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        elif mode == 'hex':
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
            x += rng.uniform(-0.03, 0.03, N)
            y += rng.uniform(-0.03, 0.03, N)
        else:
            x = rng.uniform(0.15, 0.85, N)
            y = rng.uniform(0.15, 0.85, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def langevin_step(x, y, T, rng, dt=0.01):
        """
        One step of Langevin dynamics.
        x_new = x + force * dt + sqrt(2*T*dt) * noise
        """
        # Compute forces (gradient of potential)
        # Force pushes circles apart when they overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Compute current radii
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        
        # Forces: repulsion when circles are close
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = dist[i, j]
                if d < 1e-9:
                    continue
                # Repulsive force if distance < sum of radii
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    # Force proportional to overlap
                    f = 10.0 * overlap
                    fx[i] += f * (x[i] - x[j]) / d
                    fy[i] += f * (y[i] - y[j]) / d
                    fx[j] -= f * (x[i] - x[j]) / d
                    fy[j] -= f * (y[i] - y[j]) / d
        
        # Boundary forces (push away from walls)
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += 10.0 * (r[i] - x[i])
            if 1.0 - x[i] < r[i]:
                fx[i] -= 10.0 * (r[i] - (1.0 - x[i]))
            if y[i] < r[i]:
                fy[i] += 10.0 * (r[i] - y[i])
            if 1.0 - y[i] < r[i]:
                fy[i] -= 10.0 * (r[i] - (1.0 - y[i]))
        
        # Langevin update
        noise_x = rng.normal(0, np.sqrt(2 * T * dt), N)
        noise_y = rng.normal(0, np.sqrt(2 * T * dt), N)
        
        x_new = x + fx * dt + noise_x
        y_new = y + fy * dt + noise_y
        
        x_new = np.clip(x_new, 0.01, 0.99)
        y_new = np.clip(y_new, 0.01, 0.99)
        
        return x_new, y_new

    def lbfgs_polish(x0, y0, lam, beta, time_limit=2.0):
        """
        Run L-BFGS-B to optimize positions for given penalty weight.
        """
        params0 = np.concatenate([x0, y0])
        
        def objective(params):
            return smooth_objective_with_penalty(params, lam, beta)
        
        def gradient(params):
            return smooth_gradient(params, lam, beta)
        
        # Use L-BFGS-B
        bounds = [(0.01, 0.99)] * (2 * N)
        
        try:
            result = minimize(
                objective,
                params0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            x_opt = result.x[:N]
            y_opt = result.x[N:2*N]
            
            return x_opt, y_opt, result.fun
            
        except Exception:
            return x0, y0, objective(params0)

    def optimize_with_annealing(x0, y0, seed, time_limit=10.0):
        """
        Main optimization: Langevin dynamics + L-BFGS-B with annealing.
        """
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        
        best_x = x.copy()
        best_y = y.copy()
        best_val = np.sum(compute_radii(x, y))
        
        # Annealing schedule
        T_start = 0.05
        T_end = 0.001
        n_anneal_steps = 20
        
        lam_start = 1.0
        lam_end = 0.01
        beta_start = 20.0
        beta_end = 100.0
        
        anneal_start = time.time()
        
        for step in range(n_anneal_steps):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Current temperature and parameters
            frac = step / max(1, n_anneal_steps - 1)
            T = T_start * (T_end / T_start) ** frac
            lam = lam_start * (lam_end / lam_start) ** frac
            beta = beta_start + (beta_end - beta_start) * frac
            
            # Langevin steps
            n_langevin = 50
            for _ in range(n_langevin):
                if time.time() - start_time > DEADLINE - 5:
                    break
                x, y = langevin_step(x, y, T, rng)
            
            # L-BFGS-B polish
            remaining = DEADLINE - 5 - (time.time() - start_time)
            if remaining < 1:
                break
            
            lbfgs_time = min(2.0, remaining * 0.5)
            x, y, _ = lbfgs_polish(x, y, lam, beta, time_limit=lbfgs_time)
            
            # Evaluate
            r = compute_radii(x, y)
            val = np.sum(r)
            
            if val > best_val:
                best_val = val
                best_x = x.copy()
                best_y = y.copy()
        
        return best_x, best_y, best_val

    # Initial best: hex configuration
    x0, y0 = make_initial_guess(42, 'hex')
    best_x = x0.copy()
    best_y = y0.copy()
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start with different modes and seeds
    modes = ['hex', 'grid', 'random']
    num_starts = 6

    for i in range(num_starts):
        if time.time() > DEADLINE - 15:
            break

        mode = modes[i % len(modes)]
        seed = i * 13 + 7

        x0, y0 = make_initial_guess(seed, mode)
        
        # Check if this initialization is already good
        r0 = compute_radii(x0, y0)
        val0 = np.sum(r0)
        
        remaining = DEADLINE - 15 - (time.time() - start_time)
        if remaining < 5:
            break
        
        opt_time = min(12.0, remaining * 0.8)
        x_opt, y_opt, val_opt = optimize_with_annealing(x0, y0, seed, opt_time)
        
        if val_opt > best_val:
            best_val = val_opt
            best_x = x_opt.copy()
            best_y = y_opt.copy()
            best_r = compute_radii(best_x, best_y)

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        r_projected = project_feasible(best_x, best_y, best_r)
        if check_validity(best_x, best_y, r_projected):
            best_r = r_projected
        else:
            for scale in np.linspace(0.99, 0.85, 15):
                r_test = best_r * scale
                if check_validity(best_x, best_y, r_test):
                    best_r = r_test
                    break
            else:
                best_r = compute_radii(best_x, best_y)

    # Final L-BFGS-B polish with low penalty
    if time.time() < DEADLINE - 5:
        final_time = min(5.0, DEADLINE - 5 - (time.time() - start_time))
        if final_time > 1.0:
            x_final, y_final, _ = lbfgs_polish(best_x, best_y, 0.01, 100.0, time_limit=final_time)
            r_final = compute_radii(x_final, y_final)
            val_final = np.sum(r_final)
            if val_final > best_val and check_validity(x_final, y_final, r_final):
                best_x = x_final
                best_y = y_final
                best_r = r_final
                best_val = val_final

    return np.column_stack((best_x, best_y, best_r))
