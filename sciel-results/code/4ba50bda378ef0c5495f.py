"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Discrete best-fit local search for deterministic neighborhood improvement.
Added: Delaunay-style force-directed initialization and Hard Sub-gradient polish.
Vectorized gradient via np.add.at for faster L-BFGS-B convergence.
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

    def compute_radii_smooth(x, y, beta=50.0):
        """Smooth approximation of radii using log-sum-exp soft-min."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta = -beta * full_constraints
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        log_sum_row = np.log(np.sum(np.exp(neg_beta - max_neg_row), axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r, soft_min_all

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses np.add.at for vectorized gradient accumulation."""
        x = params[:N]
        y = params[N:2*N]

        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        half_dist = 0.5 * dist
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        neg_beta_c = -beta * full_constraints
        max_nb = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_nb)
        log_sum = np.log(np.sum(exp_terms, axis=1, keepdims=True))
        r = -(max_nb + log_sum) / beta
        r = r[:, 0]

        w = exp_terms / np.sum(exp_terms, axis=1, keepdims=True)

        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Gradient of constraint c[i,j] = 0.5*dist(i,j) w.r.t x_i, y_i, x_j, y_j
        d_c_dx_i = 0.5 * dx / dist_safe
        d_c_dy_i = 0.5 * dy / dist_safe
        d_c_dx_j = -0.5 * dx / dist_safe
        d_c_dy_j = -0.5 * dy / dist_safe

        # Boundary gradients for diagonal (i==j) entries
        all_bounds = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
        min_idx = np.argmin(all_bounds, axis=1)

        grad_rb_dx = np.zeros(N)
        grad_rb_dy = np.zeros(N)
        grad_rb_dx[min_idx == 0] = 1.0
        grad_rb_dx[min_idx == 1] = -1.0
        grad_rb_dy[min_idx == 2] = 1.0
        grad_rb_dy[min_idx == 3] = -1.0

        # Set diagonal entries to boundary gradients
        d_c_dx_i[range(N), range(N)] = grad_rb_dx
        d_c_dy_i[range(N), range(N)] = grad_rb_dy
        d_c_dx_j[range(N), range(N)] = grad_rb_dx
        d_c_dy_j[range(N), range(N)] = grad_rb_dy

        # Accumulate: dr_dx[i] = sum_j w[i,j] * d_c_dx_i[i,j] + sum_j w[j,i] * d_c_dx_j[j,i]
        # Note: d_c_dx_j[j,i] = -0.5*(x_j-x_i)/dist = 0.5*(x_i-x_j)/dist = d_c_dx_i[i,j]
        # So the contribution from j's perspective to i is w[j,i] * d_c_dx_i[i,j] (by symmetry)
        # Actually let's be careful: 
        # r_i = softmin_j(c[i,j]) where c[i,j] depends on both i and j
        # dr_i/dx_i = sum_j w[i,j] * dc[i,j]/dx_i
        # dr_i/dx_j = sum_k w[i,k] * dc[i,k]/dx_j (nonzero only when k=j)
        # But we need total dr/dx_i = sum_i dr_i/dx_i (chain rule for sum_r)
        # sum_r = sum_i r_i
        # d(sum_r)/dx_i = sum_i dr_i/dx_i
        # dr_i/dx_i = sum_j w[i,j] * dc[i,j]/dx_i  (only j where c[i,j] involves x_i, which is all j since c[i,j]=0.5*dist(i,j))
        # dr_k/dx_i for k != i: dr_k/dx_i = w[k,i] * dc[k,i]/dx_i (only the term where j=i in c[k,j])
        # dc[k,i]/dx_i = -0.5*(x_k-x_i)/dist(k,i) = 0.5*(x_i-x_k)/dist = same as dc[i,k]/dx_i by symmetry? 
        # dc[i,k]/dx_i = 0.5*(x_i-x_k)/dist(i,k)
        # dc[k,i]/dx_i = -0.5*(x_k-x_i)/dist(k,i) = 0.5*(x_i-x_k)/dist(i,k) = dc[i,k]/dx_i
        # So: d(sum_r)/dx_i = sum_j w[i,j]*dc[i,j]/dx_i + sum_{k!=i} w[k,i]*dc[k,i]/dx_i
        # = sum_j w[i,j]*dc[i,j]/dx_i + sum_{k!=i} w[k,i]*dc[i,k]/dx_i
        # = w[i,i]*dc[i,i]/dx_i + sum_{j!=i} w[i,j]*dc[i,j]/dx_i + sum_{k!=i} w[k,i]*dc[i,k]/dx_i
        # The first term is the boundary gradient contribution.
        
        # Vectorized:
        # For each i, contribution from row i: sum_j w[i,j] * d_c_dx_i[i,j]
        row_contrib_dx = np.sum(w * d_c_dx_i, axis=1)
        row_contrib_dy = np.sum(w * d_c_dy_i, axis=1)
        
        # For each i, contribution from column i (other rows k where j=i):
        # sum_k w[k,i] * d_c_dx_j[k,i] but d_c_dx_j[k,i] = -0.5*dx[k,i]/dist = 0.5*dx[i,k]/dist = d_c_dx_i[i,k]
        # So col_contrib_dx[i] = sum_k w[k,i] * d_c_dx_i[i,k]
        # This is (w.T * d_c_dx_i) summed appropriately... 
        # Actually: col_contrib_dx[i] = sum_k w[k,i] * d_c_dx_i[i,k]
        # = sum_k d_c_dx_i[i,k] * w[k,i]
        # This is element-wise product of row i of d_c_dx_i with column i of w, then sum over k
        # = np.sum(d_c_dx_i * w.T, axis=1)
        
        col_contrib_dx = np.sum(d_c_dx_i * w.T, axis=1)
        col_contrib_dy = np.sum(d_c_dy_i * w.T, axis=1)

        dr_dx = row_contrib_dx + col_contrib_dx
        dr_dy = row_contrib_dy + col_contrib_dy

        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

    def hard_subgradient_polish(x, y, num_steps=20, step_size=0.005):
        """
        Refine the solution using the exact sub-gradient of the hard min function.
        Vectorized: find active constraints via argmin, then accumulate gradients.
        """
        x = x.copy()
        y = y.copy()
        
        for _ in range(num_steps):
            if time.time() - start_time > DEADLINE - 2:
                break
                
            r_bound = np.minimum(
                np.minimum(x, 1.0 - x),
                np.minimum(y, 1.0 - y)
            )

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)

            half_dist = 0.5 * dist

            # For each i, find if boundary or neighbor is the min constraint
            min_neighbor_d = np.min(half_dist, axis=1)
            boundary_active = r_bound < min_neighbor_d
            min_neighbor_j = np.argmin(half_dist, axis=1)

            grad_x = np.zeros(N)
            grad_y = np.zeros(N)

            # Boundary-active circles
            if np.any(boundary_active):
                idx_b = np.where(boundary_active)[0]
                bounds_b = np.stack([x[idx_b], 1.0-x[idx_b], y[idx_b], 1.0-y[idx_b]], axis=1)
                min_idx_b = np.argmin(bounds_b, axis=1)
                grad_x[idx_b] += np.where(min_idx_b == 0, 1.0, np.where(min_idx_b == 1, -1.0, 0.0))
                grad_y[idx_b] += np.where(min_idx_b == 2, 1.0, np.where(min_idx_b == 3, -1.0, 0.0))

            # Neighbor-active circles
            if np.any(~boundary_active):
                idx_n = np.where(~boundary_active)[0]
                j_n = min_neighbor_j[idx_n]
                d_n = dist[idx_n, j_n]
                valid = d_n > 1e-10
                if np.any(valid):
                    idx_n_v = idx_n[valid]
                    j_n_v = j_n[valid]
                    d_n_v = d_n[valid]
                    gx = 0.5 * dx[idx_n_v, j_n_v] / d_n_v
                    gy = 0.5 * dy[idx_n_v, j_n_v] / d_n_v
                    np.add.at(grad_x, idx_n_v, gx)
                    np.add.at(grad_y, idx_n_v, gy)
                    np.add.at(grad_x, j_n_v, -gx)
                    np.add.at(grad_y, j_n_v, -gy)

            norm = np.sqrt(np.sum(grad_x**2) + np.sum(grad_y**2))
            if norm < 1e-10:
                break

            alpha = step_size / norm
            x_new = x + alpha * grad_x
            y_new = y + alpha * grad_y
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)

            r_old = compute_radii(x, y)
            val_old = np.sum(r_old)
            r_new = compute_radii(x_new, y_new)
            val_new = np.sum(r_new)

            if val_new > val_old - 1e-10:
                x = x_new
                y = y_new
            else:
                alpha *= 0.5
                x_new = x + alpha * grad_x
                y_new = y + alpha * grad_y
                x_new = np.clip(x_new, 0.01, 0.99)
                y_new = np.clip(y_new, 0.01, 0.99)
                r_new = compute_radii(x_new, y_new)
                val_new = np.sum(r_new)
                if val_new > val_old - 1e-10:
                    x = x_new
                    y = y_new
                else:
                    break

        return x, y

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
        elif mode == 'delaunay':
            # Force-directed relaxation to spread points
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)
            for _ in range(50):
                fx = np.zeros(N)
                fy = np.zeros(N)
                # Repulsion
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(d, 1e10)
                f_mag = 1.0 / (d**2 + 0.01)
                np.fill_diagonal(f_mag, 0)
                fx += np.sum(-f_mag * dx / d, axis=1)
                fy += np.sum(-f_mag * dy / d, axis=1)
                # Boundary attraction
                fx += 5.0 * np.clip(0.1 - x, 0, None)
                fx -= 5.0 * np.clip(x - 0.9, 0, None)
                fy += 5.0 * np.clip(0.1 - y, 0, None)
                fy -= 5.0 * np.clip(y - 0.9, 0, None)
                
                x += 0.05 * fx
                y += 0.05 * fy
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing to optimize center positions with vectorized forces."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        best_x = x.copy()
        best_y = y.copy()

        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)

        repulse_k = 2.0
        boundary_k = 5.0

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break

            T = T_start * (cool_rate ** it)

            r_curr = compute_radii(x, y)
            curr_val = np.sum(r_curr)

            fx = np.zeros(N)
            fy = np.zeros(N)

            fx += boundary_k * np.clip(0.1 - x, 0, None)
            fx -= boundary_k * np.clip(x - 0.9, 0, None)
            fy += boundary_k * np.clip(0.1 - y, 0, None)
            fy -= boundary_k * np.clip(y - 0.9, 0, None)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)

            f_mag = repulse_k / (d**2 + 0.01)
            np.fill_diagonal(f_mag, 0)

            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)

            scale = 0.02 * T / T_start + 0.005
            dx_prop = fx * scale + rng.normal(0, scale, N)
            dy_prop = fy * scale + rng.normal(0, scale, N)

            x_new = x + dx_prop
            y_new = y + dy_prop

            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

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

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=80):
        """Discrete best-fit local search with adaptive step sizes.
        Optimized: precompute directions, use vectorized radius updates where possible."""
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        directions = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1, 1], [1, -1], [-1, 1], [-1, -1]
        ], dtype=float)
        directions /= np.sqrt(2.0)

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            best_improvement = 0.0
            best_move = None
            
            for step in step_sizes:
                for i in range(N):
                    # Try all 8 directions for circle i
                    x_cand = x[i] + directions[:, 0] * step
                    y_cand = y[i] + directions[:, 1] * step
                    
                    # Filter valid positions
                    valid = (x_cand >= 0.01) & (x_cand <= 0.99) & (y_cand >= 0.01) & (y_cand <= 0.99)
                    if not np.any(valid):
                        continue
                    
                    x_cand_v = x_cand[valid]
                    y_cand_v = y_cand[valid]
                    dir_idx_v = np.where(valid)[0]
                    
                    for k in range(len(x_cand_v)):
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_cand_v[k]
                        y_temp[i] = y_cand_v[k]
                        
                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val
                        
                        if improvement > best_improvement + 1e-10:
                            best_improvement = improvement
                            best_move = (i, dir_idx_v[k], step)
            
            if best_move is not None:
                i, dir_idx, step = best_move
                x[i] += directions[dir_idx, 0] * step
                y[i] += directions[dir_idx, 1] * step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def gradient_polish(x0, y0, use_smooth=True, beta_start=20.0, beta_end=500.0, num_steps=5):
        """Final gradient-based refinement using L-BFGS-B with analytic gradient."""
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        betas = np.geomspace(beta_start, beta_end, num_steps)
        
        current_params = params0.copy()
        best_result = None
        best_val = -np.inf
        
        def get_val_from_params(params):
            x = params[:N]
            y = params[N:2*N]
            r = compute_radii(x, y)
            if check_validity(x, y, r):
                return np.sum(r)
            for scale in [0.999, 0.99, 0.95, 0.90]:
                r_test = r * scale
                if check_validity(x, y, r_test):
                    return np.sum(r_test)
            return -np.inf

        def update_best(params):
            nonlocal best_result, best_val
            val = get_val_from_params(params)
            if val > best_val:
                best_val = val
                r_hard = compute_radii(params[:N], params[N:])
                if check_validity(params[:N], params[N:], r_hard):
                    best_result = (params[:N].copy(), params[N:].copy(), r_hard, val)
                else:
                    for scale in [0.999, 0.99, 0.98]:
                        r_test = r_hard * scale
                        if check_validity(params[:N], params[N:], r_test):
                            best_result = (params[:N].copy(), params[N:].copy(), r_test, np.sum(r_test))
                            break

        update_best(current_params)

        for beta in betas:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            try:
                res = minimize(smooth_objective_and_grad, current_params, 
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=True,
                              options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80})
                current_params = res.x
                update_best(current_params)
            except Exception:
                pass

        if time.time() - start_time < DEADLINE - 2:
            beta_final = min(beta_end * 2, 2000.0)
            try:
                res2 = minimize(smooth_objective_and_grad, current_params,
                                method='L-BFGS-B', bounds=bounds,
                                args=(beta_final,),
                                jac=True,
                                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100})
                current_params = res2.x
                update_best(current_params)
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

    # Multi-start with SA + best-fit + gradient polish
    modes = ['grid', 'hex', 'random', 'delaunay']
    num_starts = 70  # More starts enabled by faster gradient
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA with more iterations thanks to faster gradient
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=5000)

        # Best-fit local search on SA result
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish on best-fit result
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # NEW: Hard Sub-gradient polish
        if time.time() - start_time < DEADLINE - 5:
            x_hp, y_hp = hard_subgradient_polish(x_sa, y_sa, num_steps=10, step_size=0.005)
            r_hp = compute_radii(x_hp, y_hp)
            val_hp = np.sum(r_hp)
            if val_hp > val_sa:
                x_sa, y_sa, r_sa, val_sa = x_hp, y_hp, r_hp, val_hp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try best-fit on the initial guess directly (skip SA for some starts)
        if i % 4 == 3 and time.time() - start_time < DEADLINE - 15:
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x0, y0)
            if time.time() - start_time < DEADLINE - 10:
                polished2 = gradient_polish(x_bf2, y_bf2, use_smooth=True)
                if polished2 is not None:
                    xp2, yp2, rp2, vp2 = polished2
                    if vp2 > val_bf2:
                        val_bf2 = vp2
                        x_bf2, y_bf2, r_bf2 = xp2, yp2, rp2
            if val_bf2 > best_val:
                best_val = val_bf2
                best_x = x_bf2.copy()
                best_y = y_bf2.copy()
                best_r = r_bf2.copy()

    # Final high-precision polish: run gradient polish on the best solution found
    if time.time() - start_time < DEADLINE - 5:
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        for polish_pass in range(3):
            if time.time() - start_time > DEADLINE - 3:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y, use_smooth=True, 
                                           beta_start=min(beta_s, 2000.0), 
                                           beta_end=min(beta_e, 5000.0), 
                                           num_steps=6)
            if final_polish is not None:
                xp, yp, rp, vp = final_polish
                if vp > best_val:
                    best_val = vp
                    best_x = xp
                    best_y = yp
                    best_r = rp
                # Do another best-fit pass after gradient polish
                if time.time() - start_time < DEADLINE - 3:
                    x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(best_x, best_y)
                    if val_bf2 > best_val:
                        best_val = val_bf2
                        best_x = x_bf2
                        best_y = y_bf2
                        best_r = r_bf2
                # Final Hard Polish
                if time.time() - start_time < DEADLINE - 2:
                    x_hp, y_hp = hard_subgradient_polish(best_x, best_y, num_steps=5, step_size=0.002)
                    r_hp = compute_radii(x_hp, y_hp)
                    val_hp = np.sum(r_hp)
                    if val_hp > best_val:
                        best_val = val_hp
                        best_x = x_hp
                        best_y = y_hp
                        best_r = r_hp

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
