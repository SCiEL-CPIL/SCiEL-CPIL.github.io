"""
Circle packing in unit square using simulated annealing with repulsion forces.
Centers are optimized; radii are derived from local geometry.
SA provides global exploration to escape local minima.
Smooth soft-min surrogate with analytic gradient for fast L-BFGS-B polish.
Discrete best-fit local search for deterministic neighborhood improvement.
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

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient with respect to params = [x, y].
        Uses analytic gradient of the log-sum-exp soft-min.
        """
        x = params[:N]
        y = params[N:2*N]

        # Boundary constraint
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix: c_ij for all i,j
        # c_ij = 0.5 * dist(i,j) for i != j
        # c_ii = r_bound[i]
        c = 0.5 * dist
        np.fill_diagonal(c, r_bound)

        # Soft-min computation
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r[:, 0]

        # Weights: w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / sum_exp  # (N, N)

        # Gradient of c_ij w.r.t. parameters
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        
        # Off-diagonal gradients
        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal gradients for r_bound_i
        bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)
        beta_b = 100.0
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb
        
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)
        np.fill_diagonal(dc_dx_j, grad_rb_dx)
        np.fill_diagonal(dc_dy_j, grad_rb_dy)

        dr_dx_self = np.sum(w * dc_dx_i, axis=1)
        dr_dy_self = np.sum(w * dc_dy_i, axis=1)
        dr_dx_cross = np.sum(w * dc_dx_j, axis=0)
        dr_dy_cross = np.sum(w * dc_dy_j, axis=0)

        dr_dx = dr_dx_self + dr_dx_cross - np.diag(w) * np.diag(dc_dx_i)
        dr_dy = dr_dy_self + dr_dy_cross - np.diag(w) * np.diag(dc_dy_i)

        grad = np.concatenate([-dr_dx, -dr_dy])
        obj_val = -np.sum(r)
        return obj_val, grad

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
        else:
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

    def best_fit_local_search(x, y, step_sizes=None, max_rounds=60):
        """Discrete best-fit local search with adaptive step sizes."""
        if step_sizes is None:
            step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        dirs = np.array([
            [1, 0], [-1, 0], [0, 1], [0, -1],
            [1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(2), -1/np.sqrt(2)],
            [-1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), -1/np.sqrt(2)]
        ])

        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        curr_val = np.sum(r)

        for rnd in range(max_rounds):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            best_improvement = 1e-10
            best_i = -1
            best_dir = None
            best_step = None
            
            for step in step_sizes:
                x_new = x[:, None] + dirs[None, :, 0] * step
                y_new = y[:, None] + dirs[None, :, 1] * step
                
                valid = (x_new >= 0.01) & (x_new <= 0.99) & (y_new >= 0.01) & (y_new <= 0.99)
                
                for i in range(N):
                    for d in range(8):
                        if not valid[i, d]:
                            continue
                        
                        x_temp = x.copy()
                        y_temp = y.copy()
                        x_temp[i] = x_new[i, d]
                        y_temp[i] = y_new[i, d]
                        
                        r_new = compute_radii(x_temp, y_temp)
                        new_val = np.sum(r_new)
                        improvement = new_val - curr_val
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_i = i
                            best_dir = d
                            best_step = step
                
                if best_improvement > step * 0.1:
                    break
            
            if best_i >= 0:
                x[best_i] += dirs[best_dir, 0] * best_step
                y[best_i] += dirs[best_dir, 1] * best_step
                r = compute_radii(x, y)
                curr_val = np.sum(r)
            else:
                break

        return x, y, compute_radii(x, y), np.sum(compute_radii(x, y))

    def voronoi_polish(x0, y0, max_iters=500):
        """Voronoi-based refinement: move each center toward the centroid of its Voronoi cell.
        This is equivalent to Lloyd's algorithm for centroidal Voronoi tessellation.
        """
        x = x0.copy()
        y = y0.copy()
        best_val = np.sum(compute_radii(x, y))
        best_x = x.copy()
        best_y = y.copy()

        for it in range(max_iters):
            if time.time() - start_time > DEADLINE - 5:
                break

            # Compute Voronoi cells by assigning each point in a fine grid to nearest center
            grid_size = 200
            gx, gy = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
            gx_flat = gx.ravel()
            gy_flat = gy.ravel()

            # Assign each grid point to nearest center
            dists = np.zeros((grid_size * grid_size, N))
            for k in range(N):
                dists[:, k] = (gx_flat - x[k])**2 + (gy_flat - y[k])**2
            
            assignments = np.argmin(dists, axis=1)  # (grid_size^2,)

            # Compute centroids of each cell
            centroids_x = np.zeros(N)
            centroids_y = np.zeros(N)
            counts = np.zeros(N)

            for k in range(N):
                mask = assignments == k
                if np.sum(mask) > 0:
                    centroids_x[k] = np.mean(gx_flat[mask])
                    centroids_y[k] = np.mean(gy_flat[mask])
                    counts[k] = np.sum(mask)
                else:
                    centroids_x[k] = x[k]
                    centroids_y[k] = y[k]

            # Move centers toward centroids with damping
            damping = 0.5
            x_new = x + damping * (centroids_x - x)
            y_new = y + damping * (centroids_y - y)
            x_new = np.clip(x_new, 0.05, 0.95)
            y_new = np.clip(y_new, 0.05, 0.95)

            r_new = compute_radii(x_new, y_new)
            new_val = np.sum(r_new)

            if new_val >= best_val - 1e-6:
                x = x_new
                y = y_new
                if new_val > best_val:
                    best_val = new_val
                    best_x = x.copy()
                    best_y = y.copy()
            else:
                # If moving to centroid worsens, try smaller step
                for damp in [0.25, 0.1, 0.05]:
                    x_try = x + damp * (centroids_x - x)
                    y_try = y + damp * (centroids_y - y)
                    x_try = np.clip(x_try, 0.05, 0.95)
                    y_try = np.clip(y_try, 0.05, 0.95)
                    r_try = compute_radii(x_try, y_try)
                    val_try = np.sum(r_try)
                    if val_try >= best_val - 1e-6:
                        x = x_try
                        y = y_try
                        if val_try > best_val:
                            best_val = val_try
                            best_x = x.copy()
                            best_y = y.copy()
                        break
                else:
                    break

        return best_x, best_y, compute_radii(best_x, best_y), best_val

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
                              options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50})
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
                                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
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

    # Multi-start with SA + Voronoi + best-fit + gradient polish
    modes = ['grid', 'hex', 'random']
    num_starts = 35
    for i in range(num_starts):
        if time.time() - start_time > DEADLINE - 20:
            break

        mode = modes[i % len(modes)]
        x0, y0 = make_initial_guess(i, mode)

        # Run SA
        x_sa, y_sa, r_sa, val_sa = sa_optimize(x0, y0, seed=i, max_iters=2500)

        # Voronoi polish on SA result
        if time.time() - start_time < DEADLINE - 10:
            x_vor, y_vor, r_vor, val_vor = voronoi_polish(x_sa, y_sa, max_iters=200)
            if val_vor > val_sa:
                x_sa, y_sa, r_sa, val_sa = x_vor, y_vor, r_vor, val_vor

        # Best-fit local search
        x_bf, y_bf, r_bf, val_bf = best_fit_local_search(x_sa, y_sa)

        # Gradient polish
        if time.time() - start_time < DEADLINE - 10:
            polished = gradient_polish(x_bf, y_bf, use_smooth=True)
            if polished is not None:
                xp, yp, rp, vp = polished
                if vp > val_bf:
                    x_sa, y_sa, r_sa, val_sa = xp, yp, rp, vp

        # Update best
        if val_sa > best_val:
            best_val = val_sa
            best_x = x_sa.copy()
            best_y = y_sa.copy()
            best_r = r_sa.copy()

        # Also try Voronoi + best-fit on the initial guess directly for some starts
        if i % 5 == 4 and time.time() - start_time < DEADLINE - 15:
            x_vor2, y_vor2, r_vor2, val_vor2 = voronoi_polish(x0, y0, max_iters=150)
            x_bf2, y_bf2, r_bf2, val_bf2 = best_fit_local_search(x_vor2, y_vor2)
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

    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        # Voronoi polish on the best solution
        x_vor_final, y_vor_final, r_vor_final, val_vor_final = voronoi_polish(best_x, best_y, max_iters=300)
        if val_vor_final > best_val:
            best_val = val_vor_final
            best_x = x_vor_final
            best_y = y_vor_final
            best_r = r_vor_final

        # Best-fit on the best solution
        x_bf_final, y_bf_final, r_bf_final, val_bf_final = best_fit_local_search(best_x, best_y)
        if val_bf_final > best_val:
            best_val = val_bf_final
            best_x = x_bf_final
            best_y = y_bf_final
            best_r = r_bf_final

        # Multiple gradient polish passes
        for polish_pass in range(3):
            if time.time() - start_time > DEADLINE - 3:
                break
            beta_s = 50.0 * (10 ** polish_pass)
            beta_e = 1000.0 * (10 ** polish_pass)
            final_polish = gradient_polish(best_x, best_y, use_smooth=True, 
                                           beta_start=min(beta_s, 3000.0), 
                                           beta_end=min(beta_e, 8000.0), 
                                           num_steps=5)
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
