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
        r = np.maximum(r, 1e-8)
        return r

    def check_validity(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

    def grow_radii(x, y, r, max_iter=50, tol=1e-7):
        """Iteratively grow radii until no more growth is possible."""
        r = r.copy()
        for _ in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            improved = False
            for i in range(N):
                # Max radius for circle i given current positions and other radii
                r_bound = np.minimum(
                    np.minimum(x[i], 1.0 - x[i]),
                    np.minimum(y[i], 1.0 - y[i])
                )
                max_r = r_bound
                for j in range(N):
                    if i == j:
                        continue
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    max_r = min(max_r, d - r[j])
                if max_r > r[i] + tol:
                    r[i] = max_r
                    improved = True
            if not improved:
                break
        return r

    def smooth_objective_and_grad(params, beta=50.0):
        """Compute -sum(r) and its gradient using log-sum-exp soft-min."""
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

        # Build constraint matrix
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

        # Objective: -sum(r)
        obj = -np.sum(r)

        # Weights
        w = exp_terms / sum_exp  # (N, N)

        # Gradients
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe
        dc_dx_j = -0.5 * dx / dist_safe
        dc_dy_j = -0.5 * dy / dist_safe

        # Diagonal: soft-min over 4 boundaries
        bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
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

        # dr_i/dx_k = sum_j w_ij * dc_ij/dx_k
        # grad_x[k] = -sum_i dr_i/dx_k (negative because we minimize -sum(r))
        # dr_i/dx_k = w_ik * dc_dx_j[i,k] + (if i==k) sum_j w_kj * dc_dx_i[k,j]
        # Simplified: grad_x[k] = -[sum_i w_ik * dc_dx_j[i,k] + sum_j w_kj * dc_dx_i[k,j]]
        # But dc_dx_j[i,k] = -0.5*dx[i,k]/dist[i,k] and dc_dx_i[k,j] = 0.5*dx[k,j]/dist[k,j]
        # Note: dc_dx_j[i,k] = -dc_dx_i[k,i] by symmetry
        
        # Vectorized computation:
        # Term 1: for each k, sum_i w[i,k] * dc_dx_j[i,k]
        # dc_dx_j[i,k] = -0.5 * dx[i,k] / dist_safe[i,k]
        term1_x = np.sum(w * dc_dx_j, axis=0)  # sum over i for each k
        # Term 2: for each k, sum_j w[k,j] * dc_dx_i[k,j]
        term2_x = np.sum(w * dc_dx_i, axis=1)  # sum over j for each k

        # But we need to be careful: the diagonal term is handled separately
        # Actually, let's just do it directly:
        # grad_x[k] = -sum_i sum_j w[i,j] * dc_ij/dx_k
        # dc_ij/dx_k is nonzero only when k==i (dc_dx_i[i,j]) or k==j (dc_dx_j[i,j])
        
        # For term where k==j: sum_i w[i,k] * dc_dx_j[i,k]
        # For term where k==i: sum_j w[k,j] * dc_dx_i[k,j]
        # But the (k,k) term is counted in both, so we need to subtract it once
        
        # Actually the formula dr_i/dx_k = sum_j w_ij * dc_ij/dx_k
        # When k==i: sum_j w_ij * dc_ij/dx_i = sum_j w_ij * dc_dx_i[i,j]
        # When k==j (j=k, i≠k): w_ik * dc_ik/dx_k = w_ik * dc_dx_j[i,k]
        # So dr_i/dx_k for fixed k:
        #   = (i==k) sum_j w_kj * dc_dx_i[k,j] + sum_{i≠k} w_ik * dc_dx_j[i,k]
        
        # grad_x[k] = -[sum_j w[k,j] * dc_dx_i[k,j] + sum_{i≠k} w[i,k] * dc_dx_j[i,k]]
        #            = -[sum_j w[k,j] * dc_dx_i[k,j] + sum_i w[i,k] * dc_dx_j[i,k] - w[k,k]*dc_dx_j[k,k]]
        # But dc_dx_j[k,k] = -0.5*dx[k,k]/dist[k,k] = 0 (since dx[k,k]=0)
        # So: grad_x[k] = -[sum_j w[k,j] * dc_dx_i[k,j] + sum_i w[i,k] * dc_dx_j[i,k]]
        
        grad_x = -(np.sum(w * dc_dx_i, axis=1) + np.sum(w * dc_dx_j, axis=0))
        grad_y = -(np.sum(w * dc_dy_i, axis=1) + np.sum(w * dc_dy_j, axis=0))

        return obj, np.concatenate([grad_x, grad_y])

    def polish(x, y, max_iters=200, beta_start=30.0, beta_end=200.0):
        """Polish centers using L-BFGS-B with increasing beta."""
        current_x = x.copy()
        current_y = y.copy()
        
        n_betas = 5
        for bi in range(n_betas):
            if time.time() > DEADLINE - 5:
                break
            beta = beta_start + (beta_end - beta_start) * bi / max(n_betas - 1, 1)
            params = np.concatenate([current_x, current_y])
            
            result = minimize(
                smooth_objective_and_grad,
                params,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': max_iters // n_betas, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < smooth_objective_and_grad(params, beta)[0]:
                current_x = result.x[:N]
                current_y = result.x[N:]
        
        return current_x, current_y

    def local_search(x, y, max_iters=1000):
        """Discrete local search: try moving each circle in small steps."""
        x = x.copy()
        y = y.copy()
        r = compute_radii(x, y)
        best_val = np.sum(r)
        
        rng = np.random.RandomState(42)
        step_sizes = [0.02, 0.01, 0.005, 0.002]
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            for i in range(N):
                ss = step_sizes[min(it // 250, len(step_sizes) - 1)]
                for dx, dy in [(ss, 0), (-ss, 0), (0, ss), (0, -ss),
                               (ss*0.707, ss*0.707), (-ss*0.707, ss*0.707),
                               (ss*0.707, -ss*0.707), (-ss*0.707, -ss*0.707)]:
                    x_new = x.copy()
                    y_new = y.copy()
                    x_new[i] = np.clip(x[i] + dx, 0, 1)
                    y_new[i] = np.clip(y[i] + dy, 0, 1)
                    r_new = compute_radii(x_new, y_new)
                    val_new = np.sum(r_new)
                    if val_new > best_val + 1e-10:
                        x, y, r, best_val = x_new, y_new, r_new, val_new
                        improved = True
                        break
            if not improved:
                break
        
        return x, y, compute_radii(x, y)

    # Generate diverse initial configurations
    def init_grid(n_per_side=5, jitter=0.0):
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(n_per_side):
            for j in range(n_per_side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / n_per_side
                y[idx] = (j + 0.5) / n_per_side
                if jitter > 0:
                    x[idx] += np.random.uniform(-jitter, jitter)
                    y[idx] += np.random.uniform(-jitter, jitter)
                    x[idx] = np.clip(x[idx], 0.01, 0.99)
                    y[idx] = np.clip(y[idx], 0.01, 0.99)
                idx += 1
        return x, y

    def init_spiral():
        x = np.zeros(N)
        y = np.zeros(N)
        # Spiral from center outward
        cx, cy = 0.5, 0.5
        theta = 0
        for i in range(N):
            r = 0.05 + 0.08 * i
            theta = i * 2.4  # Golden angle
            x[i] = cx + r * np.cos(theta)
            y[i] = cy + r * np.sin(theta)
            x[i] = np.clip(x[i], 0.02, 0.98)
            y[i] = np.clip(y[i], 0.02, 0.98)
        return x, y

    def init_random(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        return x, y

    # Best solution tracking
    best_x, best_y, best_r = None, None, None
    best_val = 0.0

    def update_best(x, y, r):
        nonlocal best_x, best_y, best_r, best_val
        if check_validity(x, y, r):
            val = np.sum(r)
            if val > best_val:
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                best_val = val

    # Initial configurations to try
    configs = []
    configs.append(('grid5', init_grid(5, 0.0)))
    configs.append(('grid4', init_grid(4, 0.0)))
    configs.append(('grid5j', init_grid(5, 0.05)))
    configs.append(('spiral', init_spiral()))
    configs.append(('rand1', init_random(1)))
    configs.append(('rand2', init_random(42)))
    configs.append(('rand3', init_random(123)))

    # Process each configuration
    for name, (x0, y0) in configs:
        if time.time() > DEADLINE - 15:
            break

        # Step 1: Polish centers with gradient method
        x_p, y_p = polish(x0, y0)

        # Step 2: Compute radii
        r_p = compute_radii(x_p, y_p)

        # Step 3: Grow radii
        r_grown = grow_radii(x_p, y_p, r_p)

        # Step 4: Local search
        x_ls, y_ls, r_ls = local_search(x_p, y_p)

        # Step 5: Grow radii after local search
        r_ls_grown = grow_radii(x_ls, y_ls, r_ls)

        # Update best with all variants
        update_best(x_p, y_p, r_p)
        update_best(x_p, y_p, r_grown)
        update_best(x_ls, y_ls, r_ls)
        update_best(x_ls, y_ls, r_ls_grown)

    # Final safety check
    if best_x is None or not check_validity(best_x, best_y, best_r):
        # Fallback to simple grid
        x, y = init_grid(5, 0.0)
        r = compute_radii(x, y)
        r = grow_radii(x, y, r)
        if check_validity(x, y, r):
            best_x, best_y, best_r = x, y, r
        else:
            # Very conservative fallback
            side = 5
            best_x = np.array([(i + 0.5) / side for i in range(5) for _ in range(5)][:N])
            best_y = np.array([(j + 0.5) / side for j in range(5) for _ in range(5)][:N])
            best_r = np.full(N, 0.5 / side - 1e-4)

    return np.column_stack((best_x, best_y, best_r))
