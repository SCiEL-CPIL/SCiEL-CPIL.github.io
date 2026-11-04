import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin from 120s limit

    def check_validity(x, y, r, tol=1e-9):
        """Check if circles are valid (non-overlapping, within bounds)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True

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
        Uses analytic gradient of the log-sum-exp soft-min."""
        x = params[:N].copy()
        y = params[N:2*N].copy()

        # Boundary constraint (soft-min over 4 boundaries)
        beta_b = 100.0
        bounds_4 = np.stack([x, 1.0-x, y, 1.0-y], axis=1)  # (N, 4)
        neg_bb = -beta_b * bounds_4
        max_bb = np.max(neg_bb, axis=1, keepdims=True)
        exp_bb = np.exp(neg_bb - max_bb)
        sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
        w_b = exp_bb / sum_bb  # (N, 4)
        r_bound = -(max_bb + np.log(sum_bb)) / beta_b
        r_bound = r_bound[:, 0]

        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-30)
        np.fill_diagonal(dist, 1e10)

        # Build constraint matrix: c_ij for all i,j
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
        objective = -np.sum(r)

        # Weights: w_ij = exp(-beta*c_ij) / sum_j' exp(-beta*c_ij')
        w = exp_terms / sum_exp  # (N, N)

        # Gradient computation
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)

        # Off-diagonal gradients of c_ij w.r.t. x_i, y_i
        dc_dx_i = 0.5 * dx / dist_safe
        dc_dy_i = 0.5 * dy / dist_safe

        # Diagonal gradients for r_bound_i (soft-min over 4 boundaries)
        grad_rb_dx = w_b[:, 0] - w_b[:, 1]
        grad_rb_dy = w_b[:, 2] - w_b[:, 3]

        # Set diagonal entries
        np.fill_diagonal(dc_dx_i, grad_rb_dx)
        np.fill_diagonal(dc_dy_i, grad_rb_dy)

        # Total gradient w.r.t. x_k:
        # dr_dx[k] = sum_i sum_j w_ij * dc_ij/dx_k
        # For fixed k, dc_ij/dx_k is nonzero only when i==k or j==k
        # dr_dx[k] = sum_{i!=k} w_ik * dc_ik/dx_k + sum_{j!=k} w_kj * dc_kj/dx_k + w_kk * dc_kk/dx_k
        # dc_ik/dx_k (j=k, so it's dc_ij/dx_j for i, j=k) = -0.5 * dx[i,k] / dist[i,k]
        # dc_kj/dx_k (i=k, so it's dc_ij/dx_i for i=k, j) = 0.5 * dx[k,j] / dist[k,j]
        # dc_kk/dx_k = grad_rb_dx[k]

        # Contribution from j=k terms: sum_{i!=k} w[i,k] * (-dc_dx_i[i,k])
        # Contribution from i=k terms: sum_{j!=k} w[k,j] * dc_dx_i[k,j]
        # Diagonal: w[k,k] * grad_rb_dx[k]

        grad_x = np.zeros(N)
        grad_y = np.zeros(N)

        for k in range(N):
            # j=k terms (i varies, i != k): w[i,k] * dc_ik/dx_k = w[i,k] * (-0.5*dx[i,k]/dist[i,k])
            # = -w[i,k] * dc_dx_i[i,k]
            jk_contrib_x = -np.sum(w[np.arange(N) != k, k] * dc_dx_i[np.arange(N) != k, k])
            jk_contrib_y = -np.sum(w[np.arange(N) != k, k] * dc_dy_i[np.arange(N) != k, k])

            # i=k terms (j varies, j != k): w[k,j] * dc_kj/dx_k = w[k,j] * dc_dx_i[k,j]
            ik_contrib_x = np.sum(w[k, np.arange(N) != k] * dc_dx_i[k, np.arange(N) != k])
            ik_contrib_y = np.sum(w[k, np.arange(N) != k] * dc_dy_i[k, np.arange(N) != k])

            # Diagonal
            diag_x = w[k, k] * grad_rb_dx[k]
            diag_y = w[k, k] * grad_rb_dy[k]

            grad_x[k] = jk_contrib_x + ik_contrib_x + diag_x
            grad_y[k] = jk_contrib_y + ik_contrib_y + diag_y

        # dr_i/dx_k contributes to d(-sum r)/dx_k = -dr_i/dx_k
        grad = np.concatenate([-grad_x, -grad_y])

        return objective, grad

    def gradient_polish(x, y, max_iter=100, beta=50.0):
        """Polish using gradient-based optimization with analytic gradients."""
        params0 = np.concatenate([x, y])

        def obj_wrapper(p):
            obj, _ = smooth_objective_and_grad(p, beta)
            return obj

        def grad_wrapper(p):
            _, g = smooth_objective_and_grad(p, beta)
            return g

        res = minimize(obj_wrapper, params0, jac=grad_wrapper, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})

        x_new = res.x[:N]
        y_new = res.x[N:2*N]
        r_new = compute_radii(x_new, y_new)

        # Clamp
        r_new = np.maximum(r_new, 1e-6)
        x_new = np.clip(x_new, r_new, 1 - r_new)
        y_new = np.clip(y_new, r_new, 1 - r_new)
        r_new = compute_radii(x_new, y_new)

        return x_new, y_new, r_new

    def sa_step(x, y, r, temp, rng):
        """One simulated annealing step: perturb and accept/reject."""
        x_new = x.copy()
        y_new = y.copy()
        r_new = r.copy()

        n_perturb = max(1, N // 4)
        idx = rng.choice(N, size=n_perturb, replace=False)

        step_size = 0.02 * temp
        x_new[idx] += rng.normal(0, step_size, size=len(idx))
        y_new[idx] += rng.normal(0, step_size, size=len(idx))

        r_change = rng.normal(0, 0.01 * temp, size=len(idx))
        r_new[idx] = np.maximum(r_new[idx] + r_change, 0.02)

        x_new = np.clip(x_new, r_new, 1 - r_new)
        y_new = np.clip(y_new, r_new, 1 - r_new)

        def energy(xv, yv, rv):
            e = 0.0
            e += np.sum(np.maximum(0, rv - xv)**2)
            e += np.sum(np.maximum(0, xv - (1 - rv))**2)
            e += np.sum(np.maximum(0, rv - yv)**2)
            e += np.sum(np.maximum(0, yv - (1 - rv))**2)
            diff_x = xv[:, None] - xv[None, :]
            diff_y = yv[:, None] - yv[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            min_dist = rv[:, None] + rv[None, :]
            np.fill_diagonal(min_dist, 0)
            viol = np.maximum(0, min_dist - dist)
            e += 10.0 * np.sum(viol**2)
            e -= np.sum(rv)
            return e

        e_old = energy(x, y, r)
        e_new = energy(x_new, y_new, r_new)

        if e_new < e_old or rng.random() < np.exp(-(e_new - e_old) / max(temp, 1e-10)):
            return x_new, y_new, r_new
        return x, y, r

    def kinetic_annealing(x_init, y_init, r_init, n_steps=2000, rng=None):
        """Kinetic annealing: circles as rigid bodies with repulsive collisions.
        Uses symplectic-like integration with adaptive damping."""
        if rng is None:
            rng = np.random.RandomState(0)

        x = x_init.copy()
        y = y_init.copy()
        r = r_init.copy()
        vx = np.zeros(N)
        vy = np.zeros(N)

        # Mass proportional to r^2
        mass = np.maximum(r**2, 1e-6)
        inv_mass = 1.0 / mass

        dt = 0.005
        damping = 0.98
        repulsion_strength = 5.0
        center_attraction = 0.01

        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_score = -np.inf

        for step in range(n_steps):
            if time.time() > deadline - 10:
                break

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            np.fill_diagonal(dist, 1.0)

            # Repulsive force when overlapping or close
            min_dist = r[:, None] + r[None, :]
            overlap = np.maximum(0, min_dist - dist)
            np.fill_diagonal(overlap, 0)

            # Force proportional to overlap (spring-like)
            force_mag = repulsion_strength * overlap / dist
            np.fill_diagonal(force_mag, 0)

            fx += np.sum(force_mag * dx / dist, axis=1)
            fy += np.sum(force_mag * dy / dist, axis=1)

            # Center attraction
            fx -= center_attraction * (x - 0.5)
            fy -= center_attraction * (y - 0.5)

            # Boundary repulsion
            fx += np.where(x < r, repulsion_strength * (r - x), 0)
            fx += np.where(x > 1 - r, repulsion_strength * (x - (1 - r)), 0)
            fy += np.where(y < r, repulsion_strength * (r - y), 0)
            fy += np.where(y > 1 - r, repulsion_strength * (y - (1 - r)), 0)

            # Update velocities (symplectic-like: kick-drift-kick)
            vx += fx * inv_mass * dt
            vy += fy * inv_mass * dt

            # Damping
            vx *= damping
            vy *= damping

            # Update positions
            x += vx * dt
            y += vy * dt

            # Clamp to bounds
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)

            # Recompute radii from positions
            r = compute_radii(x, y)

            # Track best
            if check_validity(x, y, r, tol=1e-7):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        return best_x, best_y, best_r

    def make_initial_guess():
        """Create a grid-based initial guess."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.85
        return x, y, r

    rng = np.random.RandomState(42)

    best_x, best_y, best_r = None, None, None
    best_score = -np.inf

    def update_best(x, y, r):
        nonlocal best_x, best_y, best_r, best_score
        if check_validity(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    # Phase 1: Initial gradient polish from grid
    x, y, r = make_initial_guess()
    x_p, y_p, r_p = gradient_polish(x, y, max_iter=100, beta=50.0)
    update_best(x_p, y_p, r_p)

    # Phase 2: Kinetic annealing from grid
    if time.time() < deadline - 30:
        x_k, y_k, r_k = kinetic_annealing(x, y, r, n_steps=3000, rng=rng)
        update_best(x_k, y_k, r_k)
        # Polish the kinetic result
        if check_validity(x_k, y_k, r_k, tol=1e-7):
            x_p2, y_p2, r_p2 = gradient_polish(x_k, y_k, max_iter=100, beta=50.0)
            update_best(x_p2, y_p2, r_p2)

    # Phase 3: Multiple restarts with different strategies
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        n_restarts = int(min(8, remaining_time / 12))
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            # Random initial guess
            x_init = rng.uniform(0.1, 0.9, N)
            y_init = rng.uniform(0.1, 0.9, N)
            r_init = compute_radii(x_init, y_init)

            # Quick SA
            temp = 0.5
            x_s, y_s, r_s = x_init.copy(), y_init.copy(), r_init.copy()
            sa_steps = 300
            for _ in range(sa_steps):
                if time.time() > deadline - 15:
                    break
                x_s, y_s, r_s = sa_step(x_s, y_s, r_s, temp, rng)
                temp *= 0.998

            # Gradient polish
            x_p, y_p, r_p = gradient_polish(x_s, y_s, max_iter=80, beta=50.0)
            update_best(x_p, y_p, r_p)

            # Also try kinetic annealing from this start
            if time.time() < deadline - 20:
                x_k, y_k, r_k = kinetic_annealing(x_s, y_s, r_s, n_steps=1500, rng=rng)
                update_best(x_k, y_k, r_k)
                if check_validity(x_k, y_k, r_k, tol=1e-7):
                    x_p2, y_p2, r_p2 = gradient_polish(x_k, y_k, max_iter=50, beta=50.0)
                    update_best(x_p2, y_p2, r_p2)

    # Phase 4: Final polish of best result
    if best_score > -np.inf:
        # Try with different beta values
        for beta in [30.0, 50.0, 80.0, 100.0]:
            if time.time() > deadline - 10:
                break
            x_f, y_f, r_f = gradient_polish(best_x, best_y, max_iter=100, beta=beta)
            update_best(x_f, y_f, r_f)

    # If no valid result, return safe grid
    if best_score <= -np.inf:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i // side + 0.5) / side, (i % side + 0.5) / side, 0.5 / side - 1e-3) for i in range(N)]
        return np.array(cells)

    # Final repair: shrink radii if any overlap remains
    x, y, r = best_x, best_y, best_r
    for _ in range(30):
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(min_dist, 0)
        viol = np.maximum(0, min_dist - dist)
        np.fill_diagonal(viol, 0)
        if np.max(viol) < 1e-9:
            break
        idx_i, idx_j = np.unravel_index(np.argmax(viol), viol.shape)
        reduction = viol[idx_i, idx_j] / 2 + 1e-6
        r[idx_i] = max(r[idx_i] - reduction, 1e-4)
        r[idx_j] = max(r[idx_j] - reduction, 1e-4)

    # Final clamp
    r = np.maximum(r, 1e-4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)

    # Final validity check
    if not check_validity(x, y, r, tol=1e-7):
        # Emergency: shrink all radii
        for _ in range(50):
            diff_x = x[:, None] - x[None, :]
            diff_y = y[:, None] - y[None, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            min_dist = r[:, None] + r[None, :]
            np.fill_diagonal(min_dist, 0)
            viol = np.maximum(0, min_dist - dist)
            np.fill_diagonal(viol, 0)
            if np.max(viol) < 1e-9:
                break
            r *= 0.95
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)

    return np.column_stack((x, y, r))
