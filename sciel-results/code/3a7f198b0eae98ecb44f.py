import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def initial_grid(n):
        """Create initial grid placement with slightly smaller radii for feasibility."""
        side = int(np.ceil(np.sqrt(n)))
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.zeros(n)
        for i in range(n):
            row = i // side
            col = i % side
            xs[i] = (col + 0.5) / side
            ys[i] = (row + 0.5) / side
            rs[i] = 0.5 / side - 1e-4
        return xs, ys, rs

    def check_validity(xs, ys, rs, tol=1e-9):
        """Check if circles are valid (inside square, non-overlapping)."""
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > 1.0 - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > 1.0 - rs[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist * min_dist - tol * (min_dist * min_dist):
                    return False
        return True

    def pbd_relax(xs, ys, rs, max_iter=100, damping=1.0):
        """
        Position-based dynamics relaxation: project onto boundary and non-overlap constraints.
        Returns (xs, ys, converged).
        """
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            moved = False

            # Boundary constraints
            for i in range(N):
                r = rs[i]
                if xs[i] < r:
                    xs[i] = r
                    moved = True
                elif xs[i] > 1.0 - r:
                    xs[i] = 1.0 - r
                    moved = True
                if ys[i] < r:
                    ys[i] = r
                    moved = True
                elif ys[i] > 1.0 - r:
                    ys[i] = 1.0 - r
                    moved = True

            # Pairwise non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[j] - xs[i]
                    dy = ys[j] - ys[i]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rs[i] + rs[j]

                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-15 else 1e-8
                        overlap = min_dist - dist
                        if dist < 1e-9:
                            angle = np.random.rand() * 2 * np.pi
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist

                        push = overlap * 0.5 * 1.0
                        xs[i] -= dx * push * damping
                        ys[i] -= dy * push * damping
                        xs[j] += dx * push * damping
                        ys[j] += dy * push * damping
                        moved = True

            if not moved:
                return xs, ys, True

        return xs, ys, False

    def smooth_objective(vars_, beta=50.0):
        """
        Smooth objective using log-sum-exp for penalty terms.
        vars_ layout: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        Returns (objective, gradient).
        """
        n = N
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]

        # Smooth penalty for boundary violations
        h_left = r - x
        h_right = x + r - 1
        h_bottom = r - y
        h_top = y + r - 1

        # Overlap violations
        dx_mat = np.zeros((n, n))
        dy_mat = np.zeros((n, n))
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx_mat[i, j] = x[i] - x[j]
                    dy_mat[i, j] = y[i] - y[j]
                    dist_mat[i, j] = np.sqrt(dx_mat[i, j]**2 + dy_mat[i, j]**2 + 1e-30)
                    if dist_mat[i, j] < 1e-15:
                        dist_mat[i, j] = 1e-15

        overlap_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlap_mat[i, j] = r[i] + r[j] - dist_mat[i, j]

        def soft_plus(h):
            h_clipped = np.clip(h, -50, 50)
            return np.log1p(np.exp(beta * h_clipped)) / beta

        def soft_plus_grad(h):
            h_clipped = np.clip(h, -50, 50)
            return np.exp(beta * h_clipped) / (1 + np.exp(beta * h_clipped))

        # Boundary penalty
        h_boundary = np.concatenate([h_left, h_right, h_bottom, h_top])
        boundary_penalty = np.sum(soft_plus(h_boundary))

        # Overlap penalty
        h_overlap = []
        for i in range(n):
            for j in range(i + 1, n):
                h_overlap.append(overlap_mat[i, j])
        h_overlap = np.array(h_overlap)
        overlap_penalty = np.sum(soft_plus(h_overlap))

        obj = -np.sum(r) + 10.0 * (boundary_penalty + overlap_penalty)

        # Gradient
        grad = np.zeros(3 * n)

        for i in range(n):
            grad[i * 3 + 2] = -1.0

        bp_grad = 10.0 * soft_plus_grad(h_boundary)
        for i in range(n):
            grad[i * 3 + 0] += bp_grad[i] * (-1.0)
            grad[i * 3 + 2] += bp_grad[i] * (1.0)
            grad[i * 3 + 0] += bp_grad[n + i] * (1.0)
            grad[i * 3 + 2] += bp_grad[n + i] * (1.0)
            grad[i * 3 + 1] += bp_grad[2 * n + i] * (-1.0)
            grad[i * 3 + 2] += bp_grad[2 * n + i] * (1.0)
            grad[i * 3 + 1] += bp_grad[3 * n + i] * (1.0)
            grad[i * 3 + 2] += bp_grad[3 * n + i] * (1.0)

        op_grad = 10.0 * soft_plus_grad(h_overlap)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                g = op_grad[idx]
                d_dist_dx = dx_mat[i, j] / dist_mat[i, j]
                d_dist_dy = dy_mat[i, j] / dist_mat[i, j]

                grad[i * 3 + 0] += g * (-d_dist_dx)
                grad[i * 3 + 1] += g * (-d_dist_dy)
                grad[i * 3 + 2] += g * 1.0
                grad[j * 3 + 0] += g * (d_dist_dx)
                grad[j * 3 + 1] += g * (d_dist_dy)
                grad[j * 3 + 2] += g * 1.0
                idx += 1

        return obj, grad

    def scipy_optimize_smooth(positions, beta=50.0, maxiter=300):
        """
        Use scipy.optimize.minimize with L-BFGS-B and smooth objective.
        """
        n = len(positions)
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[i * 3 + 0] = positions[i][0]
            x0[i * 3 + 1] = positions[i][1]
            x0[i * 3 + 2] = positions[i][2]

        bounds = []
        for i in range(n):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((1e-6, 0.5))

        try:
            result = minimize(
                lambda v: smooth_objective(v, beta),
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'gtol': 1e-10}
            )

            opt_vars = result.x
            opt_positions = np.zeros((n, 3))
            for i in range(n):
                opt_positions[i][0] = opt_vars[i * 3 + 0]
                opt_positions[i][1] = opt_vars[i * 3 + 1]
                opt_positions[i][2] = opt_vars[i * 3 + 2]

            xs_new = opt_positions[:, 0]
            ys_new = opt_positions[:, 1]
            rs_new = opt_positions[:, 2]

            if check_validity(xs_new, ys_new, rs_new):
                return xs_new, ys_new, rs_new
            else:
                return positions[:, 0], positions[:, 1], positions[:, 2]

        except Exception:
            return positions[:, 0], positions[:, 1], positions[:, 2]

    def grow_and_relax(xs, ys, rs, growth_factor, max_iter=60):
        """
        Try to grow all radii by growth_factor, then relax.
        Returns (xs, ys, rs, success).
        """
        new_rs = rs * growth_factor

        if np.max(new_rs) > 0.5:
            return xs, ys, rs, False

        rs_new = new_rs.copy()
        xs_new = xs.copy()
        ys_new = ys.copy()

        xs_new, ys_new, converged = pbd_relax(xs_new, ys_new, rs_new, max_iter)

        if converged:
            if check_validity(xs_new, ys_new, rs_new):
                return xs_new, ys_new, rs_new, True

        return xs, ys, rs, False

    def local_refine(xs, ys, rs, time_limit=30.0):
        """
        Local refinement using greedy growth and relaxation.
        """
        refine_start = time.time()
        growth = 1.001
        min_growth = 1.0001

        best_score = np.sum(rs)
        best_xs = xs.copy()
        best_ys = ys.copy()
        best_rs = rs.copy()

        while time.time() < deadline - 5 and time.time() - refine_start < time_limit:
            new_xs, new_ys, new_rs, success = grow_and_relax(xs, ys, rs, growth, max_iter=40)

            if success:
                current_score = np.sum(new_rs)
                if current_score > best_score:
                    best_score = current_score
                    best_xs = new_xs.copy()
                    best_ys = new_ys.copy()
                    best_rs = new_rs.copy()
                xs, ys, rs = new_xs, new_ys, new_rs
                growth = min(1.003, growth * 1.005)
            else:
                growth *= 0.95
                if growth < min_growth:
                    break

        return best_xs, best_ys, best_rs, best_score

    # --- Main optimization loop ---

    # Phase 1: Initial grid setup
    xs, ys, rs = initial_grid(N)
    xs, ys, _ = pbd_relax(xs, ys, rs, max_iter=100)

    best_score = np.sum(rs)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_rs = rs.copy()

    # Phase 2: Greedy growth
    growth_factor = 1.002
    min_growth = 1.0001
    growth_deadline = deadline - 80

    while time.time() < growth_deadline:
        xs, ys, rs, success = grow_and_relax(xs, ys, rs, growth_factor, max_iter=50)

        if success:
            current_score = np.sum(rs)
            if current_score > best_score:
                best_score = current_score
                best_xs = xs.copy()
                best_ys = ys.copy()
                best_rs = rs.copy()
            growth_factor = min(1.004, growth_factor * 1.01)
        else:
            growth_factor *= 0.93
            if growth_factor < min_growth:
                break

    # Phase 3: Smooth optimization with L-BFGS-B to escape local minima
    # Try multiple beta values for robustness
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > deadline - 30:
            break

        positions = np.column_stack((best_xs, best_ys, best_rs))
        opt_xs, opt_ys, opt_rs = scipy_optimize_smooth(positions, beta=beta, maxiter=200)

        opt_score = np.sum(opt_rs)
        if opt_score > best_score:
            best_score = opt_score
            best_xs = opt_xs.copy()
            best_ys = opt_ys.copy()
            best_rs = opt_rs.copy()

    # Phase 4: Local refinement
    local_xs, local_ys, local_rs, local_score = local_refine(
        best_xs, best_ys, best_rs,
        time_limit=30.0
    )

    if local_score > best_score:
        best_score = local_score
        best_xs = local_xs.copy()
        best_ys = local_ys.copy()
        best_rs = local_rs.copy()

    # Phase 5: Final high-precision polish with smooth optimization
    if time.time() < deadline - 10:
        positions = np.column_stack((best_xs, best_ys, best_rs))
        opt_xs, opt_ys, opt_rs = scipy_optimize_smooth(positions, beta=200.0, maxiter=100)

        opt_score = np.sum(opt_rs)
        if opt_score > best_score:
            best_score = opt_score
            best_xs = opt_xs.copy()
            best_ys = opt_ys.copy()
            best_rs = opt_rs.copy()

    # Final validation
    if not check_validity(best_xs, best_ys, best_rs):
        # Fallback: return a valid grid
        xs_f, ys_f, rs_f = initial_grid(N)
        xs_f, ys_f, _ = pbd_relax(xs_f, ys_f, rs_f, max_iter=200)
        best_xs, best_ys, best_rs = xs_f, ys_f, rs_f

    return np.column_stack((best_xs, best_ys, best_rs))
