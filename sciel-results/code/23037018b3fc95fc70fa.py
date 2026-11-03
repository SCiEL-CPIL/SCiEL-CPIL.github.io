"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # ---------- Helpers ----------

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii(centers):
        """
        Compute maximum non-overlapping radii from centers using
        iterative proportional scaling on pairwise + boundary constraints.
        """
        centers = np.asarray(centers)
        n = len(centers)
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)

        for _ in range(80):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    d = dists[i, j]
                    s = radii[i] + radii[j]
                    if s > d and s > 0:
                        scale = d / s
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
            if not changed:
                break
        return np.maximum(0.0, radii)

    def eval_packing(centers):
        radii = compute_radii(centers)
        return np.sum(radii), radii

    # ---------- Best-fit local search (inspiration 1 core) ----------

    def best_fit_local_search(xy, r, deadline, grid_size=100, max_sweeps=40):
        """
        Iteratively update each circle to the best position (maximizing its
        radius) given all other circles fixed. Vectorized over a grid.
        """
        n = len(xy)
        best_total, best_valid = _eval_packing(xy, r)
        best_xy = xy.copy()
        best_r = r.copy()
        if best_valid:
            best_total = np.sum(best_r)
        else:
            best_total = 0.0

        gx = np.linspace(0.0, 1.0, grid_size)
        gy = np.linspace(0.0, 1.0, grid_size)
        GX, GY = np.meshgrid(gx, gy, indexing='ij')
        candidates = np.column_stack([GX.ravel(), GY.ravel()])
        M = len(candidates)

        wall_clear = np.minimum(
            np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1])
        )

        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                others_xy = xy[mask]
                others_r = r[mask]

                if len(others_xy) > 0:
                    dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                    dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                    dists = np.sqrt(dx * dx + dy * dy)
                    circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                    r_candidates = np.minimum(wall_clear, circle_clear)
                else:
                    r_candidates = wall_clear.copy()

                idx = int(np.argmax(r_candidates))
                new_r = r_candidates[idx]
                old_r = r[i]

                if new_r > old_r + 1e-10:
                    xy[i, 0] = candidates[idx, 0]
                    xy[i, 1] = candidates[idx, 1]
                    r[i] = new_r
                    improved = True

            total, valid = _eval_packing(xy, r)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

            if not improved:
                break

        return best_xy, best_r, best_total, best_valid

    def _eval_packing(xy, r):
        n = len(xy)
        for i in range(n):
            x, y, ri = xy[i, 0], xy[i, 1], r[i]
            if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
                return 0.0, False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return 0.0, False
        return np.sum(r), True

    # ---------- Smooth gradient polish (adaptive beta) ----------

    def objective_smooth(centers_flat, beta=200.0):
        centers = centers_flat.reshape(-1, 2)
        n = len(centers)
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        half_dists = dists / 2.0
        constraints = np.empty((n, n + 3))
        constraints[:, 0] = d_left
        constraints[:, 1] = d_right
        constraints[:, 2] = d_bottom
        constraints[:, 3] = d_top
        constraints[:, 4:] = half_dists
        c_min = np.min(constraints, axis=1, keepdims=True)
        shifted = -beta * (constraints - c_min)
        shifted = np.clip(shifted, -500, 500)
        log_sum = np.log(np.sum(np.exp(shifted), axis=1))
        soft_min_vals = c_min.squeeze() - log_sum / beta
        return -np.sum(soft_min_vals)

    def gradient_polish(centers, max_iters=200, beta_start=50.0, beta_end=400.0):
        """
        Adaptive beta schedule: start smooth (topology), end sharp (refine).
        """
        try:
            x0 = centers.flatten()
            # Multi-stage with increasing beta
            stages = [(beta_start, 0.3), (150.0, 0.4), (beta_end, 0.3)]
            current_x = x0
            for beta, frac in stages:
                if time.time() > deadline - 5:
                    break
                iters = int(max_iters * frac)
                res = minimize(
                    lambda x, b=beta: objective_smooth(x, b),
                    current_x,
                    method='L-BFGS-B',
                    bounds=[(0.01, 0.99)] * (2 * N),
                    options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current_x = res.x
            opt_centers = current_x.reshape(-1, 2)
            radii = compute_radii(opt_centers)
            x_full = np.column_stack([opt_centers, radii])
            if is_valid(x_full, tol=1e-6):
                return x_full
        except Exception:
            pass
        return None

    # ---------- Contact-graph Newton refinement ----------

    def build_contact_graph(centers, radii, tol=1e-5):
        contacts = []
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if abs(d - (radii[i] + radii[j])) < tol:
                    contacts.append((i, j))
        boundary_contacts = []
        for i in range(n):
            if abs(centers[i, 0] - radii[i]) < tol:
                boundary_contacts.append((i, 'L'))
            if abs(1.0 - centers[i, 0] - radii[i]) < tol:
                boundary_contacts.append((i, 'R'))
            if abs(centers[i, 1] - radii[i]) < tol:
                boundary_contacts.append((i, 'B'))
            if abs(1.0 - centers[i, 1] - radii[i]) < tol:
                boundary_contacts.append((i, 'T'))
        return contacts, boundary_contacts

    def newton_solve_contact_system(centers, radii, max_iters=40, tol=1e-10):
        n = len(centers)
        contacts, boundary_contacts = build_contact_graph(centers, radii, tol=1e-5)

        def residual(vars):
            xs = vars[0:n]
            ys = vars[n:2 * n]
            rs = vars[2 * n:3 * n]
            res = []
            for (i, j) in contacts:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx * dx + dy * dy
                r_sum = rs[i] + rs[j]
                res.append(d_sq - r_sum * r_sum)
            for (i, side) in boundary_contacts:
                if side == 'L':
                    res.append(xs[i] - rs[i])
                elif side == 'R':
                    res.append(1.0 - xs[i] - rs[i])
                elif side == 'B':
                    res.append(ys[i] - rs[i])
                elif side == 'T':
                    res.append(1.0 - ys[i] - rs[i])
            return np.array(res)

        def jacobian(vars):
            xs = vars[0:n]
            ys = vars[n:2 * n]
            rs = vars[2 * n:3 * n]
            n_constraints = len(contacts) + len(boundary_contacts)
            J = np.zeros((n_constraints, 3 * n))
            row = 0
            for (i, j) in contacts:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                r_sum = rs[i] + rs[j]
                J[row, i] = 2 * dx
                J[row, n + i] = 2 * dy
                J[row, 2 * n + i] = -2 * r_sum
                J[row, j] = -2 * dx
                J[row, n + j] = -2 * dy
                J[row, 2 * n + j] = -2 * r_sum
                row += 1
            for (i, side) in boundary_contacts:
                if side == 'L':
                    J[row, i] = 1.0
                    J[row, 2 * n + i] = -1.0
                elif side == 'R':
                    J[row, i] = -1.0
                    J[row, 2 * n + i] = -1.0
                elif side == 'B':
                    J[row, n + i] = 1.0
                    J[row, 2 * n + i] = -1.0
                elif side == 'T':
                    J[row, n + i] = -1.0
                    J[row, 2 * n + i] = -1.0
                row += 1
            return J

        x0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            r = residual(x0)
            if np.max(np.abs(r)) < tol:
                break
            J = jacobian(x0)
            try:
                dx, _, _, _ = np.linalg.lstsq(J, -r, rcond=None)
            except:
                break
            alpha = 1.0
            for _ in range(20):
                new_x = x0 + alpha * dx
                new_r = residual(new_x)
                if np.linalg.norm(new_r) < np.linalg.norm(r):
                    break
                alpha *= 0.5
            x0 = x0 + alpha * dx

        new_centers = np.column_stack([x0[0:n], x0[n:2 * n]])
        new_radii = x0[2 * n:3 * n]
        x_full = np.column_stack([new_centers, new_radii])
        if is_valid(x_full, tol=1e-6) and np.all(new_radii > 0):
            return new_centers, new_radii
        return None

    def contact_graph_optimization(centers, radii, max_newton_iters=30):
        result = newton_solve_contact_system(centers, radii, max_iters=max_newton_iters)
        if result is not None:
            new_centers, new_radii = result
            if np.sum(new_radii) > np.sum(radii):
                return new_centers, new_radii
        return centers, radii

    # ---------- Simulated annealing ----------

    def simulated_annealing(centers, max_iters=15000, T_start=0.04, T_end=0.002, seed=0):
        rng = np.random.RandomState(seed)
        current = centers.copy()
        current_score, _ = eval_packing(current)
        best = current.copy()
        best_score = current_score
        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
            i = rng.randint(N)
            step = 0.02 * (T / T_start) ** 0.5 + 0.002
            dx = rng.normal(0, step)
            dy = rng.normal(0, step)
            new_centers = current.copy()
            new_centers[i, 0] = np.clip(current[i, 0] + dx, 0.01, 0.99)
            new_centers[i, 1] = np.clip(current[i, 1] + dy, 0.01, 0.99)
            new_score, _ = eval_packing(new_centers)
            delta = new_score - current_score
            if delta > 0 or rng.uniform() < np.exp(delta / max(T, 1e-10)):
                current = new_centers
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
        return best, best_score

    # ---------- Initialization ----------

    def grid_init(side):
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    positions.append([x, y])
        while len(positions) < N:
            positions.append([0.5, 0.5])
        return np.array(positions[:N])

    def hex_init():
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) < N:
                    x = 0.1 + i * 0.16 + (0.08 if j % 2 else 0)
                    y = 0.1 + j * 0.17
                    if x < 0.9 and y < 0.9:
                        centers.append([x, y])
        rng = np.random.RandomState(42)
        while len(centers) < N:
            centers.append(list(rng.uniform(0.1, 0.9, 2)))
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    # ---------- Main optimization ----------

    best_result = None
    best_score = -np.inf

    def update_best(x_full):
        nonlocal best_result, best_score
        if x_full is None:
            return
        score = np.sum(x_full[:, 2])
        if score > best_score:
            best_score = score
            best_result = x_full.copy()

    # Phase A: Best-fit local search from multiple initializations
    inits = []
    inits.append(grid_init(5))
    inits.append(grid_init(6))
    inits.append(grid_init(7))
    inits.append(hex_init())
    for s in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131]:
        inits.append(random_init(s))

    for idx, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        centers = init_centers.copy()
        radii = compute_radii(centers)

        # Best-fit local search
        bf_xy, bf_r, bf_total, bf_valid = best_fit_local_search(
            centers, radii, deadline - 10, grid_size=100, max_sweeps=30
        )
        if bf_valid:
            x_full = np.column_stack([bf_xy, bf_r])
            update_best(x_full)

    # Phase B: Gradient polish + Newton refinement from best
    if best_result is not None and time.time() < deadline - 20:
        for trial in range(10):
            if time.time() > deadline - 8:
                break
            rng = np.random.RandomState(500 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.01, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            # SA
            sa_centers, _ = simulated_annealing(
                perturbed, max_iters=8000, T_start=0.03, T_end=0.001,
                seed=1000 + trial
            )
            # Best-fit
            radii = compute_radii(sa_centers)
            bf_xy, bf_r, _, _ = best_fit_local_search(
                sa_centers, radii, deadline - 8, grid_size=80, max_sweeps=20
            )
            x_full = np.column_stack([bf_xy, bf_r])
            update_best(x_full)

            # Gradient polish
            polished = gradient_polish(bf_xy, max_iters=150, beta_start=50.0, beta_end=400.0)
            if polished is not None:
                update_best(polished)

            # Newton
            radii_np = compute_radii(bf_xy)
            cg_centers, cg_radii = contact_graph_optimization(bf_xy, radii_np, max_newton_iters=30)
            x_full = np.column_stack([cg_centers, cg_radii])
            update_best(x_full)

    # Phase C: Fine refinement from best
    if best_result is not None and time.time() < deadline - 10:
        for trial in range(8):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(3000 + trial)
            perturbed = best_result[:, :2].copy()
            perturbed += rng.normal(0, 0.005, size=perturbed.shape)
            perturbed = np.clip(perturbed, 0.01, 0.99)

            radii = compute_radii(perturbed)
            bf_xy, bf_r, _, _ = best_fit_local_search(
                perturbed, radii, deadline - 5, grid_size=120, max_sweeps=25
            )
            x_full = np.column_stack([bf_xy, bf_r])
            update_best(x_full)

            polished = gradient_polish(bf_xy, max_iters=100, beta_start=100.0, beta_end=500.0)
            if polished is not None:
                update_best(polished)

            radii_np = compute_radii(bf_xy)
            cg_centers, cg_radii = contact_graph_optimization(bf_xy, radii_np, max_newton_iters=40)
            x_full = np.column_stack([cg_centers, cg_radii])
            update_best(x_full)

    # Fallback
    if best_result is None:
        centers = grid_init(6)
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])

    # Final validity enforcement
    radii = compute_radii(best_result[:, :2])
    radii = np.maximum(0.0, radii - 1e-9)
    best_result = np.column_stack([best_result[:, :2], radii])
    if not is_valid(best_result, tol=1e-6):
        centers = grid_init(6)
        radii = compute_radii(centers)
        radii = np.maximum(0.0, radii - 0.001)
        best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
