"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
            return False
        if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii(centers):
        """Exact max radii for fixed centers (vectorized)."""
        centers = np.asarray(centers, dtype=float)
        n = len(centers)
        radii = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                           np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        # Vectorized pairwise
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        half_dist = dist / 2.0
        # For each circle i, the constraint from j is half_dist[i,j]
        # radii[i] = min(boundary_dist[i], min_j half_dist[i,j])
        min_pair = np.min(half_dist, axis=1)
        radii = np.minimum(radii, min_pair)
        return radii

    def smooth_radii(centers, beta=200.0):
        """Soft-min approximation of radii using log-sum-exp."""
        centers = np.asarray(centers, dtype=float)
        n = len(centers)
        # Boundary distances
        db = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        # Pairwise half-distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        half_dist = dist / 2.0

        radii = np.zeros(n)
        for i in range(n):
            constraints = np.concatenate(([db[i]], half_dist[i, :]))
            constraints = constraints[np.isfinite(constraints)]
            if len(constraints) == 0:
                radii[i] = 0.5
            else:
                m = np.min(constraints)
                # soft min: -log(sum(exp(-beta*c)))/beta
                s = np.sum(np.exp(-beta * (constraints - m)))
                radii[i] = m - np.log(s) / beta
        return radii

    def smooth_obj(x_flat):
        """Negative sum of soft radii."""
        c = x_flat.reshape(-1, 2)
        r = smooth_radii(c)
        return -np.sum(r)

    def smooth_grad(x_flat):
        """Analytical gradient of soft objective (vectorized)."""
        beta = 200.0
        c = x_flat.reshape(-1, 2)
        n = len(c)
        grad = np.zeros_like(x_flat)

        # Precompute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)

        for i in range(n):
            # Constraint values for circle i: 4 boundaries + (n-1) pairwise
            # Boundary values
            bvals = np.array([c[i, 0], 1.0 - c[i, 0], c[i, 1], 1.0 - c[i, 1]])
            # Pairwise values
            pvals = dist[i, :].copy()
            pvals[i] = np.inf
            pvals = pvals / 2.0

            all_vals = np.concatenate([bvals, pvals])
            m = np.min(all_vals)
            exps = np.exp(-beta * (all_vals - m))
            s = np.sum(exps)
            weights = -exps / s  # d(soft_min)/dx = sum_k w_k * d(constraint_k)/dx

            # Boundary gradients: each boundary constraint has a simple gradient
            # left: d/dx_i0 = 1, right: d/dx_i0 = -1, bottom: d/dx_i1 = 1, top: d/dx_i1 = -1
            grad[2 * i] += weights[0] * 1.0 + weights[1] * (-1.0)
            grad[2 * i + 1] += weights[2] * 1.0 + weights[3] * (-1.0)

            # Pairwise gradients
            for j in range(n):
                if j == i:
                    continue
                d = dist[i, j]
                if d < 1e-12:
                    continue
                w = weights[4 + j] if j < i else weights[4 + j]
                # Actually pvals is indexed 0..n-1 with pvals[i]=inf
                # pvals[j] corresponds to weight index 4+j
                w = weights[4 + j]
                # d(d_ij/2)/dx_i = (c_i - c_j) / (2 * d_ij)
                grad[2 * i] += w * (c[i, 0] - c[j, 0]) / (2.0 * d)
                grad[2 * i + 1] += w * (c[i, 1] - c[j, 1]) / (2.0 * d)
                # d(d_ij/2)/dx_j = (c_j - c_i) / (2 * d_ij)
                grad[2 * j] += w * (c[j, 0] - c[i, 0]) / (2.0 * d)
                grad[2 * j + 1] += w * (c[j, 1] - c[i, 1]) / (2.0 * d)

        # Objective is -sum(soft_radii), so gradient of objective is -gradient of sum
        return -grad

    def best_fit_local_search(centers, max_sweeps=50, tol=1e-10):
        """
        Best-fit: for each circle, find position that maximizes its radius
        given all other circles fixed. Iterate until convergence.
        """
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_sweeps):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(n):
                # Compute current radius for circle i
                others = np.delete(centers, i, axis=0)
                cur_r = min(
                    centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1]
                )
                for oc in others:
                    d = np.hypot(centers[i, 0] - oc[0], centers[i, 1] - oc[1])
                    r = d / 2.0
                    if r < cur_r:
                        cur_r = r

                # Try to improve: search on a local grid around current position
                best_r = cur_r
                best_pos = centers[i].copy()
                search_range = max(cur_r * 0.5, 0.02)
                n_grid = 7
                for dx in np.linspace(-search_range, search_range, n_grid):
                    for dy in np.linspace(-search_range, search_range, n_grid):
                        nx = centers[i, 0] + dx
                        ny = centers[i, 1] + dy
                        if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                            continue
                        r = min(nx, 1.0 - nx, ny, 1.0 - ny)
                        for oc in others:
                            d = np.hypot(nx - oc[0], ny - oc[1])
                            rr = d / 2.0
                            if rr < r:
                                r = rr
                        if r > best_r + tol:
                            best_r = r
                            best_pos = np.array([nx, ny])

                if np.linalg.norm(best_pos - centers[i]) > tol:
                    centers[i] = best_pos
                    improved = True

            if not improved:
                break

        return centers

    def simulated_annealing(centers, T0=0.1, T_end=0.001, n_iter=3000, rng=None):
        """Simulated annealing with best-fit moves."""
        if rng is None:
            rng = np.random.RandomState(0)
        centers = np.array(centers, dtype=float)
        n = len(centers)
        current_score = np.sum(compute_radii(centers))
        best_centers = centers.copy()
        best_score = current_score

        n_steps = n_iter
        for it in range(n_steps):
            T = T0 * (T_end / T0) ** (it / max(n_steps - 1, 1))
            if time.time() > deadline - 20:
                break

            # Pick a random circle and move it
            i = rng.randint(n)
            move_size = 0.02 + 0.05 * (T / T0)
            new_pos = centers[i].copy()
            new_pos[0] += rng.normal(0, move_size)
            new_pos[1] += rng.normal(0, move_size)
            new_pos = np.clip(new_pos, 0.01, 0.99)

            old_pos = centers[i].copy()
            centers[i] = new_pos
            new_score = np.sum(compute_radii(centers))

            delta = new_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = centers.copy()
            else:
                centers[i] = old_pos

        return best_centers, best_score

    def gradient_polish(centers, max_iter=500):
        """Polish using L-BFGS-B on smooth surrogate."""
        x0 = centers.flatten()
        bounds = [(0.0, 1.0)] * (2 * N)

        def obj(x):
            return smooth_obj(x)

        def grad(x):
            return smooth_grad(x)

        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12})
            c = res.x.reshape(-1, 2)
            c = np.clip(c, 0.0, 1.0)
            return c
        except Exception:
            return centers

    def tangent_solve_single(centers, i, max_iter=50):
        """
        For circle i, solve the 2D subproblem: maximize r_i subject to
        r_i <= dist to boundary and r_i <= d_ij/2 for all j != i.
        Use iterative projection + Newton on active constraints.
        """
        n = len(centers)
        pos = centers[i].copy()
        others = np.delete(centers, i, axis=0)
        other_idx = [j for j in range(n) if j != i]

        def radius_at(p):
            r = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
            for j, oc in enumerate(others):
                d = np.hypot(p[0] - oc[0], p[1] - oc[1])
                rr = d / 2.0
                if rr < r:
                    r = rr
            return r

        cur_r = radius_at(pos)

        # Adaptive multi-scale grid search
        for scale in [max(cur_r * 0.5, 0.02), max(cur_r * 0.2, 0.008), max(cur_r * 0.08, 0.002), max(cur_r * 0.03, 0.0005)]:
            if time.time() > deadline - 5:
                break
            n_g = 11
            best_r = cur_r
            best_p = pos.copy()
            for dx in np.linspace(-scale, scale, n_g):
                for dy in np.linspace(-scale, scale, n_g):
                    np_ = np.array([pos[0] + dx, pos[1] + dy])
                    if np_.min() < -0.01 or np_.max() > 1.01:
                        continue
                    r = radius_at(np_)
                    if r > best_r + 1e-14:
                        best_r = r
                        best_p = np_
            if np.linalg.norm(best_p - pos) > 1e-14:
                pos = best_p
                cur_r = best_r

        # Newton polish: find position where the top 2-3 constraints are equal
        # (active set method)
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            r = radius_at(pos)
            # Find active constraints (within tolerance)
            tol_active = 1e-10
            active = []
            # Check boundaries
            for coord, axis in [(pos[0], 'left'), (1.0 - pos[0], 'right'),
                                (pos[1], 'bottom'), (1.0 - pos[1], 'top')]:
                if abs(coord - r) < tol_active:
                    active.append(('bnd', axis))
            for j, oc in enumerate(others):
                d = np.hypot(pos[0] - oc[0], pos[1] - oc[1])
                if abs(d / 2.0 - r) < tol_active:
                    active.append(('cir', j, oc))

            if len(active) < 2:
                break

            # Try to move in direction that increases r
            # Gradient of r w.r.t. pos at current point (using subgradient of min)
            # For each active constraint, compute its gradient
            grad_sum = np.zeros(2)
            count = 0
            for a in active:
                if a[0] == 'bnd':
                    if a[1] == 'left':
                        g = np.array([1.0, 0.0])
                    elif a[1] == 'right':
                        g = np.array([-1.0, 0.0])
                    elif a[1] == 'bottom':
                        g = np.array([0.0, 1.0])
                    else:
                        g = np.array([0.0, -1.0])
                else:
                    oc = a[2]
                    d = np.hypot(pos[0] - oc[0], pos[1] - oc[1])
                    if d < 1e-12:
                        continue
                    g = np.array([(pos[0] - oc[0]) / (2.0 * d),
                                  (pos[1] - oc[1]) / (2.0 * d)])
                grad_sum += g
                count += 1

            if count == 0:
                break
            grad_sum /= count
            grad_norm = np.linalg.norm(grad_sum)
            if grad_norm < 1e-12:
                break

            # Try small steps in gradient direction
            step_sizes = [0.001, 0.0005, 0.0001, 0.00005]
            improved = False
            for ss in step_sizes:
                new_pos = pos + grad_sum / grad_norm * ss
                if new_pos.min() < -0.01 or new_pos.max() > 1.01:
                    continue
                new_r = radius_at(new_pos)
                if new_r > r + 1e-14:
                    pos = new_pos
                    cur_r = new_r
                    improved = True
                    break
            if not improved:
                break

        return pos, cur_r

    def high_precision_polish(centers, max_iter=200):
        """Very fine best-fit local search with tangent solving."""
        centers = np.array(centers, dtype=float)
        n = len(centers)

        for sweep in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                new_pos, new_r = tangent_solve_single(centers, i, max_iter=30)
                if np.linalg.norm(new_pos - centers[i]) > 1e-12:
                    centers[i] = new_pos
                    improved = True

            if not improved:
                break

        return centers

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.05, 0.95, size=(N, 2))

    best_centers = None
    best_score = -np.inf
    best_result = None

    def evaluate_and_update(centers):
        nonlocal best_centers, best_score, best_result
        radii = compute_radii(centers)
        result = np.column_stack([centers, radii])
        if is_valid(result, tol=1e-6):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_result = result
                return True
        return False

    # Phase 1: Multiple SA restarts
    seeds = [0, 1, 7, 13, 23, 37, 53, 71, 89, 42, 99, 123, 200, 300, 500, 777, 1234]

    for seed in seeds:
        if time.time() > deadline - 50:
            break

        if seed == 0:
            centers = grid_init()
        else:
            centers = random_init(seed)

        # Quick best-fit to get a decent starting point
        centers = best_fit_local_search(centers, max_sweeps=15)

        # SA
        rng = np.random.RandomState(seed + 100)
        centers_sa, _ = simulated_annealing(centers, T0=0.08, T_end=0.0005,
                                             n_iter=2500, rng=rng)

        # Best-fit polish
        centers_bf = best_fit_local_search(centers_sa, max_sweeps=30)
        evaluate_and_update(centers_bf)

    # Phase 2: Gradient polish on best
    if best_centers is not None:
        if time.time() < deadline - 30:
            centers_grad = gradient_polish(best_centers, max_iter=400)
            centers_grad = best_fit_local_search(centers_grad, max_sweeps=25)
            evaluate_and_update(centers_grad)

    # Phase 3: High-precision polish with tangent solving
    if best_centers is not None:
        if time.time() < deadline - 15:
            centers_hp = high_precision_polish(best_centers, max_iter=150)
            evaluate_and_update(centers_hp)

        # Multiple gradient + high-precision cycles
        if time.time() < deadline - 8:
            for _ in range(4):
                if time.time() > deadline - 5:
                    break
                centers_g = gradient_polish(best_centers, max_iter=300)
                centers_g = high_precision_polish(centers_g, max_iter=80)
                if not evaluate_and_update(centers_g):
                    break

    # Final: one more high-precision pass
    if best_centers is not None and time.time() < deadline - 3:
        centers_final = high_precision_polish(best_centers, max_iter=50)
        evaluate_and_update(centers_final)

    # Fallback
    if best_result is None:
        centers = grid_init()
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])

    # Final safety check
    if not is_valid(best_result, tol=1e-6):
        centers = grid_init()
        radii = compute_radii(centers)
        best_result = np.column_stack([centers, radii])

    return best_result

# EVOLVE-BLOCK-END
