"""
Packing 200 equal circles in a unit square.
Strategy (combined):
1. Hexagonal lattice initialization (finer search).
2. Global + annealed optimization using a well-conditioned log-sum-exp soft-min
   surrogate (L-BFGS-B) with rigid-body perturbations to escape local minima.
3. High-precision polish via an EPIGRAPH SLSQP solve: maximize auxiliary radius t
   subject to smooth inequality constraints |c_i-c_j| - 2t >= 0 (active pairs) and
   0.5 - |c_ik| - t >= 0 (boundaries). This is the precise local solve.
4. Final micro nudge of active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)


def _soft_min_energy(params, beta=200.0):
    """
    Smooth surrogate approximating -soft_min(all constraint distances).
    E = (1/beta) * log(sum exp(-beta d_i))  ->  minimizing E maximizes the soft-min.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)

    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]

    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    d_bounds = np.concatenate([d_left, d_right, d_down, d_up])

    all_d = np.concatenate([d_pairs, d_bounds])
    d_min = np.min(all_d)
    shifted = all_d - d_min
    weights = np.exp(-beta * shifted)
    sum_w = np.sum(weights)

    E = d_min - np.log(sum_w) / beta
    p = weights / sum_w

    grad_x = np.zeros(N)
    grad_y = np.zeros(N)

    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    p_pairs = p[:len(d_pairs)]

    grad_x[i_idx] -= p_pairs * vec_x * inv_d
    grad_x[j_idx] += p_pairs * vec_x * inv_d
    grad_y[i_idx] -= p_pairs * vec_y * inv_d
    grad_y[j_idx] += p_pairs * vec_y * inv_d

    p_bounds = p[len(d_pairs):]
    p_left = p_bounds[:N]
    p_right = p_bounds[N:2 * N]
    p_down = p_bounds[2 * N:3 * N]
    p_up = p_bounds[3 * N:]

    grad_x += -p_left + p_right
    grad_y += -p_down + p_up

    total_grad = np.concatenate([grad_x, grad_y])
    return E, total_grad


def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0

    for d in np.linspace(0.052, 0.075, 40):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is not None:
        return best_c

    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_energy(c0, beta=100.0, maxiter=300):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    try:
        res = minimize(
            lambda p: _soft_min_energy(p, beta=beta),
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 50}
        )
        return np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
    except Exception:
        return c0


def _rigid_body_perturb(c, rng, radius=0.10, max_clusters=4):
    """Move clusters of nearby circles simultaneously to preserve local structure."""
    c_new = c.copy()
    n = len(c_new)
    visited = np.zeros(n, dtype=bool)

    dx = c_new[:, np.newaxis, 0] - c_new[np.newaxis, :, 0]
    dy = c_new[:, np.newaxis, 1] - c_new[np.newaxis, :, 1]
    dist = np.sqrt(dx**2 + dy**2)

    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            j = queue.pop()
            neighbors = np.where(dist[j] < radius)[0]
            for k in neighbors:
                if not visited[k]:
                    visited[k] = True
                    cluster.append(k)
                    queue.append(k)
        clusters.append(cluster)

    if len(clusters) > max_clusters:
        idx = rng.choice(len(clusters), size=max_clusters, replace=False)
        selected_clusters = [clusters[i] for i in idx]
    else:
        selected_clusters = clusters

    for cluster in selected_clusters:
        if len(cluster) < 2:
            continue
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0.003, 0.012)
        dx_move = magnitude * np.cos(angle)
        dy_move = magnitude * np.sin(angle)
        for i in cluster:
            c_new[i, 0] += dx_move
            c_new[i, 1] += dy_move

    return np.clip(c_new, -0.49999, 0.49999)


def _slsqp_epigraph(c0, time_limit=30.0):
    """
    EPIGRAPH SLSQP solve: maximize auxiliary radius t subject to smooth
    inequality constraints on active pairs and boundaries.
    x = [t, c_0..c_{N-1}],  objective = -t.
    Constraints (>= 0):
      pair:    |c_i - c_j| - 2 t  (for pairs within ~2.5*2r of each other)
      boundary:0.5 - |c_ik| - t
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)

    def _pdist_to_ij(k, n):
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j

    for _attempt in range(3):
        if time.time() > start_time + time_limit - 3:
            break

        # Rebuild active set from current configuration
        dists = pdist(c, metric='euclidean')
        # active pairs: those within 2.5 * (2r) of each other
        close_pairs = np.where(dists < 2.5 * 2.0 * r)[0]

        constraints = []
        for k in close_pairs:
            i, j = _pdist_to_ij(k, N)

            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i * 2:1 + _i * 2 + 2]
                cj = x[1 + _j * 2:1 + _j * 2 + 2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t

            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i * 2:1 + _i * 2 + 2]
                cj = x[1 + _j * 2:1 + _j * 2 + 2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i * 2:1 + _i * 2 + 2] = diff / d
                grad[1 + _j * 2:1 + _j * 2 + 2] = -diff / d
                return grad

            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})

        for i in range(N):
            for k in range(2):
                def _bound_fun(x, _i=i, _k=k):
                    t = x[0]
                    c_val = x[1 + _i * 2 + _k]
                    return 0.5 - np.abs(c_val) - t

                def _bound_jac(x, _i=i, _k=k):
                    grad = np.zeros_like(x)
                    grad[0] = -1.0
                    c_val = x[1 + _i * 2 + _k]
                    grad[1 + _i * 2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                    return grad

                constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})

        x0 = np.concatenate([[r], c.flatten()])
        bounds = [(1e-6, 0.05)] + [(-0.49999, 0.49999)] * (2 * N)

        try:
            res = minimize(
                lambda x: -x[0],
                x0,
                method='SLSQP',
                jac=lambda x: np.concatenate([[-1.0], np.zeros(2 * N)]),
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-12, 'maxls': 80}
            )
            t_opt = res.x[0]
            c_opt = np.clip(res.x[1:].reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)

            if r_opt > r:
                r = r_opt
                c = c_opt.copy()
        except Exception:
            pass

    return c, r


def _local_polish(c, time_limit=30.0):
    """Annealed soft-min polish + epigraph SLSQP + micro nudge."""
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    rng = np.random.RandomState(42)

    # --- Annealed soft-min polish with rigid-body perturbations ---
    if time.time() - start_time < time_limit * 0.55:
        n_steps = 7
        betas = np.geomspace(150.0, 1500.0, n_steps)
        for beta in betas:
            if time.time() - start_time > time_limit * 0.55:
                break
            try:
                c_opt = _optimize_energy(c, beta=beta, maxiter=120)
                r_opt = _compute_radius(c_opt)
                if r_opt > current_r:
                    c = c_opt
                    current_r = r_opt
            except Exception:
                pass

            if time.time() - start_time > time_limit * 0.55 - 3:
                break

            c_perturbed = _rigid_body_perturb(c, rng, radius=0.10, max_clusters=4)
            r_perturbed = _compute_radius(c_perturbed)
            if r_perturbed > current_r:
                c = c_perturbed
                current_r = r_perturbed
            else:
                delta_r = current_r - r_perturbed
                T = 0.00008 * max(0.0, 1.0 - (time.time() - start_time) / time_limit)
                if delta_r < T and rng.random() < np.exp(-delta_r / max(T, 1e-10)):
                    c = c_perturbed
                    current_r = r_perturbed

    # --- Epigraph SLSQP high-precision polish (the key refinement) ---
    if time.time() - start_time < time_limit - 5:
        c_ep, r_ep = _slsqp_epigraph(c, time_limit=max(5.0, time_limit - (time.time() - start_time) - 5))
        if r_ep > current_r:
            c = c_ep
            current_r = r_ep

    # --- Final micro nudge of active constraints ---
    if time.time() - start_time < time_limit - 1:
        for _ in range(60):
            if time.time() - start_time > time_limit - 1:
                break
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, np.inf)
            bdists = np.minimum(
                np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
            )
            min_pair_dist = np.min(dist_matrix)
            min_boundary_dist = np.min(bdists)
            target_r = min(min_pair_dist / 2.0, min_boundary_dist)
            if target_r > current_r + 1e-15:
                current_r = target_r

            tol = 1e-11
            active_pairs = np.argwhere(dist_matrix < min_pair_dist + tol)
            active_bounds = np.where(bdists < min_boundary_dist + tol)[0]
            if len(active_pairs) == 0 and len(active_bounds) == 0:
                break

            forces = np.zeros((N, 2))
            for i, j in active_pairs:
                if i == j:
                    continue
                vec = c[i] - c[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-14:
                    unit_vec = vec / norm
                    forces[i] += unit_vec
                    forces[j] -= unit_vec
            for i in active_bounds:
                d_left = 0.5 - c[i, 0]
                d_right = c[i, 0] + 0.5
                d_down = 0.5 - c[i, 1]
                d_up = c[i, 1] + 0.5
                min_d = min(d_left, d_right, d_down, d_up)
                if d_left == min_d:
                    forces[i, 0] += 1.0
                elif d_right == min_d:
                    forces[i, 0] -= 1.0
                elif d_down == min_d:
                    forces[i, 1] += 1.0
                else:
                    forces[i, 1] -= 1.0

            moved = False
            for step in [5e-7, 2e-7, 1e-7, 5e-8]:
                if moved:
                    break
                for i in range(N):
                    if np.linalg.norm(forces[i]) > 1e-14:
                        direction = forces[i] / np.linalg.norm(forces[i])
                        new_pos = c[i] + direction * step
                        if np.all(np.abs(new_pos) < 0.499999):
                            old_pos = c[i].copy()
                            c[i] = new_pos
                            new_r = _compute_radius(c)
                            if new_r > current_r:
                                current_r = new_r
                                moved = True
                            else:
                                c[i] = old_pos
    return c, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150 s kill

    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)

    # Global optimization with multiple beta values
    for beta in [40.0, 80.0, 140.0]:
        if time.time() > deadline - 95:
            break
        c_opt = _optimize_energy(best_c, beta=beta, maxiter=300)
        r_opt = _compute_radius(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()

    # A couple of perturbed restarts if time allows
    if time.time() < deadline - 80:
        rng = np.random.RandomState(42)
        for _attempt in range(2):
            if time.time() > deadline - 80:
                break
            perturbation = rng.randn(N, 2) * 0.008
            c_perturbed = np.clip(best_c + perturbation, -0.4999, 0.4999)
            c_opt2 = _optimize_energy(c_perturbed, beta=120.0, maxiter=200)
            r_opt2 = _compute_radius(c_opt2)
            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()

    # Local polish (anneal + epigraph SLSQP + micro) - use most remaining time
    remaining = deadline - time.time() - 5
    if remaining > 15:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # Second polish pass if time allows
    remaining = deadline - time.time() - 3
    if remaining > 12:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()

    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    return best_c
