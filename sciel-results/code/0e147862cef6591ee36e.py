"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines a violation-energy landscape (inspired by the high-scoring seed) with
iterative radius-increase, multi-start, and a contact-graph refinement stage.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _softmin_obj_and_grad(x, beta=80.0):
    """
    Smooth soft-min surrogate for the radius objective.
    Maximizes the soft-min of all constraint values (pairwise/2 and wall distances).
    Returns (negative soft-min, gradient) for minimization.
    """
    c = x.reshape(N, 3)

    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    # Get all pairwise distances (upper triangle)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_radii = pair_dists / 2.0

    # Wall distances for each sphere
    wall_dists = HALF - np.max(np.abs(c), axis=1)

    # Combine all constraint values
    all_vals = np.concatenate([pair_radii, wall_dists])
    n = len(all_vals)

    # Soft-min via log-sum-exp: min_i v_i ≈ -1/beta * log(sum(exp(-beta * v_i)))
    # For numerical stability, shift by min
    v_min = np.min(all_vals)
    shifted = all_vals - v_min
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = v_min - log_sum / beta

    # Gradient of soft-min w.r.t. x
    # d(soft_min)/dx = -1/beta * sum(exp(-beta*v_i)/sum(exp(-beta*v_j))) * dv_i/dx
    weights = exp_vals / np.sum(exp_vals)
    weights = weights / beta  # factor 1/beta

    # Gradient for pairwise radii: d(pair_r)/dx_i = (c_i - c_j) / (2 * dist_ij)
    grad = np.zeros((N, 3))
    for idx, (ii, jj) in enumerate(zip(iu[0], iu[1])):
        d = c[ii] - c[jj]
        dist = pair_dists[idx]
        if dist > 1e-12:
            g = d / (2.0 * dist)
            grad[ii] += weights[idx] * g
            grad[jj] -= weights[idx] * g

    # Gradient for wall distances: d(wall_i)/dx_ik = -sign(c_ik) for the max component
    for i in range(N):
        abs_c = np.abs(c[i])
        k_max = np.argmax(abs_c)
        w_idx = len(pair_radii) + i
        grad[i, k_max] -= weights[w_idx] * np.sign(c[i, k_max])

    return -soft_min_val, grad.flatten()


def _epigraph_obj_and_grad(x, beta=80.0):
    """
    Epigraph formulation: minimize t subject to all constraints >= t.
    Uses smooth approximation. x = [centers (N*3), t].
    """
    c = x[:N*3].reshape(N, 3)
    t = x[N*3]

    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)

    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_radii = pair_dists / 2.0

    wall_dists = HALF - np.max(np.abs(c), axis=1)

    all_vals = np.concatenate([pair_radii, wall_dists])
    n = len(all_vals)

    # Smooth violation: sum of softplus(t - v_i)
    # softplus(z) = log(1 + exp(z)), derivative = sigmoid(z)
    z = t - all_vals
    # Clip for numerical stability
    z_clipped = np.clip(z, -50, 50)
    softplus_vals = np.log1p(np.exp(z_clipped))
    energy = np.sum(softplus_vals)

    # Gradient
    sig = 1.0 / (1.0 + np.exp(-z_clipped))  # sigmoid
    weights = sig  # d(softplus)/dz = sigmoid(z)

    grad = np.zeros((N*3 + 1,))

    # Gradient w.r.t. t: sum of sigmoid(z_i)
    grad[N*3] = np.sum(weights)

    # Gradient w.r.t. centers: -sum_i sigmoid(z_i) * dv_i/dx
    grad_c = np.zeros((N, 3))
    for idx, (ii, jj) in enumerate(zip(iu[0], iu[1])):
        d = c[ii] - c[jj]
        dist = pair_dists[idx]
        if dist > 1e-12:
            g = d / (2.0 * dist)
            grad_c[ii] -= weights[idx] * g
            grad_c[jj] += weights[idx] * g

    for i in range(N):
        abs_c = np.abs(c[i])
        k_max = np.argmax(abs_c)
        w_idx = len(pair_radii) + i
        grad_c[i, k_max] += weights[w_idx] * np.sign(c[i, k_max])

    grad[:N*3] = grad_c.flatten()

    return energy, grad

def _local_optimize_violation(init_pts, target_r, t_start, deadline, max_iter=2000):
    """
    L-BFGS-B minimization of the violation energy for a fixed target radius.
    Uses the global deadline to bound runtime.
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _energy_and_grad(x, target_r)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts


def _local_optimize_softmin(init_pts, deadline, beta=80.0, max_iter=3000):
    """
    L-BFGS-B minimization of the soft-min surrogate (maximize radius).
    """
    def objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _softmin_obj_and_grad(x, beta)

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)

    try:
        result = minimize(
            objective,
            init_pts.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        opt_pts = result.x.reshape(N, 3)
        return opt_pts
    except TimeoutError:
        return init_pts
    except Exception:
        return init_pts


def _polish_high_precision(centers, deadline):
    """
    High-precision polish: use epigraph formulation with very tight tolerances
    and multiple restarts to squeeze out the last bit of radius.
    """
    best = centers.copy()
    best_r = _compute_radius(centers)
    remaining = deadline - time.time()
    if remaining < 2.0:
        return best, best_r

    # Multiple polish attempts with different betas
    for beta in [100.0, 200.0, 500.0]:
        if time.time() > deadline - 1.0:
            break
        def epig_obj(x):
            if time.time() > deadline:
                raise TimeoutError()
            return _epigraph_obj_and_grad(x, beta)

        x0 = np.concatenate([best.flatten(), [best_r]])
        bounds = [(-HALF + 1e-12, HALF - 1e-12)] * (N * 3) + [(0.0, HALF)]
        try:
            result = minimize(
                epig_obj,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_centers = result.x[:N*3].reshape(N, 3)
            new_centers = np.clip(new_centers, -HALF + 1e-12, HALF - 1e-12)
            r_new = _compute_radius(new_centers)
            if r_new > best_r:
                best = new_centers
                best_r = r_new
        except (TimeoutError, Exception):
            pass

    return best, best_r

def _generate_grid():
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _generate_bcc():
    points = []
    for ix in range(5):
        for iy in range(5):
            for iz in range(4):
                x = (ix + 0.5) / 5.0 - 0.5
                y = (iy + 0.5) / 5.0 - 0.5
                z = (iz + 0.5) / 4.0 - 0.5
                points.append([x, y, z])
                xc = (ix + 1.0) / 5.0 - 0.5
                yc = (iy + 1.0) / 5.0 - 0.5
                zc = (iz + 1.0) / 4.0 - 0.5
                points.append([xc, yc, zc])
    return np.array(points[:N])

def _contact_graph_refine(centers, radius, deadline):
    """
    Identify near-contact pairs and refine only the involved spheres
    to maximize the minimum distance among those pairs (and wall).
    """
    if time.time() > deadline - 3.0:
        return centers, _compute_radius(centers)

    centers = centers.copy()
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.1 * radius)

    if len(pairs) == 0:
        return centers, _compute_radius(centers)

    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)

    if len(involved) < 2:
        return centers, _compute_radius(centers)

    idx_array = np.array(involved)

    def sub_objective(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        return -(min_d / 2.0 + np.min(wall))

    def sub_grad(x):
        if time.time() > deadline:
            raise TimeoutError()
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)

        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)

        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)

        grad_full = np.zeros((N, 3))

        if min_pair is not None:
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                grad_full[i] += d / (2.0 * dist)
                grad_full[j] -= d / (2.0 * dist)

        i_wall = idx_array[min_wall_idx]
        abs_c = np.abs(c[i_wall])
        k_max = np.argmax(abs_c)
        grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])

        return grad_full[idx_array].flatten()

    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF + 1e-9, HALF - 1e-9)] * (len(involved) * 3)

    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        new_sub = result.x.reshape(len(involved), 3)
        centers[idx_array] = new_sub
    except TimeoutError:
        pass
    except Exception:
        pass

    centers = np.clip(centers, -HALF, HALF)
    return centers, _compute_radius(centers)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_centers = None
    best_radius = 0.0

    rng = np.random.default_rng(42)

    # Phase 1: Multi-start with violation-energy optimization
    initializations = []

    # Grid
    grid_centers = _generate_grid()
    initializations.append(grid_centers)

    # BCC
    bcc_centers = _generate_bcc()
    initializations.append(bcc_centers)

    # Perturbed grids
    for i in range(6):
        if time.time() > deadline - 60.0:
            break
        base = _generate_grid()
        noise_scale = 0.02 + 0.01 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)

    # Random
    for i in range(4):
        if time.time() > deadline - 50.0:
            break
        random_c = rng.uniform(-0.45, 0.45, size=(N, 3))
        initializations.append(random_c)

    # Target radii to try
    target_radii = [0.105, 0.108, 0.110, 0.112, 0.114, 0.116]

    for init in initializations:
        if time.time() > deadline - 40.0:
            break

        for target_r in target_radii:
            if time.time() > deadline - 35.0:
                break

            noise = rng.normal(0, 0.003, size=(N, 3))
            init_pts = np.clip(init + noise, -0.49, 0.49)

            opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)

                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()

    # Phase 1b: Soft-min optimization from best centers found so far
    if best_centers is not None and time.time() < deadline - 30.0:
        for attempt in range(3):
            if time.time() > deadline - 25.0:
                break
            noise = rng.normal(0, 0.002, size=(N, 3))
            init_pts = np.clip(best_centers + noise, -0.49, 0.49)
            opt_pts = _local_optimize_softmin(init_pts, deadline, beta=80.0, max_iter=2000)
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                r_opt = _compute_radius(opt_pts)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = opt_pts.copy()

    # Phase 2: Iterative radius increase (squeeze)
    if best_centers is not None and time.time() < deadline - 20.0:
        num_squeezes = 20
        for i in range(num_squeezes):
            if time.time() > deadline - 12.0:
                break

            success_improved = False
            increments = [0.003, 0.002, 0.001, 0.0005, 0.0002]

            for inc in increments:
                if time.time() > deadline - 10.0:
                    break

                target_r = best_radius * (1.0 + inc)

                noise = rng.normal(0, 0.002, size=(N, 3))
                init_pts = np.clip(best_centers + noise, -0.49, 0.49)

                opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                    r_opt = _compute_radius(opt_pts)

                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_centers = opt_pts.copy()
                        success_improved = True
                        break

            if not success_improved:
                noise = rng.normal(0, 0.004, size=(N, 3))
                init_pts = np.clip(best_centers + noise, -0.49, 0.49)
                target_r = best_radius * 1.003

                opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

                if np.all(np.isfinite(opt_pts)):
                    opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
                    r_opt = _compute_radius(opt_pts)

                    if r_opt > best_radius:
                        best_radius = r_opt
                        best_centers = opt_pts.copy()

    # Phase 3: Contact-graph refinement
    if best_centers is not None and time.time() < deadline - 10.0:
        for refine_iter in range(4):
            if time.time() > deadline - 7.0:
                break
            best_centers, r_refined = _contact_graph_refine(
                best_centers, best_radius, deadline
            )
            if r_refined > best_radius:
                best_radius = r_refined
            else:
                break

    # Phase 4: High-precision polish with epigraph
    if best_centers is not None and time.time() < deadline - 5.0:
        best_centers, best_radius = _polish_high_precision(best_centers, deadline)

    # Phase 5: Final tight-target polish
    if best_centers is not None and time.time() < deadline - 3.0:
        target_r = best_radius * 1.0005
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_centers + noise, -0.49, 0.49)

        opt_pts = _local_optimize_violation(init_pts, target_r, start_time, deadline)

        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -HALF + 1e-9, HALF - 1e-9)
            r_opt = _compute_radius(opt_pts)

            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = opt_pts.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _generate_grid()

    best_centers = np.clip(best_centers, -HALF, HALF)
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers

    return _generate_grid()
