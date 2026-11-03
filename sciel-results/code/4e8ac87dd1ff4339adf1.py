"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21
EPS = 1e-9


def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)


def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)."""
    return 1.0 / (1.0 + np.exp(-np.clip(beta * x, -500, 500)))


def _objective_and_grad(x, w, h, beta=100.0, eq_weight=0.05):
    """
    Softplus-penalized objective with analytical gradients (vectorized).
    x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
    Minimizes: -sum(r) + boundary penalties + pairwise overlap penalties + equalization penalty.
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Equalization pressure: penalize variance of radii to push toward symmetric packing
    if eq_weight > 0:
        r_mean = np.mean(rs)
        r_var = np.mean((rs - r_mean) ** 2)
        obj += eq_weight * r_var
        # d/dr_i of eq_weight * mean((rs - r_mean)^2) = eq_weight * 2*(rs_i - r_mean)/N
        grad[2::3] += eq_weight * 2.0 * (rs - r_mean) / N

    # Boundary penalties
    v1 = rs - xs          # x - r < 0  ->  v1 > 0 means violation
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w      # x + r > w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys          # y - r < 0
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h      # y + r > h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs              # r < 0
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise overlap penalties (vectorized per i)
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i]     += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx]     += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    # Gradient of -sum(r)
    grad[2::3] += -1.0

    return obj, grad


def _strictly_valid(circles, w, h, tol=1e-8):
    """Full-pair strict validity check. circles: (N,3)."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _delaunay_neighbors(centers):
    """Return list of neighbor-index lists via Delaunay; fallback to all pairs."""
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [list(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]


def _fast_valid(circles, w, h, neighbors, tol=1e-9):
    """Fast validity using precomputed neighbor list (approximate, for polish steps)."""
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True


def _hex_init(w, h, seed):
    """Hexagonal lattice init over several row configurations; returns best valid (N,3)."""
    rng = random.Random(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        # Fallback grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    rng = random.Random(seed)
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.25
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _polish(circles, w, h, time_limit, beta_start=10.0, beta_end=500.0, eq_weight=0.05):
    """
    Multi-phase increasing-beta softplus optimization with Delaunay-accelerated validation.
    """
    x0 = circles.reshape(-1).copy()
    n = x0.size

    # Precompute Delaunay neighbors for fast validity checks
    centers = circles[:, :2]
    neighbors = _delaunay_neighbors(centers)

    # Phases: increasing beta
    n_phases = 4
    betas = np.geomspace(beta_start, beta_end, n_phases)
    eq_weights = [eq_weight, eq_weight * 0.5, eq_weight * 0.25, 0.0]

    best_x = x0.copy()
    best_val = -np.sum(circles[:, 2])

    for phase in range(n_phases):
        if time.time() > time_limit:
            break
        beta = betas[phase]
        eq_w = eq_weights[phase]

        def obj_fn(x):
            return _objective_and_grad(x, w, h, beta, eq_w)[0]

        def grad_fn(x):
            return _objective_and_grad(x, w, h, beta, eq_w)[1]

        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, 0.5 * min(w, h)))

        try:
            result = minimize(
                obj_fn,
                best_x,
                jac=grad_fn,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            if result.fun < best_val:
                best_x = result.x
                best_val = result.fun
        except Exception:
            pass

        # Rebuild circles and check validity
        new_circles = best_x.reshape(-1, 3)
        if _fast_valid(new_circles, w, h, neighbors):
            val = float(np.sum(new_circles[:, 2]))
            if val > -best_val:
                best_val = -val
        else:
            # Try to repair
            for i in range(N):
                x_i = new_circles[i, 0]
                y_i = new_circles[i, 1]
                r_i = new_circles[i, 2]
                new_circles[i, 0] = max(r_i, min(w - r_i, x_i))
                new_circles[i, 1] = max(r_i, min(h - r_i, y_i))
                new_circles[i, 2] = max(1e-6, min(0.5 * min(w, h), r_i))
            if _fast_valid(new_circles, w, h, neighbors):
                val = float(np.sum(new_circles[:, 2]))
                if val > -best_val:
                    best_x = new_circles.reshape(-1)
                    best_val = -val

    return best_x.reshape(-1, 3)


def _force_relax(circles, w, h, time_limit, max_iter=3000):
    """
    Force-directed relaxation: vectorized version for speed.
    """
    best_circles = circles.copy()
    best_score = _score_circles(best_circles, w, h)

    idx_i, idx_j = np.triu_indices(N, k=1)
    n_pairs = len(idx_i)

    step_size = 0.003
    patience = 0
    max_patience = 200

    for it in range(max_iter):
        if time.time() > time_limit:
            break

        x = best_circles[:, 0].copy()
        y = best_circles[:, 1].copy()
        radii = best_circles[:, 2].copy()

        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)

        # Boundary forces (vectorized)
        viol_l = np.maximum(0, radii - x)
        viol_r = np.maximum(0, x + radii - w)
        viol_b = np.maximum(0, radii - y)
        viol_t = np.maximum(0, y + radii - h)
        fx += 15.0 * viol_l - 15.0 * viol_r
        fy += 15.0 * viol_b - 15.0 * viol_t

        # Pair forces (vectorized)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq + 1e-12)
        target = radii[idx_i] + radii[idx_j]
        diff = dist - target

        # Force magnitude
        k_spring = np.where(diff < 0, 80.0, 3.0)
        force_mag = k_spring * diff

        # Direction unit vectors
        inv_dist = 1.0 / np.maximum(dist, 1e-10)
        fx_dir = dx * inv_dist
        fy_dir = dy * inv_dist

        # Apply forces to pairs using np.add.at for accumulation
        np.add.at(fx, idx_i, force_mag * fx_dir)
        np.add.at(fy, idx_i, force_mag * fy_dir)
        np.add.at(fx, idx_j, -force_mag * fx_dir)
        np.add.at(fy, idx_j, -force_mag * fy_dir)

        # Radii forces: grow circles with space, shrink overlapping ones
        fr_inc = np.where(diff > 0.005, 0.15, np.where(diff < -0.005, -0.3, 0.0))
        np.add.at(fr, idx_i, fr_inc)
        np.add.at(fr, idx_j, fr_inc)

        # Equal-radius equalization: push radii toward the mean
        mean_r = np.mean(radii)
        equalize_weight = 0.05
        fr += equalize_weight * (mean_r - radii)

        # Update
        x_new = x + step_size * fx
        y_new = y + step_size * fy
        r_new = radii + step_size * 0.5 * fr

        # Clamp
        max_r = 0.5 * min(w, h)
        x_new = np.clip(x_new, radii, w - radii)
        y_new = np.clip(y_new, radii, h - radii)
        r_new = np.clip(r_new, 1e-4, max_r)

        new_circles = np.column_stack([x_new, y_new, r_new])
        new_score = _score_circles(new_circles, w, h)

        if new_score > best_score:
            best_circles = new_circles
            best_score = new_score
            patience = 0
            step_size = min(0.008, step_size * 1.03)
        else:
            patience += 1
            step_size = max(0.0005, step_size * 0.97)

        if patience > max_patience:
            break

    return best_circles


def _score_circles(circles, w, h):
    """Score = sum of radii - penalty."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    penalty = 0.0

    # Boundary violations
    penalty += 500.0 * (np.sum(np.maximum(0, r - x)) + np.sum(np.maximum(0, x + r - w)) +
                        np.sum(np.maximum(0, r - y)) + np.sum(np.maximum(0, y + r - h)))

    # Overlap violations
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    min_dist = r[idx_i] + r[idx_j]
    overlap_mask = dist_sq < min_dist * min_dist
    if np.any(overlap_mask):
        dist = np.sqrt(dist_sq[overlap_mask])
        overlaps = min_dist[overlap_mask] - dist
        penalty += 1000.0 * np.sum(overlaps)

    return np.sum(r) - penalty


def _sa_optimize(circles, w, h, time_limit):
    """
    Simulated annealing to improve state.
    """
    best_circles = circles.copy()
    best_score = _score_circles(best_circles, w, h)

    T_start = 0.5
    T_end = 1e-5
    alpha = 0.9995
    T = T_start

    while time.time() < time_limit:
        if time.time() > time_limit - 1:
            break

        new_circles = best_circles.copy()
        var_idx = random.randint(0, 2 * N)  # 0..2N-1 for x,y,r of each circle

        if var_idx < N:
            # Perturb x of circle var_idx
            new_circles[var_idx, 0] += random.gauss(0, T / T_start * 0.1 + 0.01)
            new_circles[var_idx, 0] = max(new_circles[var_idx, 2], min(w - new_circles[var_idx, 2], new_circles[var_idx, 0]))
        elif var_idx < 2 * N:
            # Perturb y of circle var_idx-N
            ci = var_idx - N
            new_circles[ci, 1] += random.gauss(0, T / T_start * 0.1 + 0.01)
            new_circles[ci, 1] = max(new_circles[ci, 2], min(h - new_circles[ci, 2], new_circles[ci, 1]))
        else:
            # Perturb r of circle var_idx-2N
            ci = var_idx - 2 * N
            new_circles[ci, 2] += random.gauss(0, T / T_start * 0.03 + 0.003)
            max_r = 0.5 * min(w, h)
            new_circles[ci, 2] = max(1e-4, min(max_r, new_circles[ci, 2]))

        new_score = _score_circles(new_circles, w, h)

        if new_score > best_score or random.random() < np.exp((new_score - best_score) / T):
            best_circles = new_circles
            best_score = new_score

        T *= alpha
        if T < T_end:
            T = T_end

    return best_circles, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Rectangle perimeter is 4, so 2*(w+h) = 4 => w+h = 2.
    # Let w = 2 * t, h = 2 * (1-t), where t in (0, 1).

    best_circles = None
    best_val = -np.inf

    # Multiple restarts with different aspect ratios and initializations
    num_restarts = 15

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Vary aspect ratio for diversity
        t_init = 0.5 + random.uniform(-0.25, 0.25)
        w = 2.0 * t_init
        h = 2.0 * (1.0 - t_init)

        # Choose initialization strategy based on restart index
        init_strategy = restart % 3
        if init_strategy == 0:
            circles = _hex_init(w, h, seed=restart * 7 + 1)
        elif init_strategy == 1:
            circles = _grid_init(w, h, seed=restart * 11 + 3)
        else:
            circles = _random_init(w, h, seed=restart * 13 + 5)

        # Ensure circles are within bounds
        for i in range(N):
            circles[i, 0] = max(circles[i, 2], min(w - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(h - circles[i, 2], circles[i, 1]))
            circles[i, 2] = max(1e-4, min(0.5 * min(w, h), circles[i, 2]))

        # Time budget for this restart
        remaining = deadline - 30 - time.time()
        if remaining < 15:
            break

        # SA phase
        sa_time = max(5, remaining * 0.3)
        sa_deadline = time.time() + sa_time
        circles, sa_score = _sa_optimize(circles, w, h, sa_deadline)

        # Force-directed relaxation phase
        if time.time() < deadline - 20:
            force_deadline = time.time() + min(8, (deadline - 20 - time.time()) * 0.35)
            circles = _force_relax(circles, w, h, force_deadline)
            force_score = _score_circles(circles, w, h)

        # Gradient polish phase with multiple passes
        if time.time() < deadline - 10:
            for polish_iter in range(3):
                if time.time() > deadline - 8:
                    break

                remaining_polish = deadline - 8 - time.time()
                if remaining_polish < 3:
                    break

                polish_deadline = time.time() + min(remaining_polish / (3 - polish_iter), 10)
                polished = _polish(circles, w, h, polish_deadline,
                                   beta_start=10.0 + polish_iter * 10,
                                   beta_end=200.0 + polish_iter * 100,
                                   eq_weight=0.05 / (1 + polish_iter * 0.5))

                # Validate and update
                if _strictly_valid(polished, w, h, tol=1e-8):
                    val = float(np.sum(polished[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_circles = polished.copy()
                        best_w, best_h = w, h
                    circles = polished
                else:
                    # Try to repair
                    for i in range(N):
                        polished[i, 0] = max(polished[i, 2], min(w - polished[i, 2], polished[i, 0]))
                        polished[i, 1] = max(polished[i, 2], min(h - polished[i, 2], polished[i, 1]))
                        polished[i, 2] = max(1e-6, min(0.5 * min(w, h), polished[i, 2]))
                    if _strictly_valid(polished, w, h, tol=1e-8):
                        val = float(np.sum(polished[:, 2]))
                        if val > best_val:
                            best_val = val
                            best_circles = polished.copy()
                            best_w, best_h = w, h
                        circles = polished

        # Update best if current is better
        if _strictly_valid(circles, w, h, tol=1e-8):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
                best_w, best_h = w, h

    # Final high-precision polish on the best solution
    if best_circles is not None and time.time() < deadline - 5:
        # Force relaxation first to potentially find better topology
        if time.time() < deadline - 8:
            force_final = _force_relax(best_circles, best_w, best_h, time.time() + 3)
            ff_score = _score_circles(force_final, best_w, best_h)
            if _strictly_valid(force_final, best_w, best_h, tol=1e-8):
                ff_val = float(np.sum(force_final[:, 2]))
                if ff_val > best_val:
                    best_circles = force_final
                    best_val = ff_val

        # Multiple polish passes with increasing beta for high precision
        for polish_pass in range(3):
            if time.time() > deadline - 3:
                break

            remaining_time = deadline - 3 - time.time()
            if remaining_time < 2:
                break

            final_polish_deadline = time.time() + min(remaining_time, 8)
            polished = _polish(best_circles, best_w, best_h, final_polish_deadline,
                               beta_start=100.0 + polish_pass * 50,
                               beta_end=500.0 + polish_pass * 200,
                               eq_weight=0.01 / (1 + polish_pass * 0.5))

            if _strictly_valid(polished, best_w, best_h, tol=1e-8):
                val = float(np.sum(polished[:, 2]))
                if val > best_val:
                    best_circles = polished
                    best_val = val

            # Small force relaxation between polish passes to escape local minima
            if polish_pass < 2 and time.time() < deadline - 4:
                force_mid = _force_relax(best_circles, best_w, best_h, time.time() + 2)
                if _strictly_valid(force_mid, best_w, best_h, tol=1e-8):
                    fm_val = float(np.sum(force_mid[:, 2]))
                    if fm_val > best_val:
                        best_circles = force_mid
                        best_val = fm_val

    # Final validity check and repair if needed
    if best_circles is None:
        w, h = 1.0, 1.0
        best_circles = _hex_init(w, h, seed=0)
    elif not _strictly_valid(best_circles, best_w, best_h, tol=1e-7):
        # Try to repair by shrinking radii and re-polishing
        repair = best_circles.copy()
        repaired = False
        for shrink in range(30):
            if _strictly_valid(repair, best_w, best_h, tol=1e-7):
                repaired = True
                break
            for i in range(N):
                repair[i, 2] *= 0.97
            # Quick reposition
            for i in range(N):
                r_r = repair[i, 2]
                x_r = repair[i, 0]
                y_r = repair[i, 1]
                x_r = max(r_r, min(best_w - r_r, x_r))
                y_r = max(r_r, min(best_h - r_r, y_r))
                repair[i, 0] = x_r
                repair[i, 1] = y_r
        if repaired:
            best_circles = repair
        else:
            w, h = 1.0, 1.0
            best_circles = _hex_init(w, h, seed=0)

    # Extract result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_circles[i, 0]
        result[i, 1] = best_circles[i, 1]
        result[i, 2] = best_circles[i, 2]

    # Final clamp to ensure validity
    for i in range(N):
        x = result[i, 0]
        y = result[i, 1]
        r = result[i, 2]

        if x - r < 0:
            x = r
        if x + r > best_w:
            x = best_w - r
        result[i, 0] = max(0, min(best_w, x))

        if y - r < 0:
            y = r
        if y + r > best_h:
            y = best_h - r
        result[i, 1] = max(0, min(best_h, y))

        max_r = 0.5 * min(best_w, best_h)
        result[i, 2] = max(1e-6, min(max_r, r))

    return result
