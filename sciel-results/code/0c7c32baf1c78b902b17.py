"""
Packing 200 equal circles in a unit square [-1/2, 1/2]^2.
Strategy:
1. Initialize with a hexagonal pattern, then relax using gradient descent on repulsive energy.
2. Minimize repulsive energy via L-BFGS-B.
3. Refine via a guided local search that targets the bottleneck constraints (min pair distance or min boundary distance).
4. Final high-precision polish.
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

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)

def _energy_and_grad(params):
    """
    Compute smooth repulsive energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^2 + sum_i sum_wall 1/d_iw^2
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)

    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)

    E_pair = np.triu(inv_d2, 1).sum()

    grad_x = np.zeros(N)
    grad_y = np.zeros(N)

    inv_d4 = inv_d2**2
    np.fill_diagonal(inv_d4, 0.0)
    
    grad_x -= 2.0 * (dx * inv_d4).sum(axis=1)
    grad_y -= 2.0 * (dy * inv_d4).sum(axis=1)

    eps = 1e-8
    dx_w = np.clip(0.5 - x, eps, None)
    dx_w_neg = np.clip(0.5 + x, eps, None)
    dy_w = np.clip(0.5 - y, eps, None)
    dy_w_neg = np.clip(0.5 + y, eps, None)

    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()

    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3

    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])

    return total_E, total_grad

def _initialize_hexagonal_relaxed():
    """
    Initialize with a hexagonal pattern and relax it using a few steps of gradient descent.
    This breaks the symmetry of the perfect lattice and moves points into a better basin.
    """
    # Approximate spacing
    d = 0.07
    points = []
    row_spacing = d * np.sqrt(3) / 2
    x_offset = d / 2

    y = -0.5 + d / 2
    row = 0
    while y <= 0.5 - d / 2:
        x = -0.5 + d / 2
        if row % 2 == 1:
            x += x_offset
        while x <= 0.5 - d / 2:
            points.append([x, y])
            x += d
        y += row_spacing
        row += 1

    points = np.array(points)
    if len(points) < N:
        # Fallback to grid if hex doesn't fit enough
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c = grid[:N].copy()
    else:
        c = points[:N].copy()
    
    c = np.clip(c, -0.49, 0.49)
    
    # Relaxation: Run a few steps of gradient descent
    # We use a simple gradient step with adaptive learning rate
    lr = 1e-4
    for _ in range(50):
        _, grad = _energy_and_grad(c.flatten())
        c -= lr * grad.reshape(N, 2)
        c = np.clip(c, -0.49, 0.49)
        # Decay learning rate
        lr *= 0.95
        
    return c

def _local_polish(c, max_iters=2000, time_limit=30.0):
    """
    Guided local search to maximize minimum distance.
    Focuses on the current bottleneck (min pair or min boundary).
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    stall_count = 0
    max_stall = 50
    step = 1e-4

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        min_val = np.min(dist_matrix)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_val = np.min(bdists)
        b_idx = np.argmin(bdists)

        improved = False

        if min_val / 2.0 < min_boundary_val:
            # Move i and j apart
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm

                # Try multiple step sizes
                for s in [step, step * 0.5, step * 2.0, step * 0.1]:
                    new_ci = c[i] + unit_vec * s
                    new_cj = c[j] - unit_vec * s

                    if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                        old_ci, old_cj = c[i].copy(), c[j].copy()
                        c[i] = new_ci
                        c[j] = new_cj
                        new_r = _compute_radius(c)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            improved = True
                            break
                        else:
                            c[i] = old_ci
                            c[j] = old_cj

                if not improved:
                    # Try random perturbation of the pair to escape local minima
                    for _ in range(2):
                        rand_dir = np.random.randn(2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        new_ci = c[i] + rand_dir * step * 0.5
                        new_cj = c[j] - rand_dir * step * 0.5
                        if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                            old_ci, old_cj = c[i].copy(), c[j].copy()
                            c[i] = new_ci
                            c[j] = new_cj
                            new_r = _compute_radius(c)
                            if new_r > current_r + 1e-12:
                                current_r = new_r
                                improved = True
                                break
                            else:
                                c[i] = old_ci
                                c[j] = old_cj

        else:
            # Move boundary circle inwards
            x, y = c[b_idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5

            min_d = min(d_left, d_right, d_down, d_up)

            for s in [step, step * 0.5, step * 2.0]:
                if min_d == d_left:
                    new_pt = c[b_idx] + np.array([s, 0])
                elif min_d == d_right:
                    new_pt = c[b_idx] + np.array([-s, 0])
                elif min_d == d_down:
                    new_pt = c[b_idx] + np.array([0, s])
                else:
                    new_pt = c[b_idx] + np.array([0, -s])

                if np.all(np.abs(new_pt) < 0.5):
                    old_pt = c[b_idx].copy()
                    c[b_idx] = new_pt
                    new_r = _compute_radius(c)
                    if new_r > current_r + 1e-12:
                        current_r = new_r
                        improved = True
                        break
                    else:
                        c[b_idx] = old_pt

        if improved:
            stall_count = 0
            step = min(step * 1.05, 1e-3)
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            stall_count += 1
            step = max(step * 0.8, 1e-6)
            if stall_count > max_stall:
                break

    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Initialize and Relax
    c_init = _initialize_hexagonal_relaxed()
    best_c = c_init.copy()
    best_r = _compute_radius(best_c)

    # 2. Main Optimization with L-BFGS-B
    if time.time() < deadline - 30:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)
            res = minimize(
                _energy_and_grad,
                best_c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_r = r_opt
                best_c = c_opt.copy()
        except Exception:
            pass

    # 3. Local Polish
    if time.time() < deadline - 5:
        time_for_polish = min(30.0, deadline - time.time() - 5)
        c_polished, r_polished = _local_polish(best_c, max_iters=5000, time_limit=time_for_polish)
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2:
        time_for_final = min(5.0, deadline - time.time() - 2)
        c_final, r_final = _local_polish(best_c, max_iters=2000, time_limit=time_for_final)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()

    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
