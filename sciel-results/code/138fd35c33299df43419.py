"""Circle packing in perimeter-4 rectangle: Spectral graph embedding + multi-start SLSQP + polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse import csr_matrix

N = 21
_LSE_K = 300.0


def _vectorized_objective(z, penalty_weight=1000.0):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    obj += penalty_weight * np.sum(overlap * overlap)

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    obj += penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _lse(vals):
    m = np.max(vals)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(vals - m)))


def _spectral_initialization(contact_edges, w0, h0, n_circles=N, max_iter=2000, lr=0.01):
    """
    Use spectral embedding to get a good initial packing.
    contact_edges: list of (i, j) pairs representing desired contacts.
    Returns (x, y, r) arrays.
    """
    # Build weighted Laplacian from contact graph
    n = n_circles
    W_mat = np.zeros((n, n))
    for i, j in contact_edges:
        W_mat[i, j] = 1.0
        W_mat[j, i] = 1.0

    # Add wall contacts implicitly by adding boundary nodes
    # We'll use a simpler approach: spectral embedding of the contact graph
    # with boundary constraints enforced via projection

    # Compute Laplacian
    L = W_mat - W_mat  # placeholder
    # Actually use: L = D - A
    D = np.diag(W_mat.sum(axis=1))
    L = D - W_mat

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(L)

    # Use the two smallest non-trivial eigenvectors (Fiedler vectors)
    # Sort by eigenvalue
    idx = np.argsort(eigvals)
    eigvecs_sorted = eigvecs[:, idx]

    # Skip the first (trivial) eigenvector
    if n >= 3:
        v1 = eigvecs_sorted[:, 1]
        v2 = eigvecs_sorted[:, 2]
    else:
        v1 = eigvecs_sorted[:, 1] if n > 1 else np.ones(n)
        v2 = eigvecs_sorted[:, 2] if n > 2 else np.zeros(n)

    # Normalize to fit in rectangle
    x = v1 - v1.min()
    x = x / (x.max() - x.min() + 1e-12) * w0
    y = v2 - v2.min()
    y = y / (y.max() - y.min() + 1e-12) * h0

    # Initial radii: small uniform
    r = np.full(n, min(w0, h0) / np.sqrt(n) * 0.3)

    # Gradient descent on spectral embedding with radius growth
    for it in range(max_iter):
        # Compute forces from contact constraints
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w0, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h0, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise overlap forces
        for i, j in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-12
            overlap = r[i] + r[j] - d
            if overlap > 0:
                fx[i] += dx / d * overlap * 10
                fy[i] += dy / d * overlap * 10
                fx[j] -= dx / d * overlap * 10
                fy[j] -= dy / d * overlap * 10

        # All-pairs overlap check (for non-contact pairs too)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                overlap = r[i] + r[j] - d
                if overlap > 0.001:
                    fx[i] += dx / d * overlap * 5
                    fy[i] += dy / d * overlap * 5
                    fx[j] -= dx / d * overlap * 5
                    fy[j] -= dy / d * overlap * 5

        # Update positions
        x += lr * fx
        y += lr * fy
        x = np.clip(x, r, w0 - r)
        y = np.clip(y, r, h0 - r)

        # Try to grow radii
        max_r = min(w0, h0) / 2
        for i in range(n):
            r_max = min(x[i], w0 - x[i], y[i], h0 - y[i])
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.001, min(r_max, max_r))
            r[i] = min(r[i] * 1.005, r_max)

    return x, y, r


def _generate_contact_graphs():
    """Generate several plausible contact graphs for 21 circles."""
    graphs = []

    # Grid-like: 5x5 minus 4 corners
    side = 5
    grid_edges = []
    for row in range(side):
        for col in range(side):
            i = row * side + col
            if col < side - 1:
                grid_edges.append((i, i + 1))
            if row < side - 1:
                grid_edges.append((i, i + side))
    # Remove edges involving corners to make it 21 circles
    corner_indices = {0, 4, 20, 24}
    filtered_edges = []
    for i, j in grid_edges:
        if i not in corner_indices and j not in corner_indices:
            filtered_edges.append((i, j))
    # Map to 0..20
    active = [i for i in range(25) if i not in corner_indices]
    mapping = {old: new for new, old in enumerate(active)}
    remapped_edges = []
    for i, j in filtered_edges:
        if i in mapping and j in mapping:
            remapped_edges.append((mapping[i], mapping[j]))
    graphs.append(remapped_edges)

    # Hexagonal-like packing
    hex_edges = []
    # 7-7-7 arrangement
    positions = []
    for row in range(3):
        offset = (row % 2) * 0.5
        for col in range(7):
            positions.append((col + offset, row))
    # Connect neighbors
    for i in range(21):
        for j in range(i + 1, 21):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            if abs(dx) < 1.2 and abs(dy) < 1.2:
                hex_edges.append((i, j))
    graphs.append(hex_edges)

    # Random sparse graph
    rng = np.random.default_rng(123)
    rand_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.15:
                rand_edges.append((i, j))
    # Ensure connectivity
    if len(rand_edges) < N - 1:
        for i in range(N - 1):
            rand_edges.append((i, i + 1))
    graphs.append(rand_edges)

    # Complete graph (all pairs)
    complete_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    graphs.append(complete_edges)

    return graphs


def _force_relax(z, max_iter=500):
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > 1e-8:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < 1e-8:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _epigraph_refine(z, time_budget):
    z = z.copy().astype(float)
    n = N

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def obj_epigraph(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs

        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h

        all_viol = np.concatenate([overlap, left_viol, right_viol, bottom_viol, top_viol])

        max_viol = _lse(all_viol / _LSE_K) * _LSE_K
        if max_viol > 0:
            obj += 5000.0 * max_viol * max_viol

        pos_viol = np.maximum(all_viol, 0.0)
        obj += 100.0 * np.sum(pos_viol * pos_viol)

        return obj

    maxiter = int(min(2000, time_budget * 150))
    try:
        res = minimize(obj_epigraph, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 150})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _constraint_refine(z, time_budget):
    z = z.copy().astype(float)
    n = N

    penalty = 1200.0

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = r_sum_pairs - d_pairs

        left_viol = r - x
        right_viol = x + r - w
        bottom_viol = r - y
        top_viol = y + r - h

        all_viol = np.concatenate([overlap, left_viol, right_viol, bottom_viol, top_viol])

        max_viol = _lse(all_viol / _LSE_K) * _LSE_K
        if max_viol > 0:
            obj += penalty * max_viol * max_viol

        pos_viol = np.maximum(all_viol, 0.0)
        obj += 150.0 * np.sum(pos_viol * pos_viol)

        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(2000, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 180})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.9995, 0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _greedy_expansion(z, time_budget):
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def max_radius_at_position(px, py, other_r, skip_i):
        r_wall = min(px, w - px, py, h - py)
        r_max = r_wall
        for j in range(n):
            if j == skip_i:
                continue
            dx = px - x[j]
            dy = py - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - other_r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    for rnd in range(30):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            best_r = r[i]
            best_x = x[i]
            best_y = y[i]

            for dx_p in [-0.02, -0.01, 0.01, 0.02, -0.005, 0.005]:
                for dy_p in [-0.02, -0.01, 0.01, 0.02, -0.005, 0.005]:
                    px = x[i] + dx_p
                    py = y[i] + dy_p
                    if px < 0 or px > w or py < 0 or py > h:
                        continue

                    r_test = max_radius_at_position(px, py, r, i)
                    if r_test > best_r + 1e-9:
                        r_new = r.copy()
                        r_new[i] = r_test
                        x_new = x.copy()
                        y_new = y.copy()
                        x_new[i] = px
                        y_new[i] = py
                        if _is_valid(x_new, y_new, r_new, w, h):
                            best_r = r_test
                            best_x = px
                            best_y = py

            if best_r > r[i] + 1e-9:
                x[i] = best_x
                y[i] = best_y
                r[i] = best_r
                improved = True

        if not improved:
            break

    for _ in range(80):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.03
        for _ in range(18):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    # Generate spectral initializations from different contact graphs
    contact_graphs = _generate_contact_graphs()
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65, 1.4, 0.6]

    initial_guesses = []

    for edges in contact_graphs[:3]:  # Use first 3 graph types
        for w0 in aspect_ratios[:6]:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            try:
                x, y, r = _spectral_initialization(edges, w0, h0, max_iter=1000)
                z = np.zeros(3 * N + 1)
                z[0::3][:N] = x
                z[1::3][:N] = y
                z[2::3][:N] = r
                z[-1] = w0
                initial_guesses.append(z)
            except Exception:
                pass

    # Also add random initializations
    rng = np.random.default_rng(42)
    for _ in range(8):
        w0 = rng.uniform(0.7, 1.3)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.05, 0.1)
            z[3 * i] = rng.uniform(r0, w0 - r0)
            z[3 * i + 1] = rng.uniform(r0, h0 - r0)
            z[3 * i + 2] = r0
        z[-1] = w0
        initial_guesses.append(z)

    # Grid initialization
    side = 5
    for w0 in [1.0, 0.9, 1.1]:
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        count = 0
        for row in range(side):
            for col in range(side):
                if count >= N:
                    break
                z[3 * count] = (col + 0.5) / side * w0
                z[3 * count + 1] = (row + 0.5) / side * h0
                z[3 * count + 2] = min(w0 / side, h0 / side) * 0.4
                count += 1
        z[-1] = w0
        initial_guesses.append(z)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0
        return True, float(np.sum(r))

    for i, z0 in enumerate(initial_guesses):
        if time.time() > deadline - 20:
            break

        z0_relaxed = _force_relax(z0, max_iter=300)
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(800, max(200, remaining * 40)))

        try:
            res = minimize(_vectorized_objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})
            z_opt = res.x
            valid, val = check_validity(z_opt)
            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()
        except Exception:
            continue

    if best_result is not None:
        # Epigraph refine
        if time.time() < deadline - 20:
            ep = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 10)
            if ep is not None:
                valid, val = check_validity(ep)
                if valid and val > best_value:
                    best_value = val
                    best_result = ep

        # Constraint refine
        if time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 5)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # Greedy expansion
        if time.time() < deadline - 3:
            greedy = _greedy_expansion(best_result, deadline - (time.time() - start_time))
            if greedy is not None:
                valid, val = check_validity(greedy)
                if valid and val > best_value:
                    best_value = val
                    best_result = greedy

    if best_result is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
