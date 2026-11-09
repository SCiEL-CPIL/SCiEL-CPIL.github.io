import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 117.0

    def pack_loss(params):
        """Loss = -sum(r) + penalty for violations. Minimize this."""
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]

        # Wall violations (circle must be inside [0,1]^2)
        wall_viol = np.maximum(0, x - r) + np.maximum(0, y - r) + \
                    np.maximum(0, -(x - r)) + np.maximum(0, -(y - r)) + \
                    np.maximum(0, (x + r) - 1) + np.maximum(0, (y + r) - 1)

        # Pairwise overlap violations
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        overlap = np.maximum(0, r[:, None] + r[None, :] - dist)
        overlap_viol = np.sum(overlap)

        total_viol = wall_viol + overlap_viol
        penalty = 100.0 * total_viol

        return -np.sum(r) + penalty

    def pack_loss_grad(params):
        """Gradient of pack_loss."""
        x = params[:N].copy()
        y = params[N:2*N].copy()
        r = params[2*N:3*N].copy()

        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)

        # Objective: -sum(r)
        gr -= 1.0

        # Wall penalties
        # max(0, x - r): if x > r, gx += 1, gr -= 1
        mask = x > r
        gx[mask] += 1.0
        gr[mask] -= 1.0
        # max(0, y - r): if y > r, gy += 1, gr -= 1
        mask = y > r
        gy[mask] += 1.0
        gr[mask] -= 1.0
        # max(0, -(x-r)) = max(0, r-x): if r > x, gx -= 1, gr += 1
        mask = r > x
        gx[mask] -= 1.0
        gr[mask] += 1.0
        # max(0, -(y-r)) = max(0, r-y): if r > y, gy -= 1, gr += 1
        mask = r > y
        gy[mask] -= 1.0
        gr[mask] += 1.0
        # max(0, (x+r)-1): if x+r > 1, gx += 1, gr += 1
        mask = x + r > 1
        gx[mask] += 1.0
        gr[mask] += 1.0
        # max(0, (y+r)-1): if y+r > 1, gy += 1, gr += 1
        mask = y + r > 1
        gy[mask] += 1.0
        gr[mask] += 1.0

        # Pairwise overlap: max(0, r_i + r_j - d_ij)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        diff = r[:, None] + r[None, :] - dist
        mask = diff > 0

        # For each violating pair (i,j): d(loss)/dx_i += (x_i - x_j)/d_ij, etc.
        for i in range(N):
            for j in range(i + 1, N):
                if mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-12:
                        continue
                    ux = (x[i] - x[j]) / d
                    uy = (y[i] - y[j]) / d
                    gx[i] += ux
                    gy[i] += uy
                    gx[j] -= ux
                    gy[j] -= uy
                    gr[i] += 1.0
                    gr[j] += 1.0

        # Scale by penalty coefficient
        gx *= 100.0
        gy *= 100.0
        gr *= 100.0

        return np.concatenate([gx, gy, gr])

    def validate(x, y, r, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        if np.any(r[:, None] + r[None, :] - dist > tol):
            return False
        return True

    def repair_and_shrink(x, y, r):
        """Shrink radii slightly and adjust positions to ensure validity."""
        # Clip to valid range
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.maximum(r, 0)

        # Iteratively shrink to fix violations
        for _ in range(50):
            if validate(x, y, r):
                break
            # Find worst violation and shrink
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            overlap = r[:, None] + r[None, :] - dist
            max_overlap = np.max(overlap)

            wall_viol = np.maximum(0, x + r - 1) + np.maximum(0, y + r - 1) + \
                        np.maximum(0, -(x - r)) + np.maximum(0, -(y - r))
            max_wall = np.max(wall_viol)

            if max_wall > max_overlap:
                # Shrink wall-violating circles
                for i in range(N):
                    wv = np.maximum(0, x[i] + r[i] - 1) + np.maximum(0, y[i] + r[i] - 1) + \
                         np.maximum(0, -(x[i] - r[i])) + np.maximum(0, -(y[i] - r[i]))
                    if wv > 0:
                        r[i] = max(0, r[i] - wv * 0.5)
            else:
                # Shrink overlapping pairs
                r *= 0.99

        # Final safety shrink
        r *= (1 - 1e-10)
        return x, y, r

    # Generate diverse initial configurations
    def make_initial_config(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, 0.03)
        return x, y, r

    def make_grid_config():
        # 5x5 grid + center
        side = 5
        spacing = 1.0 / side
        x = np.array([(i + 0.5) * spacing for i in range(side) for j in range(side)])
        y = np.array([(j + 0.5) * spacing for i in range(side) for j in range(side)])
        x = np.append(x, 0.5)
        y = np.append(y, 0.5)
        r = np.full(N, 0.08)
        return x, y, r

    def make_hex_config():
        # Hexagonal-like packing
        x = []
        y = []
        rows = 5
        for row in range(rows):
            n_cols = 6 if row % 2 == 0 else 5
            offset = 0.0 if row % 2 == 0 else 1.0 / (2 * n_cols)
            for col in range(n_cols):
                if len(x) >= N:
                    break
                x.append(offset + col / n_cols)
                y.append(0.1 + row * 0.8 / (rows - 1))
        # Pad if needed
        while len(x) < N:
            x.append(0.5)
            y.append(0.5)
        x = np.array(x[:N])
        y = np.array(y[:N])
        r = np.full(N, 0.06)
        return x, y, r

    best_score = 0.0
    best_config = None

    # Initial configs to try
    configs = [
        make_grid_config(),
        make_hex_config(),
    ]
    for seed in [42, 123, 777, 2024, 999, 31415, 27182, 16180]:
        configs.append(make_initial_config(seed))

    for cfg_idx, (x0, y0, r0) in enumerate(configs):
        if time.time() > deadline - 10:
            break

        params0 = np.concatenate([x0, y0, r0])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        try:
            result = minimize(
                pack_loss,
                params0,
                jac=pack_loss_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue

        if time.time() > deadline - 5:
            break

        x_opt = result.x[:N]
        y_opt = result.x[N:2*N]
        r_opt = result.x[2*N:3*N]

        # Repair
        x_rep, y_rep, r_rep = repair_and_shrink(x_opt, y_opt, r_opt)

        if validate(x_rep, y_rep, r_rep):
            score = float(np.sum(r_rep))
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x_rep, y_rep, r_rep))

    # If we have a good config, try further local refinement with more iterations
    if best_config is not None and time.time() < deadline - 15:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()

        params_cur = np.concatenate([x_cur, y_cur, r_cur])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        try:
            result = minimize(
                pack_loss,
                params_cur,
                jac=pack_loss_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
            )
            x_ref = result.x[:N]
            y_ref = result.x[N:2*N]
            r_ref = result.x[2*N:3*N]
            x_ref, y_ref, r_ref = repair_and_shrink(x_ref, y_ref, r_ref)
            if validate(x_ref, y_ref, r_ref):
                score_ref = float(np.sum(r_ref))
                if score_ref > best_score:
                    best_score = score_ref
                    best_config = np.column_stack((x_ref, y_ref, r_ref))
        except Exception:
            pass

    # Fallback if no valid config found
    if best_config is None:
        side_f = 6
        s = 1.0 / side_f
        cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
        best_config = np.array(cells[:N])

    # Final validation
    x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
    if not validate(x_f, y_f, r_f):
        for factor in [0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5]:
            r_f2 = r_f * factor
            if validate(x_f, y_f, r_f2):
                best_config = np.column_stack((x_f, y_f, r_f2))
                break
        else:
            side_f = 6
            s = 1.0 / side_f
            cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
            best_config = np.array(cells[:N])

    return best_config
