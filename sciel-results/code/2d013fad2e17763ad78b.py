import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def elapsed():
        return time.time() - start_time

    def validate(x, y, r, tol=1e-9):
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return False
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def solve_radii(x, y, max_iter=50000, tol=1e-12):
        """Given fixed positions, find max feasible radii via iteration."""
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        wall_limit = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = wall_limit * 0.5
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            new_r = r.copy()
            for i in range(N):
                cand = wall_limit[i]
                for j in range(N):
                    if i == j:
                        continue
                    dl = dist[i, j]
                    if dl < 1e-9:
                        continue
                    nl = dl - r[j]
                    if nl < cand:
                        cand = nl
                new_r[i] = max(0.0, cand)
            delta = np.max(np.abs(new_r - r))
            r = new_r
            if delta < tol:
                break
        return r

    def objective(params):
        """
        Energy landscape objective: minimize -sum(r) + repulsive potential.
        Vectorized for speed.
        """
        x = params[0:N]
        y = params[N:2 * N]
        r = params[2 * N:3 * N]

        obj = -np.sum(r)

        # Wall penalty: smooth
        v1 = np.maximum(0, r - x)
        v2 = np.maximum(0, r + x - 1)
        v3 = np.maximum(0, r - y)
        v4 = np.maximum(0, r + y - 1)
        penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)

        # Pairwise repulsive potential: vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        # Only count upper triangle to avoid double counting
        triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        penalty += np.sum(overlap[triu_mask]**2)

        return obj + 500.0 * penalty

    def grad_objective(params):
        """Gradient of the energy landscape objective. Vectorized."""
        x = params[0:N]
        y = params[N:2 * N]
        r = params[2 * N:3 * N]
        g = np.zeros(3 * N)
        g[2 * N:3 * N] = -1.0

        # Wall gradients
        v1 = np.maximum(0, r - x)
        m1 = v1 > 0
        g[0:N][m1] += 500.0 * (-2 * v1[m1])
        g[2 * N:3 * N][m1] += 500.0 * (2 * v1[m1])

        v2 = np.maximum(0, r + x - 1)
        m2 = v2 > 0
        g[0:N][m2] += 500.0 * (2 * v2[m2])
        g[2 * N:3 * N][m2] += 500.0 * (2 * v2[m2])

        v3 = np.maximum(0, r - y)
        m3 = v3 > 0
        g[N:2 * N][m3] += 500.0 * (-2 * v3[m3])
        g[2 * N:3 * N][m3] += 500.0 * (2 * v3[m3])

        v4 = np.maximum(0, r + y - 1)
        m4 = v4 > 0
        g[N:2 * N][m4] += 500.0 * (2 * v4[m4])
        g[2 * N:3 * N][m4] += 500.0 * (2 * v4[m4])

        # Pairwise gradients (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        r_sum = r[:, None] + r[None, :]
        overlap = np.maximum(0, r_sum - d)
        
        # Only process upper triangle
        triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_triu = overlap[triu_mask]
        d_triu = d[triu_mask]
        dx_triu = dx[triu_mask]
        dy_triu = dy[triu_mask]
        
        # Get indices
        i_idx, j_idx = np.where(triu_mask)
        
        # Avoid division by zero
        valid = d_triu > 1e-12
        if np.any(valid):
            v = overlap_triu[valid]
            d_v = d_triu[valid]
            dx_v = dx_triu[valid]
            dy_v = dy_triu[valid]
            i_v = i_idx[valid]
            j_v = j_idx[valid]
            
            factor = 500.0 * 2 * v / d_v
            
            # Scatter gradients
            np.add.at(g, i_v, -factor * dx_v)
            np.add.at(g, N + i_v, -factor * dy_v)
            np.add.at(g, 2 * N + i_v, 500.0 * 2 * v)
            
            np.add.at(g, j_v, factor * dx_v)
            np.add.at(g, N + j_v, factor * dy_v)
            np.add.at(g, 2 * N + j_v, 500.0 * 2 * v)
        
        # Handle degenerate case (d < 1e-12)
        degenerate = ~valid
        if np.any(degenerate):
            v_deg = overlap_triu[degenerate]
            i_deg = i_idx[degenerate]
            j_deg = j_idx[degenerate]
            np.add.at(g, 2 * N + i_deg, 500.0 * 2 * v_deg)
            np.add.at(g, 2 * N + j_deg, 500.0 * 2 * v_deg)

        return g

    def make_initial_configs():
        """Generate diverse initial configurations."""
        configs = []

        # 1. Symmetric 5x5 grid + center (structural anchor)
        side = 5
        spacing = 1.0 / side
        grid_pos = []
        for i in range(side):
            for j in range(side):
                grid_pos.append(((i + 0.5) * spacing, (j + 0.5) * spacing))
        grid_pos.append((0.5, 0.5))
        x = np.array([p[0] for p in grid_pos])
        y = np.array([p[1] for p in grid_pos])
        r = solve_radii(x, y)
        configs.append(np.concatenate([x, y, r]))

        # 2. Hexagonal-like packing
        hex_configs = []
        for row in range(5):
            for col in range(5):
                xx = (col + 0.5) * 0.2
                yy = (row + 0.5) * 0.2
                if row % 2 == 1:
                    xx += 0.1
                if 0 <= xx < 1.0 and 0 <= yy < 1.0:
                    hex_configs.append((xx, yy))
        hex_configs = hex_configs[:26]
        if len(hex_configs) == 26:
            x = np.array([p[0] for p in hex_configs])
            y = np.array([p[1] for p in hex_configs])
            r = solve_radii(x, y)
            configs.append(np.concatenate([x, y, r]))

        # 3. Random uniform starts
        for seed in range(8):
            rng = np.random.RandomState(100 + seed)
            x = rng.uniform(0.15, 0.85, N)
            y = rng.uniform(0.15, 0.85, N)
            r = solve_radii(x, y)
            configs.append(np.concatenate([x, y, r]))

        # 4. Clustered: two groups
        for seed in range(4):
            rng = np.random.RandomState(200 + seed)
            x1 = rng.uniform(0.1, 0.45, 13)
            y1 = rng.uniform(0.1, 0.9, 13)
            x2 = rng.uniform(0.55, 0.9, 13)
            y2 = rng.uniform(0.1, 0.9, 13)
            x = np.concatenate([x1, x2])
            y = np.concatenate([y1, y2])
            r = solve_radii(x, y)
            configs.append(np.concatenate([x, y, r]))

        # 5. Ring + center
        for seed in range(3):
            rng = np.random.RandomState(300 + seed)
            angles1 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x1 = 0.5 + 0.35 * np.cos(angles1)
            y1 = 0.5 + 0.35 * np.sin(angles1)
            angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            x2 = 0.5 + 0.18 * np.cos(angles2)
            y2 = 0.5 + 0.18 * np.sin(angles2)
            x = np.concatenate([x1, x2, [0.5, 0.5]])
            y = np.concatenate([y1, y2, [0.5, 0.5]])
            r = solve_radii(x, y)
            configs.append(np.concatenate([x, y, r]))

        # 6. Corner-weighted starts
        for seed in range(3):
            rng = np.random.RandomState(400 + seed)
            x = np.concatenate([
                rng.uniform(0.1, 0.3, 7),
                rng.uniform(0.7, 0.9, 7),
                rng.uniform(0.1, 0.3, 7),
                rng.uniform(0.7, 0.9, 7),
                rng.uniform(0.35, 0.65, 2),
            ])
            y = np.concatenate([
                rng.uniform(0.1, 0.3, 7),
                rng.uniform(0.1, 0.3, 7),
                rng.uniform(0.7, 0.9, 7),
                rng.uniform(0.7, 0.9, 7),
                rng.uniform(0.35, 0.65, 2),
            ])
            r = solve_radii(x, y)
            configs.append(np.concatenate([x, y, r]))

        return configs

    def optimize_from_start(params0):
        """Run L-BFGS-B from initial params."""
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((0.0, 0.5))

        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            jac=grad_objective,
            bounds=bounds,
            options={'maxiter': 3000, 'maxfun': 8000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result

    def local_refine(x, y, r, n_perturb=50):
        """
        Local refinement: perturb positions, re-solve radii, keep improvements.
        """
        score = float(np.sum(r))
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

        for p in range(n_perturb):
            if time.time() > deadline - 5:
                break

            rng = np.random.RandomState(42 + p * 13 + int(score * 1000))
            k = np.random.randint(1, 6)
            idxs = np.random.choice(N, k, replace=False)
            scale = 0.02 + 0.01 * (p % 3)

            x_try = best_x.copy()
            y_try = best_y.copy()
            x_try[idxs] += np.random.normal(0, scale, k)
            y_try[idxs] += np.random.normal(0, scale, k)
            x_try = np.clip(x_try, 0.05, 0.95)
            y_try = np.clip(y_try, 0.05, 0.95)

            r_try = solve_radii(x_try, y_try)
            r_try *= (1 - 1e-10)

            if validate(x_try, y_try, r_try):
                s_try = float(np.sum(r_try))
                if s_try > score + 1e-10:
                    best_x, best_y, best_r = x_try, y_try, r_try
                    score = s_try

        return best_x, best_y, best_r, score

    # Generate initial configs
    configs = make_initial_configs()

    best_score = 0.0
    best_config = None

    # Process each initial config
    for idx, cfg in enumerate(configs):
        if elapsed() > deadline - 10:
            break

        x0 = cfg[0:N]
        y0 = cfg[N:2 * N]
        r0 = cfg[2 * N:3 * N]

        # Check if initial is valid
        if validate(x0, y0, r0):
            s = float(np.sum(r0))
            if s > best_score:
                best_score = s
                best_config = np.column_stack([x0, y0, r0])

        # Run L-BFGS-B energy landscape optimization
        try:
            result = optimize_from_start(cfg)
            params_opt = result.x
            x_opt = params_opt[0:N]
            y_opt = params_opt[N:2 * N]
            r_opt = params_opt[2 * N:3 * N]

            if validate(x_opt, y_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_opt, y_opt, r_opt])
            else:
                # Repair: solve radii for optimized positions
                r_repair = solve_radii(x_opt, y_opt)
                if validate(x_opt, y_opt, r_repair):
                    s = float(np.sum(r_repair))
                    if s > best_score:
                        best_score = s
                        best_config = np.column_stack([x_opt, y_opt, r_repair])
        except Exception:
            pass

    # Local refinement on best config
    if best_config is not None and elapsed() < deadline - 15:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()

        x_ref, y_ref, r_ref, score_ref = local_refine(x_cur, y_cur, r_cur, n_perturb=100)
        if score_ref > best_score:
            best_score = score_ref
            best_config = np.column_stack([x_ref, y_ref, r_ref])

    # Second round of L-BFGS-B from refined config
    if best_config is not None and elapsed() < deadline - 10:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()
        params_cur = np.concatenate([x_cur, y_cur, r_cur])
        try:
            result = optimize_from_start(params_cur)
            params_opt = result.x
            x_opt = params_opt[0:N]
            y_opt = params_opt[N:2 * N]
            r_opt = params_opt[2 * N:3 * N]
            if validate(x_opt, y_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_config = np.column_stack([x_opt, y_opt, r_opt])
            else:
                r_repair = solve_radii(x_opt, y_opt)
                if validate(x_opt, y_opt, r_repair):
                    s = float(np.sum(r_repair))
                    if s > best_score:
                        best_score = s
                        best_config = np.column_stack([x_opt, y_opt, r_repair])
        except Exception:
            pass

    # Final local refinement
    if best_config is not None and elapsed() < deadline - 5:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()
        x_ref, y_ref, r_ref, score_ref = local_refine(x_cur, y_cur, r_cur, n_perturb=50)
        if score_ref > best_score:
            best_score = score_ref
            best_config = np.column_stack([x_ref, y_ref, r_ref])

    # Fallback if no valid config found
    if best_config is None:
        side_f = 6
        s = 1.0 / side_f
        cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
        best_config = np.array(cells[:N])
        best_score = float(np.sum(best_config[:, 2]))

    # Final validation
    x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
    if not validate(x_f, y_f, r_f):
        for factor in [0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7]:
            r_f2 = r_f * factor
            if validate(x_f, y_f, r_f2):
                best_config = np.column_stack([x_f, y_f, r_f2])
                break
        else:
            side_f = 6
            s = 1.0 / side_f
            cells = [((i + 0.5) * s, (j + 0.5) * s, 0.4 * s) for i in range(side_f) for j in range(side_f)]
            best_config = np.array(cells[:N])

    return best_config
