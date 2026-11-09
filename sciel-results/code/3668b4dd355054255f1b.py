import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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
        Objective: minimize -sum(r) + penalty for constraint violations.
        params: flat array of length 3n: [x0,y0,r0, x1,y1,r1, ...]
        """
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]

        penalty = 0.0
        # Boundary penalties
        for i in range(N):
            if x[i] < r[i]:
                penalty += (r[i] - x[i]) ** 2
            if x[i] > 1.0 - r[i]:
                penalty += (x[i] - (1.0 - r[i])) ** 2
            if y[i] < r[i]:
                penalty += (r[i] - y[i]) ** 2
            if y[i] > 1.0 - r[i]:
                penalty += (y[i] - (1.0 - r[i])) ** 2

        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        return -np.sum(r) + 1000.0 * penalty

    def grad_objective(params):
        """Gradient of the objective."""
        x = params[0::3].copy()
        y = params[1::3].copy()
        r = params[2::3].copy()
        grad = np.zeros_like(params)

        # Boundary gradients
        for i in range(N):
            if x[i] < r[i]:
                grad[3*i] += 2000.0 * (r[i] - x[i])
                grad[3*i+2] -= 2000.0 * (r[i] - x[i])
            if x[i] > 1.0 - r[i]:
                grad[3*i] -= 2000.0 * (x[i] - (1.0 - r[i]))
                grad[3*i+2] += 2000.0 * (x[i] - (1.0 - r[i]))
            if y[i] < r[i]:
                grad[3*i+1] += 2000.0 * (r[i] - y[i])
                grad[3*i+2] -= 2000.0 * (r[i] - y[i])
            if y[i] > 1.0 - r[i]:
                grad[3*i+1] -= 2000.0 * (y[i] - (1.0 - r[i]))
                grad[3*i+2] += 2000.0 * (y[i] - (1.0 - r[i]))

        # Overlap gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    diff = min_dist - dist
                    factor = 2000.0 * diff / dist
                    grad[3*i] += factor * dx
                    grad[3*i+1] += factor * dy
                    grad[3*i+2] += 2000.0 * diff
                    grad[3*j] -= factor * dx
                    grad[3*j+1] -= factor * dy
                    grad[3*j+2] += 2000.0 * diff
                elif dist <= 1e-10:
                    grad[3*i+2] += 2000.0 * min_dist
                    grad[3*j+2] += 2000.0 * min_dist

        # Gradient of -sum(r)
        grad[2::3] -= 1.0

        return grad

    def project_to_valid(circles):
        """Project circles to be inside unit square with positive radii."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            r = max(r, 1e-6)
            x = np.clip(x, r, 1.0 - r)
            y = np.clip(y, r, 1.0 - r)
            circles[i] = [x, y, r]
        return circles

    def make_initial_configs():
        """Generate diverse initial configurations in interleaved format."""
        configs = []

        # 1. Symmetric 5x5 grid + center
        side = 5
        spacing = 1.0 / side
        grid_pos = []
        for i in range(side):
            for j in range(side):
                grid_pos.append(((i + 0.5) * spacing, (j + 0.5) * spacing))
        grid_pos.append((0.5, 0.5))
        params = np.zeros(3 * N)
        for idx, (px, py) in enumerate(grid_pos):
            params[3*idx] = px
            params[3*idx+1] = py
            params[3*idx+2] = 0.5 * spacing
        configs.append(params)

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
            params = np.zeros(3 * N)
            for idx, (px, py) in enumerate(hex_configs):
                params[3*idx] = px
                params[3*idx+1] = py
                params[3*idx+2] = 0.08
            configs.append(params)

        # 3. Random uniform starts
        for seed in range(10):
            rng = np.random.RandomState(100 + seed)
            params = np.zeros(3 * N)
            for idx in range(N):
                params[3*idx] = rng.uniform(0.15, 0.85)
                params[3*idx+1] = rng.uniform(0.15, 0.85)
                params[3*idx+2] = 0.05
            configs.append(params)

        # 4. Clustered: two groups
        for seed in range(5):
            rng = np.random.RandomState(200 + seed)
            params = np.zeros(3 * N)
            for idx in range(N):
                if idx < 13:
                    params[3*idx] = rng.uniform(0.1, 0.45)
                    params[3*idx+1] = rng.uniform(0.1, 0.9)
                else:
                    params[3*idx] = rng.uniform(0.55, 0.9)
                    params[3*idx+1] = rng.uniform(0.1, 0.9)
                params[3*idx+2] = 0.05
            configs.append(params)

        # 5. Ring + center
        for seed in range(4):
            rng = np.random.RandomState(300 + seed)
            params = np.zeros(3 * N)
            angles1 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            for idx in range(12):
                params[3*idx] = 0.5 + 0.35 * np.cos(angles1[idx])
                params[3*idx+1] = 0.5 + 0.35 * np.sin(angles1[idx])
                params[3*idx+2] = 0.05
            angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 0.1, 1)
            for idx in range(12):
                params[3*(12+idx)] = 0.5 + 0.18 * np.cos(angles2[idx])
                params[3*(12+idx)+1] = 0.5 + 0.18 * np.sin(angles2[idx])
                params[3*(12+idx)+2] = 0.05
            params[3*24] = 0.5
            params[3*24+1] = 0.5
            params[3*24+2] = 0.05
            params[3*25] = 0.5
            params[3*25+1] = 0.5
            params[3*25+2] = 0.05
            configs.append(params)

        # 6. Corner-weighted starts
        for seed in range(4):
            rng = np.random.RandomState(400 + seed)
            params = np.zeros(3 * N)
            corner_regions = [
                (0.1, 0.3, 0.1, 0.3),
                (0.7, 0.9, 0.1, 0.3),
                (0.1, 0.3, 0.7, 0.9),
                (0.7, 0.9, 0.7, 0.9),
            ]
            for idx in range(N):
                region = corner_regions[idx % 4]
                params[3*idx] = rng.uniform(region[0], region[1])
                params[3*idx+1] = rng.uniform(region[2], region[3])
                params[3*idx+2] = 0.05
            configs.append(params)

        return configs

    def optimize_from_start(params0):
        """Run L-BFGS-B from initial params."""
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 0.5))

        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                jac=grad_objective,
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            return result
        except Exception:
            return None

    def local_refine(x, y, r, n_perturb=80):
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
            scale = 0.015 + 0.01 * (p % 3)

            x_try = best_x.copy()
            y_try = best_y.copy()
            x_try[idxs] += rng.normal(0, scale, k)
            y_try[idxs] += rng.normal(0, scale, k)
            x_try = np.clip(x_try, 0.02, 0.98)
            y_try = np.clip(y_try, 0.02, 0.98)

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
        if elapsed() > deadline - 15:
            break

        x0 = cfg[0::3]
        y0 = cfg[1::3]
        r0 = cfg[2::3]

        # Run L-BFGS-B optimization
        result = optimize_from_start(cfg)
        if result is not None:
            params_opt = result.x
            x_opt = params_opt[0::3]
            y_opt = params_opt[1::3]
            r_opt = params_opt[2::3]

            # Project to valid
            circles = np.column_stack([x_opt, y_opt, r_opt])
            circles = project_to_valid(circles)
            x_opt = circles[:, 0]
            y_opt = circles[:, 1]
            r_opt = circles[:, 2]

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

        # Also try from the initial config directly (in case optimization diverged)
        r_init = solve_radii(x0, y0)
        if validate(x0, y0, r_init):
            s = float(np.sum(r_init))
            if s > best_score:
                best_score = s
                best_config = np.column_stack([x0, y0, r_init])

    # Local refinement on best config
    if best_config is not None and elapsed() < deadline - 20:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()

        x_ref, y_ref, r_ref, score_ref = local_refine(x_cur, y_cur, r_cur, n_perturb=100)
        if score_ref > best_score:
            best_score = score_ref
            best_config = np.column_stack([x_ref, y_ref, r_ref])

    # Second round of L-BFGS-B from refined config
    if best_config is not None and elapsed() < deadline - 15:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()
        params_cur = np.zeros(3 * N)
        params_cur[0::3] = x_cur
        params_cur[1::3] = y_cur
        params_cur[2::3] = r_cur

        result = optimize_from_start(params_cur)
        if result is not None:
            params_opt = result.x
            x_opt = params_opt[0::3]
            y_opt = params_opt[1::3]
            r_opt = params_opt[2::3]
            circles = np.column_stack([x_opt, y_opt, r_opt])
            circles = project_to_valid(circles)
            x_opt = circles[:, 0]
            y_opt = circles[:, 1]
            r_opt = circles[:, 2]
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

    # Final local refinement
    if best_config is not None and elapsed() < deadline - 10:
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
        cells = []
        for i in range(side_f):
            for j in range(side_f):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) * s, (j + 0.5) * s, 0.4 * s))
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
            cells = []
            for i in range(side_f):
                for j in range(side_f):
                    if len(cells) >= N:
                        break
                    cells.append(((i + 0.5) * s, (j + 0.5) * s, 0.4 * s))
            best_config = np.array(cells[:N])

    return best_config
