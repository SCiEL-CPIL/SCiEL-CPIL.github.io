import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 117.0

    def _hex_lattice(n):
        """Generate a hexagonal lattice configuration for n circles in unit square."""
        # Try to fit a hexagonal pattern
        # For 26 circles, try 5 rows: 5,6,5,6,4 or similar
        configs = []
        # Pattern: rows with alternating counts
        row_patterns = [
            [5, 6, 5, 6, 4],
            [6, 5, 6, 5, 4],
            [5, 5, 6, 5, 5],
            [6, 6, 5, 5, 4],
            [5, 6, 6, 5, 4],
        ]
        
        for pattern in row_patterns:
            if sum(pattern) < n:
                continue
            x_coords = []
            y_coords = []
            num_rows = len(pattern)
            row_height = 1.0 / (num_rows + 0.5)
            
            for row_idx, count in enumerate(pattern):
                y_pos = (row_idx + 0.75) * row_height
                for col in range(count):
                    if len(x_coords) >= n:
                        break
                    # Offset every other row
                    offset = 0.0 if row_idx % 2 == 0 else 0.5 / count
                    x_pos = (col + 0.5) / count + offset * (1.0 / count)
                    # Keep within bounds
                    x_pos = np.clip(x_pos, 0.1, 0.9)
                    x_coords.append(x_pos)
                    y_coords.append(y_pos)
            
            if len(x_coords) == n:
                x = np.array(x_coords)
                y = np.array(y_coords)
                # Estimate radius based on spacing
                # Find minimum distance between any two points
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
                r = np.full(n, 0.45 * min_dist)
                configs.append((x, y, r))
        
        if configs:
            return configs[0]
        
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n)))
        x = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:n])
        y = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:n])
        r = np.full(n, 0.5 / side - 0.01)
        return x, y, r

    def _random_config(seed, scale=0.03):
        """Generate a random initial configuration."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = np.full(N, scale)
        return x, y, r

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
        mask = x > r
        gx[mask] += 1.0
        gr[mask] -= 1.0
        mask = y > r
        gy[mask] += 1.0
        gr[mask] -= 1.0
        mask = r > x
        gx[mask] -= 1.0
        gr[mask] += 1.0
        mask = r > y
        gy[mask] -= 1.0
        gr[mask] += 1.0
        mask = x + r > 1
        gx[mask] += 1.0
        gr[mask] += 1.0
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
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.maximum(r, 0)

        for _ in range(50):
            if validate(x, y, r):
                break
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
                for i in range(N):
                    wv = np.maximum(0, x[i] + r[i] - 1) + np.maximum(0, y[i] + r[i] - 1) + \
                         np.maximum(0, -(x[i] - r[i])) + np.maximum(0, -(y[i] - r[i]))
                    if wv > 0:
                        r[i] = max(0, r[i] - wv * 0.5)
            else:
                r *= 0.99

        r *= (1 - 1e-10)
        return x, y, r

    def force_relax(x, y, r, iters=50, dt=0.01):
        """Simple force-based relaxation to push circles apart and toward walls."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for _ in range(iters):
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-10:
                        force = (min_dist - dist) / dist
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
                    elif dist <= 1e-10:
                        # Random push
                        angle = np.random.uniform(0, 2*np.pi)
                        fx[i] += np.cos(angle)
                        fy[i] += np.sin(angle)
                        fx[j] -= np.cos(angle)
                        fy[j] -= np.sin(angle)
            
            # Wall attraction/repulsion
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i])
                if x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i]))
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i])
                if y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i]))
            
            x += dt * fx
            y += dt * fy
            
            # Clip to valid range
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
        
        return x, y, r

    best_score = 0.0
    best_config = None

    # Generate diverse initial configurations
    configs = []
    
    # Hexagonal lattice start
    hex_cfg = _hex_lattice(N)
    configs.append(hex_cfg)
    
    # Several random starts with different scales
    for seed in [42, 123, 777, 2024, 999, 31415, 27182, 16180]:
        for scale in [0.02, 0.04, 0.06]:
            configs.append(_random_config(seed + scale * 1000, scale))
    
    # Grid start
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
    y_grid = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
    r_grid = np.full(N, 0.5 / side - 0.01)
    configs.append((x_grid, y_grid, r_grid))

    for cfg_idx, (x0, y0, r0) in enumerate(configs):
        if time.time() > deadline - 10:
            break

        # First do force relaxation to get a near-valid starting point
        x_rel, y_rel, r_rel = force_relax(x0, y0, r0, iters=30, dt=0.02)
        
        params0 = np.concatenate([x_rel, y_rel, r_rel])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        try:
            result = minimize(
                pack_loss,
                params0,
                jac=pack_loss_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue

        if time.time() > deadline - 5:
            break

        x_opt = result.x[:N]
        y_opt = result.x[N:2*N]
        r_opt = result.x[2*N:3*N]

        x_rep, y_rep, r_rep = repair_and_shrink(x_opt, y_opt, r_opt)

        if validate(x_rep, y_rep, r_rep):
            score = float(np.sum(r_rep))
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x_rep, y_rep, r_rep))

    # If we have a good config, try further local refinement
    if best_config is not None and time.time() < deadline - 15:
        x_cur = best_config[:, 0].copy()
        y_cur = best_config[:, 1].copy()
        r_cur = best_config[:, 2].copy()

        # Do a force relaxation step to escape shallow local optima
        x_rel, y_rel, r_rel = force_relax(x_cur, y_cur, r_cur, iters=20, dt=0.01)
        params_cur = np.concatenate([x_rel, y_rel, r_rel])
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

        try:
            result = minimize(
                pack_loss,
                params_cur,
                jac=pack_loss_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100}
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

        # Try perturbation-based restarts from best
        if time.time() < deadline - 10:
            rng = np.random.RandomState(42)
            for restart in range(3):
                if time.time() > deadline - 8:
                    break
                
                x_pert = x_cur + rng.randn(N) * 0.05
                y_pert = y_cur + rng.randn(N) * 0.05
                r_pert = r_cur.copy()
                
                x_pert, y_pert, r_pert = force_relax(x_pert, y_pert, r_pert, iters=15, dt=0.01)
                params_pert = np.concatenate([x_pert, y_pert, r_pert])
                
                try:
                    result = minimize(
                        pack_loss,
                        params_pert,
                        jac=pack_loss_grad,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-8}
                    )
                    x_r = result.x[:N]
                    y_r = result.x[N:2*N]
                    r_r = result.x[2*N:3*N]
                    x_r, y_r, r_r = repair_and_shrink(x_r, y_r, r_r)
                    if validate(x_r, y_r, r_r):
                        score_r = float(np.sum(r_r))
                        if score_r > best_score:
                            best_score = score_r
                            best_config = np.column_stack((x_r, y_r, r_r))
                except Exception:
                    continue

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
