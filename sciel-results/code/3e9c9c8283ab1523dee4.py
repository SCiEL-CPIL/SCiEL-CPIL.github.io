import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    best_result = None
    best_score = -np.inf

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx**2 + dy**2
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist_sq[iu] < (min_dist[iu] - tol)**2):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=100, rng=None):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        if rng is None:
            rng = np.random

        for _ in range(n_iters):
            moved = False

            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist_sq = dx**2 + dy**2
                    min_dist = r[i] + r[j]

                    if dist_sq < min_dist**2 and dist_sq > 1e-24:
                        dist = np.sqrt(dist_sq)
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1 - wi

                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap

                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist_sq <= 1e-24:
                        angle = rng.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True

            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True

            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def objective(z):
        # z is [x1, y1, r1, x2, y2, r2, ...]
        r = z[2::3]
        return -np.sum(r)

    def constraints_func(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        cons = []
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        cons.append(x - r)
        cons.append(1.0 - x - r)
        cons.append(y - r)
        cons.append(1.0 - y - r)
        
        # Overlap constraints: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
        # Equivalent to dist_sq - (ri+rj)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                rad_sum_sq = (r[i] + r[j])**2
                cons.append(dist_sq - rad_sum_sq)
        
        return np.concatenate(cons)

    def bounds():
        bounds = []
        for _ in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
        return bounds

    def solve_slsqp(z0, max_iter=500):
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds(),
                           constraints=[{'type': 'ineq', 'fun': constraints_func}],
                           options={'maxiter': max_iter, 'ftol': 1e-12})
            return res.x
        except Exception:
            return z0

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart % 10 == 0:
            # Grid packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            # Perturbed grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            # Hexagonal packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 10 == 3:
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            # Hexagonal variant
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 10 == 7:
            # Clustered: 4 groups of 8
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        elif restart % 10 == 8:
            # Random medium
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        # Convert to interleaved format [x1, y1, r1, ...]
        z = np.zeros(3*n)
        z[0::3] = xy[:, 0]
        z[1::3] = xy[:, 1]
        z[2::3] = r
        return z

    def convert_to_interleaved(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        z = np.zeros(3*n)
        z[0::3] = xy[:, 0]
        z[1::3] = xy[:, 1]
        z[2::3] = r
        return z

    def convert_from_interleaved(z):
        n = N
        xy = np.zeros((n, 2))
        xy[:, 0] = z[0::3]
        xy[:, 1] = z[1::3]
        r = z[2::3]
        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(12345)

    # Phase 1: Restart with SLSQP
    n_restarts = 20  # SLSQP is slower, so fewer restarts
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        z0 = get_initial_config(restart, rng)
        
        # Small perturbation to break symmetry if needed
        if restart > 0:
            z0 += rng.uniform(-0.01, 0.01, 3*N)
            z0[2::3] = np.clip(z0[2::3], 0.01, 0.4)
            z0[0::3] = np.clip(z0[0::3], 0.01, 0.99)
            z0[1::3] = np.clip(z0[1::3], 0.01, 0.99)

        # Solve with SLSQP
        z_res = solve_slsqp(z0, max_iter=300)
        
        # Convert back to (x, y, r) format for validation
        params_res = convert_from_interleaved(z_res)
        
        # Repair if necessary
        if not is_valid(params_res):
            params_res = push_apart(params_res, n_iters=50, rng=rng)
            if not is_valid(params_res):
                # Shrink radii
                r = params_res[2*N:]
                r *= 0.9
                params_res[2*N:] = r
                if not is_valid(params_res):
                    continue

        score = get_score(params_res)
        if score > best_score:
            best_score = score
            best_result = params_res.copy()

    # Phase 2: Polish the best result
    if best_result is not None and time.time() < deadline - 5:
        z_best = convert_to_interleaved(best_result)
        
        # High precision polish with SLSQP
        z_polished = solve_slsqp(z_best, max_iter=1000)
        params_polished = convert_from_interleaved(z_polished)
        
        if not is_valid(params_polished):
            params_polished = push_apart(params_polished, n_iters=50, rng=rng)
        
        if is_valid(params_polished):
            score = get_score(params_polished)
            if score > best_score:
                best_score = score
                best_result = params_polished.copy()

        # Try a few local perturbations
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            z_pert = best_result.copy()
            z_pert = convert_to_interleaved(z_pert)
            z_pert += rng.uniform(-0.005, 0.005, 3*N)
            z_pert[2::3] = np.clip(z_pert[2::3], 0.01, 0.4)
            z_pert[0::3] = np.clip(z_pert[0::3], 0.01, 0.99)
            z_pert[1::3] = np.clip(z_pert[1::3], 0.01, 0.99)
            
            z_polished = solve_slsqp(z_pert, max_iter=500)
            params_polished = convert_from_interleaved(z_polished)
            
            if not is_valid(params_polished):
                params_polished = push_apart(params_polished, n_iters=50, rng=rng)
            
            if is_valid(params_polished):
                score = get_score(params_polished)
                if score > best_score:
                    best_score = score
                    best_result = params_polished.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
