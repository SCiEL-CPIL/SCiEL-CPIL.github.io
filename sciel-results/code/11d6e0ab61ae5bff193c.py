import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s limit

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
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def objective_and_grad(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        viol_left = np.maximum(0, -d_left)
        viol_right = np.maximum(0, -d_right)
        viol_bottom = np.maximum(0, -d_bottom)
        viol_top = np.maximum(0, -d_top)

        boundary_viol = np.sum(viol_left**2) + np.sum(viol_right**2) + \
                        np.sum(viol_bottom**2) + np.sum(viol_top**2)

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-16)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        penalty_w = 5000.0
        value = -np.sum(r) + penalty_w * (boundary_viol + overlap_viol)

        grad = np.zeros_like(params)

        grad[:2*n].reshape(n, 2)[:, 0] += penalty_w * (-2 * viol_left + 2 * viol_right)
        grad[:2*n].reshape(n, 2)[:, 1] += penalty_w * (-2 * viol_bottom + 2 * viol_top)
        grad[2*n:] += penalty_w * (2 * viol_left - 2 * viol_right + 2 * viol_bottom - 2 * viol_top)

        overlap_grad_xy = np.zeros((n, 2))
        overlap_grad_r = np.zeros(n)

        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            o = overlap[k]
            if o > 0:
                d = dist[i, j]
                if d < 1e-15:
                    continue
                gx = 2 * o * (dx[i, j] / d)
                gy = 2 * o * (dy[i, j] / d)
                overlap_grad_xy[i, 0] += gx
                overlap_grad_xy[i, 1] += gy
                overlap_grad_xy[j, 0] -= gx
                overlap_grad_xy[j, 1] -= gy
                overlap_grad_r[i] += 2 * o
                overlap_grad_r[j] += 2 * o

        grad[:2*n].reshape(n, 2)[:] += penalty_w * overlap_grad_xy
        grad[2*n:] += penalty_w * overlap_grad_r

        grad[2*n:] -= 1.0

        return value, grad

    def push_apart(params, n_iters=200):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-16)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            overlap_pairs = np.maximum(0, min_dist[iu] - dist[iu])

            if np.any(overlap_pairs > 1e-12):
                moved = True
                idx_i, idx_j = iu
                for k in range(len(idx_i)):
                    i, j = idx_i[k], idx_j[k]
                    d = dist[i, j]
                    if d < 1e-12:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.005 * np.cos(angle)
                        xy[i, 1] -= 0.005 * np.sin(angle)
                        xy[j, 0] += 0.005 * np.cos(angle)
                        xy[j, 1] += 0.005 * np.sin(angle)
                        continue
                    ov = overlap_pairs[k]
                    wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                    wj = 1 - wi
                    push_x = (dx[i, j] / d) * ov
                    push_y = (dy[i, j] / d) * ov
                    xy[i, 0] -= push_x * wi
                    xy[i, 1] -= push_y * wi
                    xy[j, 0] += push_x * wj
                    xy[j, 1] += push_y * wj

            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])

            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def hex_init(rng, variant=0):
        n = N
        if variant == 0:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            spacing = 0.165
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5) * spacing + (spacing * 0.5 if row % 2 == 1 else 0)
                y = (row + 0.5) * spacing * np.sqrt(3)/2
                x_offset = (1 - 5*spacing - spacing*0.5) / 2
                y_offset = (1 - 5*spacing*np.sqrt(3)/2) / 2
                xy[i, 0] = x + x_offset
                xy[i, 1] = y + y_offset
                r[i] = spacing * 0.48
        elif variant == 1:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i, 0] = (col + 0.5) / 8
                xy[i, 1] = (row + 0.5) / 4
                r[i] = 0.5/8 - 0.002
        elif variant == 2:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i, 0] = (col + 0.5) / 4
                xy[i, 1] = (row + 0.5) / 8
                r[i] = 0.5/4 - 0.002
        elif variant == 3:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            spacing = 0.165
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5) * spacing + (spacing * 0.5 if row % 2 == 1 else 0)
                y = (row + 0.5) * spacing * np.sqrt(3)/2
                x_offset = (1 - 5*spacing - spacing*0.5) / 2
                y_offset = (1 - 5*spacing*np.sqrt(3)/2) / 2
                xy[i, 0] = x + x_offset + rng.uniform(-0.01, 0.01)
                xy[i, 1] = y + y_offset + rng.uniform(-0.01, 0.01)
                r[i] = spacing * 0.48
        elif variant == 4:
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.06, 0.08, size=n)
        elif variant == 5:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                xy[i, 0] = (col + 0.5) / 6
                xy[i, 1] = (row + 0.5) / 6
                r[i] = 0.5/6 - 0.002
        elif variant == 6:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            spacing = 0.16
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5) * spacing + (spacing * 0.5 if row % 2 == 1 else 0)
                y = (row + 0.5) * spacing * np.sqrt(3)/2
                x_offset = (1 - 5*spacing - spacing*0.5) / 2
                y_offset = (1 - 5*spacing*np.sqrt(3)/2) / 2
                xy[i, 0] = x + x_offset
                xy[i, 1] = y + y_offset
                r[i] = spacing * 0.49
        elif variant == 7:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i, 0] = cx + (sx - 0.5)*0.35
                xy[i, 1] = cy + (sy - 0.5)*0.35
                r[i] = 0.055
        else:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                xy[i, 0] = (col + 0.5) / 6
                xy[i, 1] = (row + 0.5) / 6
                r[i] = 0.5/6 - 0.002

        return np.concatenate([xy.flatten(), r])

    def optimize(params, max_iter=500, time_limit=10):
        t0 = time.time()
        try:
            result = minimize(objective_and_grad, params, method='L-BFGS-B',
                             jac=True,
                             options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13,
                                      'maxfun': max_iter * 5})
            if time.time() - t0 > time_limit:
                return params
            return result.x
        except Exception:
            return params

    def grow_radii(params, n_grow=50, growth_rate=0.003, time_limit=10):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        t0 = time.time()

        for _ in range(n_grow):
            if time.time() - t0 > time_limit:
                break
            r_new = r * (1 + growth_rate)
            params_new = np.concatenate([xy.flatten(), r_new])
            params_opt = optimize(params_new, max_iter=100, time_limit=2)
            if is_valid(params_opt):
                xy = params_opt[:2*n].reshape(n, 2)
                r = params_opt[2*n:]
            else:
                r_new2 = r * (1 + growth_rate * 0.5)
                params_new2 = np.concatenate([xy.flatten(), r_new2])
                params_opt2 = optimize(params_new2, max_iter=80, time_limit=1)
                if is_valid(params_opt2):
                    xy = params_opt2[:2*n].reshape(n, 2)
                    r = params_opt2[2*n:]
                else:
                    break

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(42)
    n_variants = 8

    # Phase 1: Diverse restarts with gradient-based optimization
    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break

        variant = restart % n_variants
        params = hex_init(rng, variant)
        params = push_apart(params, n_iters=150)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        refined = optimize(params, max_iter=300, time_limit=5)
        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            repaired = push_apart(refined, n_iters=150)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(100):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                if is_valid(np.concatenate([xy.flatten(), r])):
                    score = get_score(np.concatenate([xy.flatten(), r]))
                    if score > best_score:
                        best_score = score
                        best_result = np.concatenate([xy.flatten(), r]).copy()

    # Phase 2: Grow radii on best
    if best_result is not None and time.time() < deadline - 15:
        grown = grow_radii(best_result, n_grow=100, growth_rate=0.003, time_limit=12)
        if is_valid(grown):
            score = get_score(grown)
            if score > best_score:
                best_score = score
                best_result = grown.copy()

    # Phase 3: Polish with more iterations
    if best_result is not None and time.time() < deadline - 8:
        polished = optimize(best_result, max_iter=500, time_limit=8)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        for _ in range(15):
            if time.time() > deadline - 3:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.003, 0.003, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=80)
            polished2 = optimize(perturbed, max_iter=200, time_limit=2)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    # Phase 4: Final grow
    if best_result is not None and time.time() < deadline - 2:
        grown2 = grow_radii(best_result, n_grow=40, growth_rate=0.002, time_limit=5)
        if is_valid(grown2):
            score = get_score(grown2)
            if score > best_score:
                best_score = score
                best_result = grown2.copy()

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

    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
