"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 21


def _check_validity(circles, W, H, tol=1e-9):
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(N):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True


def _separate_circles(circles, W, H, max_iter=2000):
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - 1e-12) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        for i in range(n):
            max_r = max(1e-4, min(min(x[i], W - x[i]), min(y[i], H - y[i])))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    return result, _check_validity(result, W, H)


def _soft_max(v, scale):
    v_safe = np.clip(v / scale, -50, 50)
    smax = scale * np.log1p(np.exp(v_safe))
    sig = 1.0 / (1.0 + np.exp(-v_safe))
    return smax, sig


def _objective_and_grad(params, W, H, eps, bwt, owt):
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Boundary penalties via soft-max
    for v, gi, gr in [(r - x, 0, 2), (r - y, 1, 2), (x + r - W, 0, 2), (y + r - H, 1, 2)]:
        smax, gsmax = _soft_max(v, eps)
        obj += bwt * np.sum(smax ** 2)
        grad[gi::3] += 2.0 * bwt * smax * gsmax * np.sign(v)
        grad[gr::3] += 2.0 * bwt * smax * gsmax * np.sign(v)

    # r >= 0
    smax, gsmax = _soft_max(-r, eps)
    obj += bwt * np.sum(smax ** 2)
    grad[2::3] -= 2.0 * bwt * smax * gsmax

    # Overlap penalties: vectorized
    iu, ju = np.triu_indices(n, k=1)
    dx_ij = x[iu] - x[ju]
    dy_ij = y[iu] - y[ju]
    dist_ij = np.sqrt(dx_ij**2 + dy_ij**2)
    dist_safe = np.where(dist_ij < 1e-12, 1e-12, dist_ij)
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _soft_max(v_vec, eps)
    obj += owt * np.sum(smax_vec ** 2)

    factor = 2.0 * owt * smax_vec * gsmax_vec
    nx = -dx_ij / dist_safe
    ny = -dy_ij / dist_safe

    np.add.at(grad, iu, factor * nx)
    np.add.at(grad, n + iu, factor * ny)
    np.add.at(grad, 2 * n + iu, factor)
    np.add.at(grad, ju, -factor * nx)
    np.add.at(grad, n + ju, -factor * ny)
    np.add.at(grad, 2 * n + ju, factor)

    return obj, grad


def _local_optimize(circles, W, H, deadline):
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Progressive multi-pass: coarse to fine
    schedule = [
        (5e-4, 150.0, 150.0, 800),
        (2e-4, 250.0, 250.0, 800),
        (1e-4, 400.0, 400.0, 800),
        (5e-5, 600.0, 600.0, 800),
        (2e-5, 800.0, 800.0, 600),
        (1e-5, 1000.0, 1000.0, 400),
    ]

    for eps, bwt, owt, maxit in schedule:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(
                _objective_and_grad, params, args=(W, H, eps, bwt, owt),
                method='L-BFGS-B', jac=True, bounds=bounds,
                options={'maxiter': maxit, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
            )
            params = res.x
        except Exception:
            pass

    return params.reshape(N, 3)


def _linearized_refine(circles, W, H, deadline, max_iter=8):
    if time.time() > deadline - 10.0:
        return None

    n = len(circles)
    cx = circles[:, 0].copy()
    cy = circles[:, 1].copy()
    r = circles[:, 2].copy()

    best_value = float(np.sum(r))
    best_circles = circles.copy()

    for it in range(max_iter):
        if time.time() > deadline - 12.0:
            break

        c_obj = np.zeros(3 * n)
        c_obj[2*n:] = -1.0

        A_rows = []
        b_rows = []

        for i in range(n):
            row = np.zeros(3 * n)
            row[i] = -1.0; row[2*n+i] = 1.0
            A_rows.append(row); b_rows.append(0.0)

            row = np.zeros(3 * n)
            row[i] = 1.0; row[2*n+i] = 1.0
            A_rows.append(row); b_rows.append(W)

            row = np.zeros(3 * n)
            row[n+i] = -1.0; row[2*n+i] = 1.0
            A_rows.append(row); b_rows.append(0.0)

            row = np.zeros(3 * n)
            row[n+i] = 1.0; row[2*n+i] = 1.0
            A_rows.append(row); b_rows.append(H)

        for i in range(n):
            for j in range(i + 1, n):
                dx0 = cx[i] - cx[j]
                dy0 = cy[i] - cy[j]
                d0 = np.sqrt(dx0*dx0 + dy0*dy0)
                if d0 < 1e-12:
                    dx0, dy0 = 1.0, 0.0
                    d0 = 1.0

                row = np.zeros(3 * n)
                row[i] = -dx0; row[j] = dx0
                row[n+i] = -dy0; row[n+j] = dy0
                row[2*n+i] = 1.0; row[2*n+j] = 1.0
                A_rows.append(row); b_rows.append(0.0)

        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)

        bounds_lp = []
        for i in range(n):
            bounds_lp.append((1e-8, W - 1e-8))
            bounds_lp.append((1e-8, H - 1e-8))
        for i in range(n):
            bounds_lp.append((1e-8, min(W, H) / 2))

        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
            if res.status == 0:
                new_circles = np.column_stack((res.x[:n], res.x[n:2*n], res.x[2*n:]))
                if _check_validity(new_circles, W, H, tol=1e-6):
                    val = float(np.sum(new_circles[:, 2]))
                    if val > best_value + 1e-10:
                        best_value = val
                        best_circles = new_circles.copy()
                        cx, cy, r = new_circles[:, 0].copy(), new_circles[:, 1].copy(), new_circles[:, 2].copy()
                    else:
                        break
                else:
                    relaxed, valid = _separate_circles(new_circles, W, H, max_iter=200)
                    if valid:
                        val = float(np.sum(relaxed[:, 2]))
                        if val > best_value + 1e-10:
                            best_value = val
                            best_circles = relaxed.copy()
                            cx, cy, r = relaxed[:, 0].copy(), relaxed[:, 1].copy(), relaxed[:, 2].copy()
                        else:
                            break
                    else:
                        break
            else:
                break
        except Exception:
            break

    return best_circles


def _make_hexagonal_grid(n, w, h):
    rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            offset = 0.5 * (w / 5) if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5 + offset
            r = 0.35 * min(w / 5, h / total_rows)
            circles.append([x, y, r])
    return np.array(circles[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    best_value = 0.0
    best_result = None

    ratios = [1.0, 0.85, 1.18, 0.75, 1.35, 0.65, 1.55, 0.55, 1.8, 0.48, 2.0, 0.42, 2.2]

    for ar in ratios:
        if time.time() > deadline - 18.0:
            break

        w = ar / (1 + ar) * 2
        h = 1 / (1 + ar) * 2

        for (ww, hh) in [(w, h), (h, w)]:
            if time.time() > deadline - 18.0:
                break

            try:
                circles = _make_hexagonal_grid(N, ww, hh)
                circles, valid = _separate_circles(circles, ww, hh, max_iter=1500)
                if not valid:
                    continue

                current_value = float(np.sum(circles[:, 2]))

                # Progressive smooth-penalty optimization
                if time.time() < deadline - 20.0:
                    optimized = _local_optimize(circles, ww, hh, deadline)
                    optimized, valid2 = _separate_circles(optimized, ww, hh, max_iter=500)
                    if valid2:
                        circles = optimized
                        current_value = float(np.sum(circles[:, 2]))

                # Linearized LP refinement
                if time.time() < deadline - 14.0:
                    refined = _linearized_refine(circles, ww, hh, deadline, max_iter=8)
                    if refined is not None:
                        circles = refined
                        current_value = float(np.sum(circles[:, 2]))

                if current_value > best_value:
                    best_value = current_value
                    best_result = circles.copy()

            except Exception:
                continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])

    return best_result
