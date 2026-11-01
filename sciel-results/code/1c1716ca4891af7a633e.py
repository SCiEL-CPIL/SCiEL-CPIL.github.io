import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def _elapsed():
        return time.time() - start_time

    def _is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1.0 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1.0 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def _sum_r(result):
        return float(np.sum(result[2*N:]))

    def _make_valid(result):
        xy = result[:2*N].reshape(N, 2).copy()
        r = result[2*N:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1.0 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1.0 - r)
        return np.concatenate([xy.flatten(), r])

    def _hex_init(rng):
        n = N
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row, col = i // 6, i % 6
            x = (col + 0.5) / 6.0 + (0.25 / 6.0 if row % 2 == 1 else 0.0)
            y = (row + 0.5) / 6.0
            xy[i] = [x, y]
            r[i] = 0.073
        return np.concatenate([xy.flatten(), r])

    def _rect_init(rng, rows=6, cols=6):
        n = N
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            row, col = i // cols, i % cols
            xy[i] = [(col + 0.5) / cols, (row + 0.5) / rows]
            r[i] = 0.5 / cols - 0.003
        return np.concatenate([xy.flatten(), r])

    def _random_init(rng, lo=0.08, hi=0.92, rlo=0.055, rhi=0.085):
        n = N
        xy = rng.uniform(lo, hi, size=(n, 2))
        r = rng.uniform(rlo, rhi, size=n)
        return np.concatenate([xy.flatten(), r])

    def _objective(params, beta=50.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        def _smooth_max_neg(arr, beta=beta):
            return np.logaddexp(0.0, -beta * arr) / beta

        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r

        boundary_pen = (np.sum(_smooth_max_neg(d_left)) +
                        np.sum(_smooth_max_neg(d_right)) +
                        np.sum(_smooth_max_neg(d_bottom)) +
                        np.sum(_smooth_max_neg(d_top)))

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap_viol = min_dist[iu] - dist[iu]
        overlap_pen = np.sum(_smooth_max_neg(-overlap_viol))

        return -np.sum(r) + 100.0 * (boundary_pen + overlap_pen)

    def _objective_tight(params, beta=100.0):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        def _smooth_max_neg(arr, beta=beta):
            return np.logaddexp(0.0, -beta * arr) / beta

        d_left = xy[:, 0] - r
        d_right = 1.0 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1.0 - xy[:, 1] - r

        boundary_pen = (np.sum(_smooth_max_neg(d_left)) +
                        np.sum(_smooth_max_neg(d_right)) +
                        np.sum(_smooth_max_neg(d_bottom)) +
                        np.sum(_smooth_max_neg(d_top)))

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap_viol = min_dist[iu] - dist[iu]
        overlap_pen = np.sum(_smooth_max_neg(-overlap_viol))

        return -np.sum(r) + 500.0 * (boundary_pen + overlap_pen)

    def _push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1.0 - wi
                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap
                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1.0 - r[i]:
                    xy[i, 0] = 1.0 - r[i]; moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1.0 - r[i]:
                    xy[i, 1] = 1.0 - r[i]; moved = True
            if not moved:
                break
        return np.concatenate([xy.flatten(), r])

    def _optimize(params, max_iter=200):
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-14})
            return res.x
        except Exception:
            return params

    def _tight_polish(params, n_perturb=5, max_iter=300):
        best = params.copy()
        best_score = _sum_r(best)
        bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N

        for p in range(n_perturb):
            if _elapsed() > deadline - 5:
                break
            # Small random perturbation
            rng_p = np.random.RandomState(hash((id(params), p)) % 2**32)
            perturbed = params.copy()
            xy_p = perturbed[:2*N].reshape(N, 2)
            r_p = perturbed[2*N:]
            xy_p += rng_p.normal(0, 0.005, size=(N, 2))
            xy_p[:, 0] = np.clip(xy_p[:, 0], r_p, 1.0 - r_p)
            xy_p[:, 1] = np.clip(xy_p[:, 1], r_p, 1.0 - r_p)
            perturbed[:2*N] = xy_p.flatten()

            try:
                res = minimize(_objective_tight, perturbed, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-15})
                if _is_valid(res.x):
                    s = _sum_r(res.x)
                    if s > best_score:
                        best = res.x.copy()
                        best_score = s
            except Exception:
                pass

            # Also try without perturbation but tighter objective
            if _elapsed() > deadline - 5:
                break
            try:
                res2 = minimize(_objective_tight, best, method='L-BFGS-B',
                                bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-15})
                if _is_valid(res2.x):
                    s2 = _sum_r(res2.x)
                    if s2 > best_score:
                        best = res2.x.copy()
                        best_score = s2
            except Exception:
                pass

        return best

    rng = np.random.RandomState(42)
    best_result = None
    best_score = -np.inf

    # Phase 1: Diverse initializations
    inits = []
    # Hexagonal
    inits.append(_hex_init(rng))
    # Perturbed hex
    h = _hex_init(rng).copy()
    h[:2*N] += rng.normal(0, 0.01, size=(N, 2)).flatten()
    inits.append(h)
    # Rectangular grids
    inits.append(_rect_init(rng, 6, 6))
    inits.append(_rect_init(rng, 8, 4))
    inits.append(_rect_init(rng, 4, 8))
    inits.append(_rect_init(rng, 7, 5))
    inits.append(_rect_init(rng, 5, 7))
    # Perturbed rects
    for base_rows, base_cols in [(6, 6), (8, 4), (4, 8)]:
        rc = _rect_init(rng, base_rows, base_cols).copy()
        rc[:2*N] += rng.normal(0, 0.008, size=(N, 2)).flatten()
        inits.append(rc)
    # Random
    for _ in range(10):
        inits.append(_random_init(rng))

    n_restarts = len(inits)
    for restart in range(n_restarts):
        if _elapsed() > deadline - 30:
            break

        params = inits[restart]
        params = _push_apart(params, n_iters=100)
        params = _optimize(params, max_iter=200)

        if _is_valid(params):
            score = _sum_r(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

    # Phase 2: Tight polish on best
    if best_result is not None and _elapsed() < deadline - 10:
        best_result = _tight_polish(best_result, n_perturb=8, max_iter=300)
        if _is_valid(best_result):
            best_score = _sum_r(best_result)

    # Phase 3: Additional restarts with tight polish if time allows
    if _elapsed() < deadline - 20:
        for extra in range(15):
            if _elapsed() > deadline - 15:
                break
            params = _random_init(rng)
            params = _push_apart(params, n_iters=80)
            params = _optimize(params, max_iter=150)
            if _is_valid(params):
                score = _sum_r(params)
                if score > best_score:
                    best_score = score
                    best_result = params.copy()
            # Quick tight polish
            if _elapsed() < deadline - 10:
                polished = _tight_polish(params, n_perturb=3, max_iter=200)
                if _is_valid(polished):
                    s = _sum_r(polished)
                    if s > best_score:
                        best_score = s
                        best_result = polished.copy()

    # Final validation
    if best_result is not None:
        final = _make_valid(best_result)
        if _is_valid(final):
            return np.column_stack([final[:2*N].reshape(N, 2), final[2*N:]])

    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row, col = i // 6, i % 6
        cells.append([(col + 0.5) / 6, (row + 0.5) / 6, 0.5 / 6 - 1e-3])
    return np.array(cells)
