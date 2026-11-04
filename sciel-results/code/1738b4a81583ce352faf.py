"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
DEADLINE = 145.0  # seconds, leave 5s margin for 150s kill


def construct():
    start_time = time.time()

    def _remaining(margin=5.0):
        return DEADLINE - (time.time() - start_time) - margin

    def _score(centers):
        """Exact sum of max radii for fixed centers (vectorized)."""
        c = np.asarray(centers, dtype=float)
        radii = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                           np.minimum(c[:, 1], 1.0 - c[:, 1]))
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2, axis=1)) / 2.0
        radii = np.minimum(radii, min_pair)
        return np.sum(radii)

    def _radii_of(centers):
        c = np.asarray(centers, dtype=float)
        radii = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                           np.minimum(c[:, 1], 1.0 - c[:, 1]))
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2, axis=1)) / 2.0
        radii = np.minimum(radii, min_pair)
        return radii

    def _is_valid(result, tol=1e-7):
        x, y, r = result[:, 0], result[:, 1], result[:, 2]
        if np.any(x - r < -tol) or np.any(x + r > 1.0 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1.0 + tol):
            return False
        if np.any(r < -tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def _make_result(centers):
        c = np.asarray(centers, dtype=float)
        r = _radii_of(c)
        return np.column_stack([c, r])

    # ---------- Tension network via Delaunay + smooth optimization ----------
    def _delaunay_edges(centers):
        """Return list of (i,j) edges from Delaunay triangulation."""
        c = np.asarray(centers, dtype=float)
        try:
            tri = Delaunay(c)
            edges = set()
            for simplex in tri.simplices:
                for a in range(3):
                    i, j = simplex[a], simplex[(a + 1) % 3]
                    edges.add((min(i, j), max(i, j)))
            return list(edges)
        except Exception:
            # Fallback: all pairs
            return [(i, j) for i in range(N) for j in range(i + 1, N)]

    def _tension_obj(x, edges, beta=80.0):
        """
        x layout: [centers (2N), radii (N)]
        Minimize: sum over edges of (r_i + r_j - d_ij)^2  (overlap penalty)
               + sum over all i of (r_i - dist_to_wall_i)^2  (wall penalty)
               - lambda * sum(r_i)  (reward for large radii)
        This is a smooth surrogate that drives toward a valid packing.
        """
        c = x[:2 * N].reshape(-1, 2)
        r = x[2 * N:]
        
        # Wall distances
        wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                          np.minimum(c[:, 1], 1.0 - c[:, 1]))
        
        # Overlap penalty on Delaunay edges
        overlap_penalty = 0.0
        for (i, j) in edges:
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            gap = r[i] + r[j] - d
            if gap > 0:
                overlap_penalty += gap * gap
            else:
                # Small attraction to encourage touching
                overlap_penalty += 0.001 * gap * gap
        
        # Wall penalty
        wall_penalty = 0.0
        for i in range(N):
            gap = r[i] - wall[i]
            if gap > 0:
                wall_penalty += gap * gap
            else:
                wall_penalty += 0.001 * gap * gap
        
        # Reward for large radii
        reward = 0.5 * np.sum(r)
        
        return overlap_penalty + wall_penalty - reward

    def _tension_grad(x, edges, beta=80.0):
        c = x[:2 * N].reshape(-1, 2)
        r = x[2 * N:]
        grad = np.zeros_like(x)
        
        wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                          np.minimum(c[:, 1], 1.0 - c[:, 1]))
        
        # Wall gradient
        for i in range(N):
            gap = r[i] - wall[i]
            coeff = 2.0 * gap if gap > 0 else 0.002 * gap
            # d wall_i / d c_i
            if c[i, 0] < 0.5:
                grad[2 * i] += coeff * 1.0
            else:
                grad[2 * i] += coeff * (-1.0)
            if c[i, 1] < 0.5:
                grad[2 * i + 1] += coeff * 1.0
            else:
                grad[2 * i + 1] += coeff * (-1.0)
            # d wall_penalty / d r_i
            grad[2 * N + i] += coeff
        
        # Edge gradient
        for (i, j) in edges:
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                continue
            gap = r[i] + r[j] - d
            coeff = 2.0 * gap if gap > 0 else 0.002 * gap
            
            # d gap / d c_i = (c_i - c_j) / d
            grad[2 * i] += -coeff * dx / d
            grad[2 * i + 1] += -coeff * dy / d
            grad[2 * j] += coeff * dx / d
            grad[2 * j + 1] += coeff * dy / d
            
            # d gap / d r_i = 1, d gap / d r_j = 1
            grad[2 * N + i] += coeff
            grad[2 * N + j] += coeff
        
        # Reward gradient: d(-0.5 * sum(r))/d r_i = -0.5
        grad[2 * N:] += -0.5
        
        return grad

    def _tension_optimize(centers, max_iter=200):
        c = np.asarray(centers, dtype=float).copy()
        c = np.clip(c, 0.01, 0.99)
        r0 = _radii_of(c)
        x0 = np.concatenate([c.flatten(), r0])
        bounds = [(0.01, 0.99)] * (2 * N) + [(0.0, 0.5)] * N
        
        # Get initial Delaunay edges
        edges = _delaunay_edges(c)
        
        try:
            # First pass
            res = minimize(_tension_obj, x0, jac=lambda x: _tension_grad(x, edges),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
            c = res.x[:2 * N].reshape(-1, 2)
            r = res.x[2 * N:]
            c = np.clip(c, 0.01, 0.99)
            r = np.clip(r, 0.0, 0.5)
            
            # Second pass with updated Delaunay edges
            edges2 = _delaunay_edges(c)
            x1 = np.concatenate([c.flatten(), r])
            res2 = minimize(_tension_obj, x1, jac=lambda x: _tension_grad(x, edges2),
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': max_iter // 2, 'gtol': 1e-8, 'ftol': 1e-10})
            c = res2.x[:2 * N].reshape(-1, 2)
            r = res2.x[2 * N:]
            c = np.clip(c, 0.01, 0.99)
            r = np.clip(r, 0.0, 0.5)
            
            return c
        except Exception:
            return c

    # ---------- Exact high-precision per-circle subproblem ----------
    def _radius_at(c, i, others):
        p = c[i]
        r = min(p[0], 1.0 - p[0], p[1], 1.0 - p[1])
        if others.shape[0] > 0:
            d = np.hypot(p[0] - others[:, 0], p[1] - others[:, 1])
            r = min(r, np.min(d) / 2.0)
        return r

    def _solve_circle(c, i, max_iter=50):
        """Maximize radius of circle i with others fixed (active-set)."""
        pos = c[i].copy()
        others = np.delete(c, i, axis=0)
        cur_r = _radius_at(c, i, others)
        for _ in range(max_iter):
            if _remaining(2.0) < 0:
                break
            tol_a = 1e-11
            active_g = []
            for val, g in [(pos[0], np.array([1.0, 0.0])),
                           (1.0 - pos[0], np.array([-1.0, 0.0])),
                           (pos[1], np.array([0.0, 1.0])),
                           (1.0 - pos[1], np.array([0.0, -1.0]))]:
                if abs(val - cur_r) < tol_a:
                    active_g.append(g)
            if others.shape[0] > 0:
                d = np.hypot(pos[0] - others[:, 0], pos[1] - others[:, 1])
                for k in range(others.shape[0]):
                    if abs(d[k] / 2.0 - cur_r) < tol_a:
                        if d[k] < 1e-12:
                            continue
                        g = np.array([(pos[0] - others[k, 0]) / (2.0 * d[k]),
                                      (pos[1] - others[k, 1]) / (2.0 * d[k])])
                        active_g.append(g)
            if len(active_g) < 2:
                break
            gs = np.mean(active_g, axis=0)
            gn = np.linalg.norm(gs)
            if gn < 1e-13:
                break
            gs = gs / gn
            improved = False
            for ss in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                np_ = pos + gs * ss
                if np_.min() < -0.02 or np_.max() > 1.02:
                    continue
                c[i] = np_
                nr = _radius_at(c, i, others)
                if nr > cur_r + 1e-13:
                    pos = np_
                    cur_r = nr
                    improved = True
                    break
            if not improved:
                break
        c[i] = pos
        return pos, cur_r

    def _high_precision_polish(centers, max_sweeps=100):
        c = np.array(centers, dtype=float).copy()
        for sweep in range(max_sweeps):
            if _remaining(3.0) < 0:
                break
            improved = False
            for i in range(N):
                pos, r = _solve_circle(c, i, max_iter=35)
                if np.linalg.norm(pos - c[i]) > 1e-13:
                    c[i] = pos
                    improved = True
            if not improved:
                break
        return c

    # ---------- Initializers ----------
    def _grid_init():
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < N:
                    pts.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(pts[:N])

    def _hex_init():
        pts = []
        a = 0.18
        row = 0
        while len(pts) < N + 20:
            y = 0.06 + row * a * np.sqrt(3) / 2
            if y > 0.95:
                break
            xoff = (a / 2) if (row % 2) else 0.0
            x = 0.06 + xoff
            col = 0
            while x <= 0.95 and len(pts) < N + 20:
                pts.append([x, y])
                x += a
                col += 1
            row += 1
        if len(pts) < N:
            return _grid_init()
        arr = np.array(pts)
        sc = np.minimum(np.minimum(arr[:, 0], 1.0 - arr[:, 0]),
                        np.minimum(arr[:, 1], 1.0 - arr[:, 1]))
        idx = np.argsort(-sc)[:N]
        return arr[idx]

    def _random_init(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(0.08, 0.92, size=(N, 2))

    def _jittered_grid(seed, jitter=0.03):
        rng = np.random.RandomState(seed)
        base = _grid_init()
        base = base + rng.uniform(-jitter, jitter, size=base.shape)
        return np.clip(base, 0.05, 0.95)

    def _repulsion_relax(centers, iters=300, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        c = np.array(centers, dtype=float).copy()
        n = N
        for it in range(iters):
            if _remaining(2.0) < 0:
                break
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c[i, 0] - c[j, 0]
                    dy = c[i, 1] - c[j, 1]
                    d = np.hypot(dx, dy)
                    if d < 1e-9:
                        dx, dy = rng.normal(0, 0.01), rng.normal(0, 0.01)
                        d = np.hypot(dx, dy)
                    target = 0.18
                    if d < target:
                        f = (target - d) / d * 0.5
                        c[i, 0] += dx * f
                        c[i, 1] += dy * f
                        c[j, 0] -= dx * f
                        c[j, 1] -= dy * f
            for i in range(n):
                if c[i, 0] < 0.1:
                    c[i, 0] += (0.1 - c[i, 0]) * 0.5
                if c[i, 0] > 0.9:
                    c[i, 0] -= (c[i, 0] - 0.9) * 0.5
                if c[i, 1] < 0.1:
                    c[i, 1] += (0.1 - c[i, 1]) * 0.5
                if c[i, 1] > 0.9:
                    c[i, 1] -= (c[i, 1] - 0.9) * 0.5
            c = np.clip(c, 0.03, 0.97)
        return c

    # ---------- Best-fit local search ----------
    def _best_fit(centers, max_sweeps=15):
        c = np.array(centers, dtype=float).copy()
        n = N
        for sweep in range(max_sweeps):
            if _remaining(3.0) < 0:
                break
            improved = False
            for i in range(n):
                others = np.delete(c, i, axis=0)
                cur_r = _radius_at(c, i, others)
                best_r = cur_r
                best_pos = c[i].copy()
                rng = np.random.RandomState(1000 + i + sweep)
                for _ in range(30):
                    np_ = c[i] + rng.uniform(-max(cur_r * 0.4, 0.015),
                                             max(cur_r * 0.4, 0.015), size=2)
                    if np_.min() < 0.0 or np_.max() > 1.0:
                        continue
                    r = min(
                        min(np_[0], 1.0 - np_[0], np_[1], 1.0 - np_[1]),
                        (np.min(np.hypot(np_[0] - others[:, 0], np_[1] - others[:, 1])) / 2.0)
                        if others.shape[0] else 0.5)
                    if r > best_r + 1e-12:
                        best_r = r
                        best_pos = np_
                if np.linalg.norm(best_pos - c[i]) > 1e-12:
                    c[i] = best_pos
                    improved = True
            if not improved:
                break
        return c

    # ---------- Driver ----------
    best_result = None
    best_score = -np.inf

    def _update(centers):
        nonlocal best_result, best_score
        r = _radii_of(centers)
        res = np.column_stack([centers, r])
        if _is_valid(res, tol=1e-6):
            s = np.sum(r)
            if s > best_score:
                best_score = s
                best_result = res
                return True
        return False

    # Phase 0: structured inits -> repulsion -> tension optimize -> polish
    init_candidates = []
    init_candidates.append(_grid_init())
    init_candidates.append(_hex_init())
    for s in [1, 2, 3]:
        init_candidates.append(_jittered_grid(s))
    for s in [0, 5, 11, 29, 47, 61, 97, 131, 211, 333, 555, 777]:
        init_candidates.append(_random_init(s))

    for ic in init_candidates:
        if _remaining(25.0) < 0:
            break
        c = ic.copy()
        c = _repulsion_relax(c, iters=150)
        c = _tension_optimize(c, max_iter=180)
        c = _best_fit(c, max_sweeps=8)
        _update(c)

    # Phase 1: high-precision polish on best
    if best_result is not None:
        if _remaining(20.0) > 0:
            c = best_result[:, :2].copy()
            c = _high_precision_polish(c, max_sweeps=80)
            _update(c)

    # Phase 2: perturb best region + tension optimize + polish
    if best_result is not None:
        base = best_result[:, :2].copy()
        for s in [10, 20, 30, 40, 50, 60]:
            if _remaining(15.0) < 0:
                break
            rng = np.random.RandomState(5000 + s)
            c = base + rng.normal(0, 0.04, size=base.shape)
            c = np.clip(c, 0.03, 0.97)
            c = _tension_optimize(c, max_iter=150)
            c = _best_fit(c, max_sweeps=6)
            _update(c)

    # Phase 3: final polish cycles
    if best_result is not None:
        for _ in range(4):
            if _remaining(8.0) < 0:
                break
            c = best_result[:, :2].copy()
            c = _tension_optimize(c, max_iter=150)
            c = _high_precision_polish(c, max_sweeps=50)
            if not _update(c):
                break

    # Final safety pass
    if best_result is not None and _remaining(3.0) > 0:
        c = best_result[:, :2].copy()
        c = _high_precision_polish(c, max_sweeps=40)
        _update(c)

    # Fallback
    if best_result is None:
        c = _grid_init()
        r = _radii_of(c)
        best_result = np.column_stack([c, r])

    if not _is_valid(best_result, tol=1e-6):
        c = _grid_init()
        r = _radii_of(c)
        best_result = np.column_stack([c, r])

    return best_result
