import numpy as np
import time

N = 21


def _inradii(sites, W, H):
    """Inradius of each Voronoi cell (site + rectangle boundary edges)."""
    n = len(sites)
    xs = sites[:, 0]
    ys = sites[:, 1]
    r = np.empty(n)
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        # distance to the 4 rectangle walls
        r[i] = min(xi, W - xi, yi, H - yi)
        # distance to each neighbor's bisector
        for j in range(n):
            if i == j:
                continue
            dx = xs[j] - xi
            dy = ys[j] - yi
            d2 = dx * dx + dy * dy
            if d2 < 1e-14:
                continue
            d = np.sqrt(d2)
            # |projection of (site_j - site_i) onto direction (site_j - site_i)| = d
            # distance to bisector = d/2
            r[i] = min(r[i], 0.5 * d)
    return r


def _nm_maximize(obj, x0, bounds, max_iter=1200, step=0.06, min_step=1e-5,
                 time_limit=None):
    """Nelder-Mead maximization with box clipping and a time limit."""
    n = len(x0)
    # initial simplex
    simplex = [x0.copy()]
    for k in range(n):
        p = x0.copy()
        p[k] = p[k] + step
        simplex.append(p)

    def clip(v):
        return np.clip(v, bounds[0], bounds[1])

    f = [obj(clip(p)) for p in simplex]
    order = np.argsort(f)  # ascending (worst first)
    simplex = [simplex[i].copy() for i in order]
    f = [f[i] for i in order]

    best_val = max(f)
    best_x = simplex[-1].copy()

    it = 0
    while it < max_iter:
        if time_limit is not None and time.time() > time_limit:
            break
        # worst is simplex[0]
        worst = simplex[0]
        f_worst = f[0]
        # centroid of all but worst
        cen = np.mean(simplex[1:], axis=0)
        # reflection
        alpha = 1.0
        xr = clip(cen + alpha * (cen - worst))
        fr = obj(xr)
        if fr >= f[1]:  # better than second worst
            if fr < f[-1]:  # better than best
                gamma = 2.0
                xe = clip(cen + gamma * (cen - worst))
                fe = obj(xe)
                if fe > fr:
                    simplex[0] = xe
                    f[0] = fe
                else:
                    simplex[0] = xr
                    f[0] = fr
            else:
                simplex[0] = xr
                f[0] = fr
        else:
            # contraction or shrink
            beta = 0.5
            xc = clip(worst + beta * (cen - worst))
            fc = obj(xc)
            if fc > f_worst:
                simplex[0] = xc
                f[0] = fc
            else:
                # shrink toward best
                pb = simplex[-1]
                for idx in range(n):
                    if idx == 0:
                        continue
                    simplex[idx] = clip(pb + 0.5 * (simplex[idx] - pb))
                    f[idx] = obj(simplex[idx])
                simplex[0] = clip(pb + 0.5 * (worst - pb))
                f[0] = obj(simplex[0])

        # re-sort
        order = np.argsort(f)
        simplex = [simplex[i].copy() for i in order]
        f = [f[i] for i in order]

        if f[-1] > best_val:
            best_val = f[-1]
            best_x = simplex[-1].copy()

        # convergence check
        spread = max(f) - min(f)
        if spread < min_step * 0.01:
            break
        it += 1

    return best_x, best_val


def _hex_init(W, H, seed):
    """Hexagonal lattice initialization spread over the rectangle."""
    rng = np.random.RandomState(seed)
    cols = 5
    rows = 5
    xs = np.linspace(0.12 * W, 0.88 * W, cols)
    ys = np.linspace(0.12 * H, 0.88 * H, rows)
    sites = []
    for j in range(rows):
        for i in range(cols):
            x = xs[i] + (0.5 * xs[1] - xs[0]) * (j % 2)
            y = ys[j]
            sites.append([x, y])
    sites = np.array(sites[:N])
    # small jitter
    sites += rng.uniform(-0.01, 0.01, sites.shape)
    return sites


def _grid_init(W, H, seed):
    rng = np.random.RandomState(seed)
    cols = 5
    rows = 5
    xs = np.linspace(0.15 * W, 0.85 * W, cols)
    ys = np.linspace(0.15 * H, 0.85 * H, rows)
    sites = np.array([[x, y] for j in range(rows) for i in range(cols)])[:N]
    sites += rng.uniform(-0.015, 0.015, sites.shape)
    return sites


def _random_init(W, H, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1 * W, 0.9 * W, (N, 2))


def _valid(sites, W, H, tol=1e-7):
    """Check that the inscribed-disk packing is valid (should always hold by
    construction, but verify numerically)."""
    r = _inradii(sites, W, H)
    if np.any(r < -tol):
        return False
    r = np.maximum(r, 0.0)
    # boundary containment
    if np.any(sites[:, 0] + r > W + tol) or np.any(sites[:, 0] - r < -tol):
        return False
    if np.any(sites[:, 1] + r > H + tol) or np.any(sites[:, 1] - r < -tol):
        return False
    # pairwise non-overlap
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = sites[i_idx, 0] - sites[j_idx, 0]
    dy = sites[i_idx, 1] - sites[j_idx, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - tol):
        return False
    return True


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin under 150s

    best_sum = 0.0
    best_out = None

    aspect_ratios = [1.0, 1.1, 0.9, 1.2, 0.85, 1.3, 0.8, 1.4, 0.75, 1.5]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        # W + H = 2, W/H = ar  =>  W = 2*ar/(1+ar), H = 2/(1+ar)
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)

        bounds = (np.array([0.0, 0.0]), np.array([W, H]))

        inits = [
            _hex_init(W, H, 1000 + int(ar * 100)),
            _grid_init(W, H, 2000 + int(ar * 100)),
            _random_init(W, H, 3000 + int(ar * 100)),
        ]

        for init_sites in inits:
            if time.time() > deadline:
                break
            # objective: maximize sum of inradii
            def obj(v, W=W, H=H):
                return float(np.sum(_inradii(v, W, H)))

            # quick pre-eval
            pre = obj(init_sites)
            if pre > best_sum:
                best_sum = pre
                r = _inradii(init_sites, W, H)
                cand = np.column_stack([init_sites, r])
                if _valid(init_sites, W, H):
                    best_out = cand

            # Nelder-Mead from this init
            res_x, res_val = _nm_maximize(
                obj, init_sites, bounds,
                max_iter=1400, step=0.05,
                time_limit=deadline - 2.0,
            )
            if res_val > best_sum:
                best_sum = res_val
                r = _inradii(res_x, W, H)
                cand = np.column_stack([res_x, r])
                if _valid(res_x, W, H):
                    best_out = cand

    # Fallback: guaranteed-valid hex packing
    if best_out is None:
        W, H = 1.0, 1.0
        sites = _hex_init(W, H, 7)
        r = _inradii(sites, W, H)
        best_out = np.column_stack([sites, r])

    return best_out
