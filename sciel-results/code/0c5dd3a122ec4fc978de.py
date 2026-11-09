"""Hexagonal-lattice init + repulsive-force relaxation + smooth soft-min polish."""
import numpy as np
import time

N = 200


def _score(pts):
    """Exact radius: min over pairs (dist/2) and walls (0.5 - |coord|)."""
    pts = np.asarray(pts, dtype=np.float64)
    # pairwise
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pairs = dist.min() / 2.0
    r_walls = (0.5 - np.abs(pts)).min()
    return min(r_pairs, r_walls)


def _soft_min_radius(pts, beta, t):
    """
    Smooth surrogate objective: we want to maximize t subject to
    soft constraints that pairwise dist >= 2t and wall >= t.
    We use a log-sum-exp style soft-max of the 'violations' and subtract.
    Objective to maximize:  t - (1/beta) * logsumexp(beta * violations)
    where violations_i = (2t - d_i) for pairs, and (t - w_i) for walls.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    # violations for pairs: 2t - dist (want <= 0)
    viol_pairs = 2.0 * t - dist
    # walls: t - (0.5 - |x|) = t - 0.5 + |x|, want <= 0  =>  (t - (0.5 - |x|))
    wall_margin = 0.5 - np.abs(pts)  # shape (n,2)
    viol_walls = t - wall_margin
    viol = np.concatenate([viol_pairs.ravel(), viol_walls.ravel()])
    # logsumexp of beta*viol
    b = beta * viol
    bmax = b.max()
    lse = bmax + np.log(np.sum(np.exp(b - bmax)))
    return t - (1.0 / beta) * lse


def _soft_min_radius_grad(pts, beta, t):
    """Gradient of the soft objective w.r.t. pts and t."""
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    viol_pairs = 2.0 * t - dist
    wall_margin = 0.5 - np.abs(pts)
    viol_walls = t - wall_margin
    viol = np.concatenate([viol_pairs.ravel(), viol_walls.ravel()])
    b = beta * viol
    bmax = b.max()
    e = np.exp(b - bmax)
    s = e.sum()
    w = e / s  # softmax weights
    # d objective / d pts
    # objective = t - (1/beta)*logsumexp(beta*viol)
    # d/dpts = -(1/beta)*sum_i w_i * beta * dviol_i/dpts = -sum_i w_i dviol_i/dpts
    grad_pts = np.zeros_like(pts)
    grad_t = 1.0
    # pairs
    w_pairs = w[:n * (n - 1)].reshape(n, n)
    # dviol_pairs_ij/dp_i = d(2t - dist)/dp_i = -d(dist)/dp_i = + d_ij / dist  (since dist = |p_i - p_j|, d(dist)/dp_i = (p_i-p_j)/dist)
    # d(dist)/dp_i = (p_i - p_j)/dist = d_ij/dist
    # so dviol/dp_i = - (d_ij/dist)
    # contribution to grad_pts[i] = -sum_j w_ij * (-d_ij/dist) = sum_j w_ij * d_ij/dist
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wij = w_pairs[i, j]
            if wij == 0.0:
                continue
            dij = d[i, j]  # p_i - p_j
            distij = dist[i, j]
            grad_pts[i] += wij * dij / distij
            # symmetric: dviol/dp_j = - d(dist)/dp_j = -(-d_ij/dist) = +d_ij/dist ... wait
            # d(dist)/dp_j = (p_j - p_i)/dist = -dij/dist
            # dviol/dp_j = -d(dist)/dp_j = +dij/dist
            # grad_pts[j] += -sum_i w_ij * dviol_ij/dp_j = -sum_i w_ij*(dij/dist)
            grad_pts[j] -= wij * dij / distij
    # walls
    w_walls = w[n * (n - 1):].reshape(n, 2)
    # viol_walls = t - (0.5 - |x|) ; dviol/dx = -d(0.5-|x|)/dx = -(-sign(x)) = sign(x)
    # grad_pts += -sum w * sign(x)
    for i in range(n):
        for k in range(2):
            wwk = w_walls[i, k]
            if wwk == 0.0:
                continue
            grad_pts[i, k] -= wwk * np.sign(pts[i, k])
            grad_t -= wwk  # dviol/dt = 1, grad_t = 1 - sum w * 1
    return grad_pts, grad_t


def _generate_hexagonal_grid():
    """Hexagonal (triangular) lattice points inside [-0.5,0.5]^2, pick N closest to centre."""
    # spacing so that ~N points fit; hex packing: horizontal spacing a, vertical a*sqrt(3)/2
    # area per point ~ a^2 * sqrt(3)/2. We want ~N points in area 1 => a ~ sqrt(2/(N*sqrt(3)))
    a = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    # generate over a slightly larger range to be safe
    x0 = -0.5 - a
    x1 = 0.5 + a
    y0 = -0.5 - a
    y1 = 0.5 + a
    nx = int(np.ceil((x1 - x0) / a)) + 1
    dy = a * np.sqrt(3.0) / 2.0
    ny = int(np.ceil((y1 - y0) / dy)) + 1
    pts = []
    for j in range(ny):
        y = y0 + j * dy
        offset = (a / 2.0) if (j % 2) else 0.0
        for i in range(nx):
            x = x0 + i * a + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < N:
        # fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
    # pick N closest to origin (densest region)
    d2 = np.sum(pts * pts, axis=1)
    idx = np.argsort(d2)
    return pts[idx[:N]]


def _repulsive_relax(pts, n_iter=300, dt=0.02):
    """Simple repulsive force: push apart pairs closer than 2r_est, keep inside walls."""
    pts = np.array(pts, dtype=np.float64)
    n = pts.shape[0]
    # estimate target radius from hex packing
    r_est = np.sqrt(1.0 / (N * np.sqrt(3.0) * 2.0))  # ~0.0335
    min_dist = 2.0 * r_est
    for it in range(n_iter):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dij = pts[i] - pts[j]
                dist = np.sqrt(np.sum(dij * dij))
                if dist < 1e-12:
                    # jitter
                    forces[i] += 0.001 * np.random.randn(2)
                    forces[j] -= 0.001 * np.random.randn(2)
                    continue
                if dist < min_dist:
                    # repulsion strength grows as they overlap
                    f = (min_dist - dist) / dist
                    forces[i] += f * dij
                    forces[j] -= f * dij
        # wall repulsion
        for i in range(n):
            for k in range(2):
                if pts[i, k] > 0.5 - r_est:
                    forces[i, k] -= (pts[i, k] - (0.5 - r_est)) * 5.0
                elif pts[i, k] < -0.5 + r_est:
                    forces[i, k] += ((-0.5 + r_est) - pts[i, k]) * 5.0
        pts += dt * forces
        # clamp inside
        pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
    return pts


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin under 150s (and 120s eval)
    np.random.seed(12345)

    # 1) Hexagonal init
    pts = _generate_hexagonal_grid()
    best_pts = pts.copy()
    best_score = _score(pts)

    # 2) Repulsive relaxation (fast, coarse)
    pts = _repulsive_relax(pts, n_iter=200, dt=0.03)
    s = _score(pts)
    if s > best_score:
        best_score, best_pts = s, pts.copy()

    # 3) Smooth soft-min gradient polish (L-BFGS-like via scipy)
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None:
        beta = 200.0
        # epigraph variable t; maximize objective => minimize -objective
        x0 = np.concatenate([best_pts.ravel(), [best_score * 0.9]])
        def neg_obj(x):
            p = x[:2 * N].reshape(N, 2)
            t = x[2 * N]
            # keep t bounded
            t = np.clip(t, 0.0, 0.5)
            return -_soft_min_radius(p, beta, t)
        def neg_obj_grad(x):
            p = x[:2 * N].reshape(N, 2)
            t = x[2 * N]
            t = np.clip(t, 0.0, 0.5)
            gp, gt = _soft_min_radius_grad(p, beta, t)
            return np.concatenate([gp.ravel(), [-gt]])
        for attempt in range(3):
            if time.time() > deadline:
                break
            try:
                res = minimize(neg_obj, x0, jac=neg_obj_grad, method='L-BFGS-B',
                               options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50})
                p = res.x[:2 * N].reshape(N, 2)
                p = np.clip(p, -0.5 + 1e-9, 0.5 - 1e-9)
                s = _score(p)
                if s > best_score:
                    best_score, best_pts = s, p.copy()
                # refine from this result
                x0 = np.concatenate([p.ravel(), [np.clip(res.x[2 * N], 0.0, 0.5)]])
            except Exception:
                break

    # 4) Final exact local polish: coordinate-descent style nudge on the min-constraint
    #    (tighter tolerance, a few passes)
    for _ in range(3):
        if time.time() > deadline:
            break
        p = best_pts.copy()
        # find the bottleneck: smallest pair or wall
        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        wall = 0.5 - np.abs(p)
        r_pairs = dist / 2.0
        # identify critical pairs (dist < 2*(best+eps)) and nudge them apart
        for it in range(50):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dij = p[i] - p[j]
                    distij = np.sqrt(np.sum(dij * dij))
                    if distij < 2.0 * best_score + 1e-7:
                        # nudge apart slightly
                        if distij < 1e-12:
                            continue
                        nudge = dij / distij * 1e-5
                        p[i] += nudge
                        p[j] -= nudge
                        moved = True
            # wall fix
            for i in range(N):
                for k in range(2):
                    if p[i, k] > 0.5 - best_score - 1e-7:
                        p[i, k] = 0.5 - best_score - 1e-7
                        moved = True
                    elif p[i, k] < -0.5 + best_score + 1e-7:
                        p[i, k] = -0.5 + best_score + 1e-7
                        moved = True
            p = np.clip(p, -0.5 + 1e-9, 0.5 - 1e-9)
            if not moved:
                break
        s = _score(p)
        if s > best_score:
            best_score, best_pts = s, p.copy()

    return best_pts
