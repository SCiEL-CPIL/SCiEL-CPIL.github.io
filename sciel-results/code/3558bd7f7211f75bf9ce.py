"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


def _initial_layout(n, seed=0):
    """Generate a decent initial layout using a jittered grid / hex-like arrangement."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    # start with a grid
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            pts.append([x, y])
    pts = np.array(pts[:n])
    # add small jitter to break symmetry
    jitter = 0.02 / side
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    return pts


def _build_constraint_pairs(pts, k=6):
    """Find nearest-neighbor pairs using a k-nearest approach."""
    n = len(pts)
    if n < 2:
        return []
    d = cdist(pts, pts)
    pairs = []
    for i in range(n):
        idx = np.argsort(d[i])
        for j in idx[1:k+1]:
            a, b = min(i, j), max(i, j)
            if (a, b) not in pairs:
                pairs.append((a, b))
    return pairs


def _objective_and_grad(x, n, pairs):
    """Penalty objective: minimize -(sum radii) + penalties for violations.
    x = [x0..x_{n-1}, y0..y_{n-1}, r0..r_{n-1}]
    """
    pos = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    # boundary: r <= min(x, 1-x, y, 1-y)
    # circle-circle: dist >= r_i + r_j
    penalty = 0.0
    eps = 1e-9
    # boundary violations
    for i in range(n):
        r = radii[i]
        x0, y0 = pos[i]
        for c in (x0, 1 - x0, y0, 1 - y0):
            if r > c:
                penalty += (r - c) ** 2
    # circle-circle violations
    for (a, b) in pairs:
        d = np.linalg.norm(pos[a] - pos[b])
        s = radii[a] + radii[b]
        if s > d:
            penalty += (s - d) ** 2
    val = -np.sum(radii) + 1e4 * penalty
    return val


def _check_valid(pts, radii, tol=1e-7):
    """Check all circles are inside square and non-overlapping."""
    for i in range(len(pts)):
        x, y = pts[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_and_clamp(pts, radii):
    """Clamp positions/radii to feasible region (approx)."""
    radii = np.maximum(radii, 0.0)
    for i in range(len(pts)):
        r = radii[i]
        x, y = pts[i]
        # ensure inside square
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        pts[i] = [x, y]
    return pts, radii


def _optimize_from(start_pts, start_radii, deadline):
    """Iteratively grow radii and relax positions using SLSQP with contact pairs."""
    n = len(start_pts)
    pts = start_pts.copy()
    radii = start_radii.copy()

    def pack(x):
        pos = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        return pos, r

    # Start with a small common radius
    r0 = 0.02
    radii[:] = r0

    best_sum = np.sum(radii)
    best_state = (pts.copy(), radii.copy())

    max_iters = 300
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Build pairs based on current positions
        pairs = _build_constraint_pairs(pts, k=5)

        # Objective: minimize -sum(r) + penalty
        def obj(x):
            pos, r = pack(x)
            pen = 0.0
            for i in range(n):
                r_i = r[i]
                x0, y0 = pos[i]
                for c in (x0, 1 - x0, y0, 1 - y0):
                    if r_i > c:
                        pen += (r_i - c) ** 2
            for (a, b) in pairs:
                d = np.linalg.norm(pos[a] - pos[b])
                s = r[a] + r[b]
                if s > d:
                    pen += (s - d) ** 2
            return -np.sum(r) + 1e5 * pen

        def obj_grad(x):
            pos, r = pack(x)
            g = np.zeros_like(x)
            # gradient of -sum(r)
            g[2*n:] = -1.0
            # boundary penalty gradients
            for i in range(n):
                r_i = r[i]
                x0, y0 = pos[i]
                # x - r >= 0
                if r_i > x0:
                    g[i] += 2e5 * (r_i - x0) * (-1)   # d/dx of (r-x)^2
                    g[2*n + i] += 2e5 * (r_i - x0) * (1)
                if r_i > 1 - x0:
                    g[i] += 2e5 * (r_i - (1 - x0)) * (1)
                    g[2*n + i] += 2e5 * (r_i - (1 - x0)) * (-1)
                if r_i > y0:
                    g[n + i] += 2e5 * (r_i - y0) * (-1)
                    g[2*n + i] += 2e5 * (r_i - y0) * (1)
                if r_i > 1 - y0:
                    g[n + i] += 2e5 * (r_i - (1 - y0)) * (1)
                    g[2*n + i] += 2e5 * (r_i - (1 - y0)) * (-1)
            # circle-circle
            for (a, b) in pairs:
                dvec = pos[a] - pos[b]
                d = np.linalg.norm(dvec)
                if d < 1e-12:
                    continue
                s = r[a] + r[b]
                if s > d:
                    diff = s - d
                    grad_d = dvec / d
                    g[a] += 2e5 * diff * grad_d
                    g[n + a] += 0
                    g[b] -= 2e5 * diff * grad_d
                    g[n + b] += 0
                    g[2*n + a] += 2e5 * diff
                    g[2*n + b] += 2e5 * diff
            return g

        x0 = np.concatenate([pts.ravel(), radii])
        # Bounds: positions in [0,1], radii in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1))
        for i in range(n):
            bounds.append((0, 1))
        for i in range(n):
            bounds.append((0, 0.5))

        try:
            res = minimize(obj, x0, method='L-BFGS-B', jac=obj_grad,
                           bounds=bounds, options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < obj(x0):
                pos, r = pack(res.x)
                # clamp
                pos, r = _fix_and_clamp(pos, r)
                pts, radii = pos, r
                cur_sum = np.sum(radii)
                if cur_sum > best_sum and _check_valid(pts, radii, tol=1e-6):
                    best_sum = cur_sum
                    best_state = (pts.copy(), radii.copy())
        except Exception:
            pass

    # Try to grow radii a bit more with a final local push
    # Simple: try to increase each radius slightly and fix
    return best_state


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s

    best_state = None
    best_sum = -1.0

    # Try multiple seeds / initial layouts
    for seed in range(6):
        if time.time() > deadline:
            break
        pts = _initial_layout(N, seed=seed)
        radii = np.full(N, 0.015)
        state = _optimize_from(pts, radii, deadline)
        if state is None:
            continue
        p, r = state
        if not _check_valid(p, r, tol=1e-6):
            # try to repair by shrinking
            for i in range(N):
                r[i] = min(r[i], p[i][0], 1 - p[i][0], p[i][1], 1 - p[i][1])
            if not _check_valid(p, r, tol=1e-6):
                continue
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_state = (p, r)

    if best_state is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    p, r = best_state
    out = np.zeros((N, 3))
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 1]
    out[:, 2] = r
    return out
# EVOLVE-BLOCK-END
