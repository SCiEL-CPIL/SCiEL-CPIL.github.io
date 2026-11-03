"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.optimize import LinearConstraint

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _force_relax(c, r, time_budget=4.0, max_iter=20000):
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        # pairwise overlap repulsion
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            # unit vectors from j to i
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        # wall push
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _softmin_refine(c0, k, time_budget=3.0, maxiter=200, maxfun=5000,
                    ftol=1e-15, gtol=1e-10):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k

        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': maxiter, 'maxfun': maxfun,
                                'ftol': ftol, 'gtol': gtol,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _delaunay_contact_graph(c, r):
    """Return set of (i,j) pairs that are near-contact in the Delaunay graph."""
    try:
        tri = Delaunay(c)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = int(simplex[a]), int(simplex[b])
                    edges.add((min(i, j), max(i, j)))
        # Filter to near-contact edges (within 2.5*r to capture potential contacts)
        near = set()
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            if d < 2.5 * r:
                near.add((i, j))
        return near
    except Exception:
        # Fallback: all pairs within 2.8*r
        near = set()
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2.8 * r:
                    near.add((i, j))
        return near


def _slsqp_epigraph(c0, r0, time_budget=5.0, maxiter=200, ftol=1e-14):
    """SLSQP epigraph formulation: maximize t subject to
    t <= |c_i - c_j|/2 for all near-contact pairs,
    t <= wall distances for all points.
    Uses vectorized constraints from Delaunay contact graph."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)
    
    # Get contact graph
    contacts = _delaunay_contact_graph(c0, r0)
    n_contacts = len(contacts)
    
    def obj(x):
        # x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, t]
        return -x[-1]  # minimize -t = maximize t
    
    def obj_grad(x):
        g = np.zeros(2 * N + 1)
        g[-1] = -1.0
        return g
    
    # Vectorized inequality constraints: all >= 0
    # d_ij/2 - t >= 0 for contact pairs
    # wall_dist_i - t >= 0 for each point and each wall
    
    def constraint_fun(x):
        pts = x[:2*N].reshape(N, 2)
        t = x[-1]
        # Pairwise: d_ij/2 - t
        pair_cons = np.zeros(n_contacts)
        for idx, (i, j) in enumerate(contacts):
            d = np.linalg.norm(pts[i] - pts[j])
            pair_cons[idx] = d / 2.0 - t
        # Wall: for each point, distance to each of 4 walls minus t
        wall_cons = np.zeros(4 * N)
        wall_cons[0::4] = pts[:, 0] + 0.5 - t      # left wall
        wall_cons[1::4] = 0.5 - pts[:, 0] - t      # right wall
        wall_cons[2::4] = pts[:, 1] + 0.5 - t      # bottom wall
        wall_cons[3::4] = 0.5 - pts[:, 1] - t      # top wall
        return np.concatenate([pair_cons, wall_cons])
    
    def constraint_jac(x):
        pts = x[:2*N].reshape(N, 2)
        n_total = n_contacts + 4 * N
        J = np.zeros((n_total, 2 * N + 1))
        
        # Pairwise constraint gradients
        for idx, (i, j) in enumerate(contacts):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                continue
            u = (pts[i] - pts[j]) / d
            J[idx, 2*i] = 0.5 * u[0]
            J[idx, 2*i+1] = 0.5 * u[1]
            J[idx, 2*j] = -0.5 * u[0]
            J[idx, 2*j+1] = -0.5 * u[1]
            J[idx, -1] = -1.0
        
        # Wall constraint gradients
        base = n_contacts
        for i in range(N):
            J[base + 4*i, 2*i] = 1.0
            J[base + 4*i, -1] = -1.0
            J[base + 4*i+1, 2*i] = -1.0
            J[base + 4*i+1, -1] = -1.0
            J[base + 4*i+2, 2*i+1] = 1.0
            J[base + 4*i+2, -1] = -1.0
            J[base + 4*i+3, 2*i+1] = -1.0
            J[base + 4*i+3, -1] = -1.0
        
        return J
    
    x0 = np.concatenate([c0.ravel(), [r0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    cons = {'type': 'ineq', 'fun': constraint_fun, 'jac': constraint_jac}
    
    try:
        start = time.time()
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol, 'iprint': 0})
        if time.time() - start > time_budget:
            return c0
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
        return c
    except Exception:
        return c0


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 12:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 12:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=4000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.05
                hi = r
            else:
                hi = r
                if hi - lo < 1e-6:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.5, max_iter=2000))
    for c0 in inits[:4]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.5, max_iter=2000))

    for k in (80.0, 150.0, 250.0, 400.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 3) Delaunay-based SLSQP epigraph polish on the best layout
    if time.time() - start_time < DEADLINE - 10:
        c = _slsqp_epigraph(best, best_r, time_budget=5.0, maxiter=200, ftol=1e-14)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 4) Final high-precision polish on the best layout with tighter tolerance
    if time.time() - start_time < DEADLINE - 6:
        c = _softmin_refine(best, 300.0, time_budget=3.0,
                            maxiter=400, maxfun=12000,
                            ftol=1e-18, gtol=1e-12)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Second SLSQP epigraph pass after soft-min polish
    if time.time() - start_time < DEADLINE - 4:
        c = _slsqp_epigraph(best, best_r, time_budget=4.0, maxiter=200, ftol=1e-15)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 6) Ultra-high-precision final pass
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 500.0, time_budget=2.5,
                            maxiter=300, maxfun=10000,
                            ftol=1e-18, gtol=1e-13)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 7) Final SLSQP with expanded contact graph for maximum precision
    if time.time() - start_time < DEADLINE - 1:
        c = _slsqp_epigraph(best, best_r, time_budget=2.0, maxiter=150, ftol=1e-16)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
