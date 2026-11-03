"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius(c):
    """Compute the maximum common radius for centers c of shape (N,2)."""
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    diff = c[iu[0]] - c[iu[1]]
    d = np.sqrt((diff ** 2).sum(axis=1))
    rp = d.min() / 2.0
    rw = np.minimum(np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                    np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)).min()
    return min(rp, rw)


def _init_hexagonal():
    """Hexagonal lattice of ~50 points inside the square."""
    pts = []
    # hexagonal lattice: rows offset
    # spacing s ~ 0.14 (radius ~0.07)
    s = 0.14
    # generate grid over [-0.5, 0.5]
    xs = np.arange(-0.5 + s / 2, 0.5, s)
    for row, y in enumerate(np.arange(-0.5 + s / 2, 0.5, s * np.sqrt(3) / 2)):
        offset = (s / 2) if row % 2 else 0.0
        for x in xs:
            xx = x + offset
            if -0.5 <= xx <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([xx, y])
    pts = np.array(pts)
    if len(pts) >= N:
        # pick N closest to a regular grid? just take first N after shuffling by distance from center
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists)[:N]
        return pts[idx].copy()
    else:
        # fallback: 7x7 grid
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        g = np.array([(x, y) for y in ticks for x in ticks])
        return g[:N].copy()


def _expand_radius(c, R0, R1, step, deadline, t0):
    """Incrementally grow radius, resolving collisions by pushing apart."""
    c = c.copy()
    R = R0
    n = len(c)
    while R < R1:
        if (time.time() - t0) > deadline - 10:
            break
        # resolve collisions at radius R
        for _ in range(200):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d < 2 * R - 1e-12:
                        if d < 1e-12:
                            diff = np.array([1e-6, 1e-6])
                            d = np.linalg.norm(diff)
                        push = (2 * R - d) / d * 0.5
                        c[i] += push * diff
                        c[j] -= push * diff
                        moved = True
            # wall constraints
            for i in range(n):
                c[i, 0] = np.clip(c[i, 0], -0.5 + R, 0.5 - R)
                c[i, 1] = np.clip(c[i, 1], -0.5 + R, 0.5 - R)
            if not moved:
                break
        R += step
    return c


def _slsqp_polish(c, t0, deadline):
    """SLSQP optimization to maximize r subject to constraints."""
    n = c.shape[0]
    x0 = np.concatenate([c.flatten(), [_radius(c)]])
    # variables: 2n centers + 1 radius r
    # objective: minimize -r
    def obj(x):
        return -x[-1]

    def grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g

    # constraints: r - d_ij/2 <= 0 for all pairs, r - wall_i <= 0
    # To keep number of constraints manageable, use only active-ish constraints? 
    # SLSQP can handle many, but 50*49/2=1225 pair constraints + 50 wall = 1275 constraints
    # That might be too slow. Let's use a subset or a different approach.
    # Alternative: use only the minimum constraints? No, that's non-smooth.
    # Let's try with all constraints but limit iterations.
    iu = np.triu_indices(n, k=1)
    pairs = list(zip(iu[0], iu[1]))
    constraints = []
    for i, j in pairs:
        def con(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            d = np.linalg.norm(ci - cj)
            return r - d / 2.0
        constraints.append({'type': 'ineq', 'fun': con})
    for i in range(n):
        def con(x, i=i):
            ci = x[2*i:2*i+2]
            r = x[-1]
            # wall distances: min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
            w = np.min([0.5 - ci[0], ci[0] + 0.5, 0.5 - ci[1], ci[1] + 0.5])
            return r - w
        constraints.append({'type': 'ineq', 'fun': con})

    bounds = [(-0.5, 0.5)] * (2 * n) + [(0.0, 0.5)]
    # SLSQP with many constraints is slow. Let's try with maxiter and time check.
    # This might be too slow for 1275 constraints. Let's reduce by only using constraints 
    # that are close to active? Or use a different method.
    # Given the time limit, let's try a simpler polish: local search on the radius.
    # Actually, let's just do a few iterations of SLSQP with a reduced set of constraints.
    # Or skip SLSQP if it's too slow and just return the expanded result.
    # Let's try SLSQP with a timeout.
    try:
        # Use only a subset of constraints: the 100 closest pairs and all walls
        # Compute current distances
        c_cur = x0[:2*n].reshape(n, 2)
        iu = np.triu_indices(n, k=1)
        diff = c_cur[iu[0]] - c_cur[iu[1]]
        d = np.sqrt((diff ** 2).sum(axis=1))
        # Get the 200 smallest distances
        idx_small = np.argsort(d)[:200]
        active_pairs = [(iu[0][k], iu[1][k]) for k in idx_small]
        constraints = []
        for i, j in active_pairs:
            def con(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                r = x[-1]
                d = np.linalg.norm(ci - cj)
                return r - d / 2.0
            constraints.append({'type': 'ineq', 'fun': con})
        for i in range(n):
            def con(x, i=i):
                ci = x[2*i:2*i+2]
                r = x[-1]
                w = np.min([0.5 - ci[0], ci[0] + 0.5, 0.5 - ci[1], ci[1] + 0.5])
                return r - w
            constraints.append({'type': 'ineq', 'fun': con})

        res = minimize(obj, x0, method='SLSQP', jac=grad, bounds=bounds,
                       constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        if res.success or (time.time() - t0) < deadline - 5:
            c_new = res.x[:2*n].reshape(n, 2)
            r_new = _radius(c_new)
            r_old = _radius(c)
            if r_new > r_old:
                return c_new
    except Exception:
        pass
    return c


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin

    # Start with hexagonal initialization
    c = _init_hexagonal()
    
    # Expand radius incrementally
    R0 = 0.05
    R1 = 0.072
    step = 0.001
    c = _expand_radius(c, R0, R1, step, deadline, t0)
    
    # Polish with SLSQP
    if (time.time() - t0) < deadline - 20:
        c = _slsqp_polish(c, t0, deadline)
    
    # Ensure validity
    c = np.clip(c, -0.5 + 1e-15, 0.5 - 1e-15)
    if not np.all(np.isfinite(c)) or _radius(c) <= 0:
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return c
# EVOLVE-BLOCK-END
