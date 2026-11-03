"""Rewritten packing program: Langevin dynamics + active-set contact refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import cKDTree

DEADLINE = 115.0  # safety margin before the 150s kill


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
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _langevin_dynamics(c0, r, time_budget=8.0, max_iter=200000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r.
    Uses velocity-Verlet with thermal noise to escape local minima.
    Returns a feasible layout for radius r if found."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    
    start = time.time()
    dt = 0.003
    gamma = 0.15  # friction coefficient
    T_init = 0.015
    T_min = 0.000005
    n = max_iter
    
    best_c = c.copy()
    best_score = _true_score(c)
    
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Temperature schedule: geometric cooling
        T = T_init * (T_min / T_init) ** (i / max(1, n))
        
        # Compute forces
        disp = np.zeros_like(c)
        
        # Pairwise overlap repulsion (using KDTree for efficiency)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        
        if len(pairs) > 0:
            for idx in pairs:
                ii, jj = idx[0], idx[1]
                diff = c[ii] - c[jj]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[ii] += force
                    disp[jj] -= force
        
        # Wall forces
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        
        # Langevin update: velocity-Verlet with noise
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        
        # Position update
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        
        # Check feasibility
        feasible = True
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        if len(pairs_new) > 0:
            feasible = False
        
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                # Accept with Metropolis criterion if close
                if rng.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            # Reject, keep velocity but damp
            v *= 0.5
    
    return best_c


def _active_set_refine(c0, r, time_budget=5.0):
    """Refine using active-set method: identify tight constraints and solve
    a small QP to maximize minimum distance while maintaining contacts."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    
    for _ in range(10):
        if time.time() - start > time_budget:
            break
        
        # Find tight constraints
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        
        w = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        
        min_pair = d.min()
        min_wall = w.min()
        
        # Identify active constraints (within tolerance)
        tol = 1e-6
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(i, j) for i, j in active_pairs if i < j]
        
        active_walls = []
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    pen = c[k, axis] - (0.5 - r)
                else:
                    pen = (-0.5 + r) - c[k, axis]
                if pen < tol:
                    active_walls.append((k, axis, sign))
        
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        
        # Build a simple Newton step: move points to increase min distance
        # For each active pair, push apart along the connecting vector
        disp = np.zeros_like(c)
        for i, j in active_pairs:
            diff = c[i] - c[j]
            d_ij = np.linalg.norm(diff)
            if d_ij > 1e-12:
                force = diff / d_ij * 0.001
                disp[i] += force
                disp[j] -= force
        
        for k, axis, sign in active_walls:
            disp[k, axis] += sign * 0.001
        
        # Apply small step
        c = np.clip(c + 0.01 * disp, -0.5 + r, 0.5 - r)
    
    return c


def _softmin_refine(c0, k, time_budget=3.0):
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
                       options={'maxiter': 500, 'maxfun': 15000,
                                'ftol': 1e-18, 'gtol': 1e-12,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


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
        # pairwise overlap repulsion using KDTree for speed
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-9, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                i, j = idx[0], idx[1]
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
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


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(10):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 25:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(12):
            if time.time() - start_time > DEADLINE - 25:
                break
            cur = _force_relax(cur, r, time_budget=1.2, max_iter=4000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.04
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin dynamics to escape local minima at current best radius
    if time.time() - start_time < DEADLINE - 18:
        ld_budget = min(12.0, DEADLINE - 18 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=150000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur
            # Try a second Langevin pass from a slightly perturbed version
            if time.time() - start_time < DEADLINE - 15 and ld_budget > 6.0:
                rng_ld2 = np.random.default_rng(999)
                perturbed = best + rng_ld2.normal(0, 0.002, size=best.shape)
                perturbed = np.clip(perturbed, -0.5 + best_r, 0.5 - best_r)
                cur2 = _langevin_dynamics(perturbed, best_r, time_budget=min(5.0, DEADLINE - 15 - (time.time() - start_time)), max_iter=80000, rng=rng_ld2)
                s2 = _true_score(cur2)
                if s2 > best_r:
                    best_r, best = s2, cur2
    
    # 3) Active-set refinement on the best layout
    if time.time() - start_time < DEADLINE - 10:
        as_budget = min(5.0, DEADLINE - 10 - (time.time() - start_time))
        if as_budget > 1.0:
            cur = _active_set_refine(best, best_r, time_budget=as_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) Smooth soft-min polish from the best layout (and a couple others).
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=2000))
    for c0 in inits[:4]:
        if time.time() - start_time > DEADLINE - 12:
            break
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=2000))

    # Progressive k refinement: start moderate, increase for precision
    for k in (60.0, 120.0, 200.0, 350.0):
        for c0 in candidates:
            if time.time() - start_time > DEADLINE - 6:
                break
            c = _softmin_refine(c0, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c
                candidates[0] = best  # update best candidate for next k

    # 5) Final high-precision polish on the best layout with tighter tolerance
    if time.time() - start_time < DEADLINE - 3:
        c = _softmin_refine(best, 500.0, time_budget=5.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
        # One more ultra-tight pass
        if time.time() - start_time < DEADLINE - 1:
            c = _softmin_refine(best, 800.0, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
