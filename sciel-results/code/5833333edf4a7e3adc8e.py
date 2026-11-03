"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

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
    """Grow/relax a hard-disk packing with adaptive timestep."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    prev_max_pen = np.inf
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
        
        # Adaptive timestep: reduce if penetration is increasing
        max_disp = np.max(np.sqrt((disp ** 2).sum(axis=1)))
        if max_disp > 1e-8:
            dt = np.clip(dt * 0.95, 0.005, 0.1)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _get_contact_pairs(c, r, tol=0.02):
    """Get pairs of circles that are close to touching (within tol of 2r)."""
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    mask = d < 2 * r * (1 + tol)
    i, j = np.where(mask)
    return list(set(zip(i.tolist(), j.tolist())))


def _slsqp_epigraph(c0, r0, time_budget=10.0, max_iter=200):
    """SLSQP epigraph formulation: maximize t subject to
    |c_i - c_j| >= 2t for all contact pairs, and wall constraints.
    Uses Delaunay-based contact detection for efficiency."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r0 = max(r0, 0.01)
    start = time.time()
    
    # Get initial contact pairs
    contacts = _get_contact_pairs(c0, r0, tol=0.05)
    
    # Variable: [c_1x, c_1y, ..., c_Nx, c_Ny, t]
    x0 = np.concatenate([c0.ravel(), [r0]])
    
    def objective(x):
        """Minimize -t (maximize t)."""
        return -x[-1]
    
    def objective_grad(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
    
    # Constraints: for each contact pair (i,j), |c_i - c_j| - 2t >= 0
    # And wall constraints: c_ik + t - 0.5 >= 0 and 0.5 - c_ik - t >= 0
    
    def make_constraints():
        cons = []
        # Pairwise constraints for contact pairs
        for (i, j) in contacts:
            def pair_cons(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                t = x[-1]
                dist = np.sqrt(np.sum((ci - cj)**2))
                return dist - 2*t
            def pair_cons_grad(x, i=i, j=j):
                g = np.zeros_like(x)
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                t = x[-1]
                diff = ci - cj
                dist = np.sqrt(np.sum(diff**2)) + 1e-15
                g[2*i:2*i+2] = diff / dist
                g[2*j:2*j+2] = -diff / dist
                g[-1] = -2.0
                return g
            cons.append({'type': 'ineq', 'fun': pair_cons, 'jac': pair_cons_grad})
        
        # Wall constraints: for each point and each wall
        for i in range(N):
            for axis in range(2):
                # Right wall: 0.5 - c_ik - t >= 0
                def right_wall(x, i=i, axis=axis):
                    return 0.5 - x[2*i+axis] - x[-1]
                def right_wall_grad(x, i=i, axis=axis):
                    g = np.zeros_like(x)
                    g[2*i+axis] = -1.0
                    g[-1] = -1.0
                    return g
                cons.append({'type': 'ineq', 'fun': right_wall, 'jac': right_wall_grad})
                
                # Left wall: c_ik + t + 0.5 >= 0
                def left_wall(x, i=i, axis=axis):
                    return x[2*i+axis] + x[-1] + 0.5
                def left_wall_grad(x, i=i, axis=axis):
                    g = np.zeros_like(x)
                    g[2*i+axis] = 1.0
                    g[-1] = 1.0
                    return g
                cons.append({'type': 'ineq', 'fun': left_wall, 'jac': left_wall_grad})
        
        return cons
    
    try:
        res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2*N) + [(r0 * 0.5, 0.5)],
                       constraints=make_constraints(),
                       options={'maxiter': max_iter, 'ftol': 1e-14,
                                'disp': 0})
        c = np.clip(res.x[:2*N].reshape(N, 2), -0.5, 0.5)
        t = res.x[-1]
        s = _true_score(c)
        if s > r0:
            return c, s
        return c0, r0
    except Exception:
        return c0, r0


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start = time.time()

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
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-16, 'gtol': 1e-12,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(8):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        # estimate a starting radius from the layout
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 20:
                break
            cur = _force_relax(cur, r, time_budget=2.0, max_iter=5000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.03
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) SLSQP epigraph optimization - the main refinement
    if time.time() - start_time < DEADLINE - 15:
        # Multiple rounds of SLSQP with updated contact pairs
        for round_num in range(3):
            if time.time() - start_time > DEADLINE - 15:
                break
            # Get current contact pairs
            contacts = _get_contact_pairs(best, best_r, tol=0.03)
            
            # Run SLSQP epigraph
            c_new, r_new = _slsqp_epigraph(best, best_r, 
                                           time_budget=min(15.0, DEADLINE - 15 - (time.time() - start_time)),
                                           max_iter=150)
            if r_new > best_r:
                best_r, best = r_new, c_new
            
            # Force relax to ensure feasibility, then re-optimize
            if time.time() - start_time < DEADLINE - 10:
                best = _force_relax(best, best_r * 0.99, time_budget=2.0, max_iter=3000)
                best_r = _true_score(best)
                
                # Second SLSQP pass with refreshed contacts
                c_new, r_new = _slsqp_epigraph(best, best_r,
                                               time_budget=min(10.0, DEADLINE - 10 - (time.time() - start_time)),
                                               max_iter=100)
                if r_new > best_r:
                    best_r, best = r_new, c_new

    # 3) Soft-min polish for final refinement
    if time.time() - start_time < DEADLINE - 5:
        for k in (200.0, 400.0, 800.0):
            if time.time() - start_time > DEADLINE - 5:
                break
            c = _softmin_refine(best, k, time_budget=3.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 4) Final SLSQP polish with very tight tolerance
    if time.time() - start_time < DEADLINE - 2:
        contacts = _get_contact_pairs(best, best_r, tol=0.02)
        c_new, r_new = _slsqp_epigraph(best, best_r,
                                       time_budget=min(5.0, DEADLINE - 2 - (time.time() - start_time)),
                                       max_iter=50)
        if r_new > best_r:
            best_r, best = r_new, c_new

    # Ensure final validity
    best = np.clip(best, -0.5, 0.5)
    
    return best
# EVOLVE-BLOCK-END
