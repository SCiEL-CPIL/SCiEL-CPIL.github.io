"""Packing 50 equal circles in unit square via force-relax + softmin + active-set."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
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
    g = np.array([(x, y) for y in ticks for x in ticks])  # 49 points
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    """Generate a hexagonal-like initial configuration."""
    pts = []
    spacing = 0.14
    y = -0.45
    row = 0
    while len(pts) < N and y < 0.45:
        x_offset = 0 if row % 2 == 0 else spacing * 0.866
        x = -0.45 + x_offset
        while x < 0.45 and len(pts) < N:
            pts.append([x, y])
            x += spacing
        y += spacing * 0.866
        row += 1
    if len(pts) < N:
        # Pad with random points
        rng = np.random.default_rng(0)
        while len(pts) < N:
            pts.append(rng.uniform(-0.45, 0.45, size=2).tolist())
    return np.array(pts[:N])


def _random_init(rng):
    return rng.uniform(-0.45, 0.45, size=(N, 2))


def _force_relax(c, r, time_budget=3.0, max_iter=15000):
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


def _softmin_refine(c0, k, time_budget=4.0):
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
                       options={'maxiter': 150, 'maxfun': 4000,
                                'ftol': 1e-13, 'gtol': 1e-9,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _active_set_refine(c0, r_init, time_budget=5.0):
    """
    Active-set based refinement: identify the binding constraints (pairs and walls
    that determine the minimum radius) and optimize only those degrees of freedom
    while keeping others fixed or slightly relaxed.
    
    This is a structural change: instead of optimizing all 100 coordinates at once,
    we focus on the critical constraints that limit the radius.
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)
    start = time.time()
    
    # Compute current constraints
    def compute_constraints(pts, t):
        """Return list of (type, indices, value) for all constraints."""
        cons = []
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
                cons.append(('pair', (i, j), d / 2.0 - t))
        # Walls
        for i in range(N):
            for axis in range(2):
                cons.append(('wall_pos', (i, axis, 1), 0.5 - pts[i, axis] - t))
                cons.append(('wall_neg', (i, axis, -1), pts[i, axis] + 0.5 - t))
        return cons
    
    # Find active constraints (those with small slack)
    def find_active(pts, t, threshold=0.001):
        cons = compute_constraints(pts, t)
        active = [c for c in cons if c[2] < threshold]
        return active
    
    # Build reduced optimization problem using only active constraints
    # We'll use a sequential approach: identify active set, then solve
    # a QP-like problem for the active variables
    
    c = c0.copy()
    t = r_init
    
    for iteration in range(10):
        if time.time() - start > time_budget:
            break
        
        # Find active constraints
        active = find_active(c, t, threshold=0.002)
        
        if len(active) == 0:
            break
        
        # Determine which points are involved in active constraints
        active_points = set()
        for con_type, indices, _ in active:
            if con_type == 'pair':
                active_points.add(indices[0])
                active_points.add(indices[1])
            else:
                active_points.add(indices[0])
        
        if len(active_points) < 2:
            break
        
        # Sort active points
        active_idx = sorted(active_points)
        
        # For now, do a simple gradient step on active points
        # to reduce violation of active constraints
        disp = np.zeros_like(c)
        
        for con_type, indices, slack in active:
            if slack >= 0:
                continue  # Constraint satisfied
            
            if con_type == 'pair':
                i, j = indices
                diff = c[i] - c[j]
                d = np.sqrt(np.sum(diff ** 2))
                if d < 1e-12:
                    continue
                # Push apart
                force = diff / d * (-slack) * 0.5
                disp[i] += force
                disp[j] -= force
            elif con_type == 'wall_pos':
                i, axis, sign = indices
                disp[i, axis] += sign * (-slack) * 0.5
            elif con_type == 'wall_neg':
                i, axis, sign = indices
                disp[i, axis] += sign * (-slack) * 0.5
        
        # Apply small step
        step_size = 0.01
        c = np.clip(c + step_size * disp, -0.5, 0.5)
        
        # Update t to be the minimum constraint value
        cons = compute_constraints(c, 0)
        t = min(c[2] for c in cons)
    
    return c


def _langevin_dynamics(c0, r_target, time_budget=5.0):
    """
    Langevin dynamics with cooling to find a dense packing.
    Uses random perturbations to escape local minima.
    """
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    start = time.time()
    
    # Temperature schedule
    T0 = 0.01
    T_final = 0.0001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        
        # Cooling schedule
        T = T0 * (T_final / T0) ** (step / n_steps)
        
        # Compute forces
        disp = np.zeros_like(c)
        
        # Pairwise repulsion
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r_target
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r_target - d) / (2 * r_target)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        
        # Wall repulsion
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r_target)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r_target) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r_target)
        
        # Langevin step: deterministic + stochastic
        noise = np.random.normal(0, np.sqrt(2 * T * 0.01), size=c.shape)
        c = np.clip(c + 0.01 * disp + noise, -0.5 + r_target, 0.5 - r_target)
    
    return c


def construct():
    deadline = time.time() + DEADLINE
    
    best_score = -np.inf
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_grid_init())
    inits.append(_hex_init())
    for _ in range(8):
        inits.append(_random_init(rng))
    
    for init in inits:
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Force relax to get a feasible packing
        # Start with a conservative radius estimate
        r_est = 0.05
        c_relaxed = _force_relax(init, r_est, time_budget=2.0)
        
        # Phase 2: Langevin dynamics to escape local minima
        r_target = 0.06
        c_langevin = _langevin_dynamics(c_relaxed, r_target, time_budget=4.0)
        
        # Phase 3: Soft-min refinement with increasing k
        c_refined = c_langevin
        for k in [50, 100, 200, 500]:
            if time.time() > deadline - 8:
                break
            c_refined = _softmin_refine(c_refined, k, time_budget=3.0)
        
        # Phase 4: Active-set refinement for final polish
        r_current = _true_score(c_refined)
        if r_current > 0:
            c_final = _active_set_refine(c_refined, r_current, time_budget=3.0)
            r_final = _true_score(c_final)
            if r_final > r_current:
                c_refined = c_final
        
        # Evaluate
        score = _true_score(c_refined)
        if score > best_score:
            best_score = score
            best_pts = c_refined.copy()
    
    # Final polish with high k if time permits
    if best_pts is not None and time.time() < deadline - 5:
        for k in [1000, 2000]:
            if time.time() > deadline - 2:
                break
            best_pts = _softmin_refine(best_pts, k, time_budget=2.0)
        
        score = _true_score(best_pts)
        if score > best_score:
            best_score = score
    
    if best_pts is None:
        # Fallback: simple grid
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
