"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26


def _compute_radii(x, y):
    """Radius of each circle = min(dist to nearest other circle/2, dist to walls)."""
    n = len(x)
    if n <= 1:
        return np.full(n, 0.5)
    d = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / 2.0
    wall = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.maximum(0.0, np.minimum(nn, wall))


def _objective_from_positions(x, y):
    r = _compute_radii(x, y)
    return r.sum()


def _penalty_objective(z, alpha=500.0):
    """
    z: [x1..xN, y1..yN, r1..rN]
    Objective: -sum(r) + alpha * (overlap_penalty + wall_penalty)
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    
    # Wall constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    wall_margin = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = np.maximum(0.0, r - wall_margin)
    wall_pen = np.sum(wall_viol ** 2)
    
    # Overlap constraints: d_ij >= r_i + r_j
    pos = np.column_stack([x, y])
    d = cdist(pos, pos)
    # Pairwise radii sum
    r_sum = r[:, None] + r[None, :]
    overlap_viol = np.maximum(0.0, r_sum - d)
    # Only count each pair once (upper triangle)
    iu = np.triu_indices(N, k=1)
    overlap_pen = np.sum(overlap_viol[iu] ** 2)
    
    return -np.sum(r) + alpha * (overlap_pen + wall_pen)


def _slsqp_constraints(z):
    """
    Return list of constraint dicts for SLSQP.
    g_i(z) >= 0
    """
    x = z[:N]
    y = z[N:2*N]
    r = z[2*N:3*N]
    
    constraints = []
    
    # Wall constraints
    wall_margin = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: wall_margin[i] - r[i]})
        # Wait, this lambda closure is wrong for variable wall_margin.
        # Need to recompute or use a different structure.
        # SLSQP constraints are functions of z.
        # Let's define them properly.
    
    # Actually, defining N*4 + N*(N-1)/2 constraints is too many and slow for SLSQP.
    # Better to use penalty method with L-BFGS-B for the main optimization,
    # and only a few key constraints if using SLSQP, or just L-BFGS-B with penalty.
    
    # Let's stick to L-BFGS-B with a strong penalty. It's faster and more robust for this.
    # The previous attempt at SLSQP with many constraints is likely to be slow/unstable.
    
    # So, we will just use L-BFGS-B with penalty.
    # We need a good initialization.
    
    return None # Not used in this version


def _init_hexagonal(n, jitter=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    rows = int(np.ceil(np.sqrt(n * 0.866)))
    pts = []
    for j in range(rows):
        y = (j + 0.5) / rows
        offset = (j % 2) * 0.5
        cols = int(np.ceil(np.sqrt(n) * 1.15))
        for i in range(cols):
            x = (i + 0.5 + offset) / (cols + 0.5)
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
        if len(pts) >= n:
            break
    if len(pts) < n:
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                pts.append(((i + 0.5) / side, (j + 0.5) / side))
    pts = pts[:n]
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    if jitter > 0:
        x += rng.uniform(-jitter, jitter, n)
        y += rng.uniform(-jitter, jitter, n)
    x = np.clip(x, 0.01, 0.99)
    y = np.clip(y, 0.01, 0.99)
    
    # Initial radii: small value
    r = np.full(n, 0.02)
    return np.concatenate([x, y, r])


def _init_random(n, rng):
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    return np.concatenate([x, y, r])


def _optimize_from(z0, alpha=100.0, time_budget=10.0):
    """Run L-BFGS-B from z0."""
    t0 = time.time()
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
    
    # We can't easily pass time_budget to minimize, so we run it and hope it finishes.
    # L-BFGS-B usually converges reasonably fast.
    res = minimize(_penalty_objective, z0, method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-9})
    
    return res.x, res.fun


def construct():
    t_start = time.time()
    DEADLINE = 140.0
    rng = np.random.default_rng(2024)
    
    best_z = None
    best_score = -1.0
    
    # Try multiple initializations and alpha values
    inits = []
    # Hexagonal with varying jitter
    for j in [0.0, 0.01, 0.02, 0.05]:
        inits.append(_init_hexagonal(N, jitter=j, rng=rng))
    # Random inits
    for _ in range(4):
        inits.append(_init_random(N, rng))
    
    alphas = [100.0, 500.0, 1000.0]
    
    for init in inits:
        if time.time() - t_start > DEADLINE - 20.0:
            break
        for alpha in alphas:
            if time.time() - t_start > DEADLINE - 15.0:
                break
            # Run optimizer
            # Note: _penalty_objective uses a fixed alpha internally? No, I defined it with default.
            # I need to pass alpha or define a closure.
            
            # Let's define a closure for the objective with the specific alpha
            obj = lambda z, a=alpha: _penalty_objective(z, a)
            
            bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
            res = minimize(obj, init, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            
            z_opt = res.x
            # Evaluate true score (sum of valid radii)
            x = z_opt[:N]
            y = z_opt[N:2*N]
            r_computed = _compute_radii(x, y)
            
            # Check validity: are the proposed r_i <= r_computed?
            # If not, the penalty wasn't high enough, or we are infeasible.
            # For scoring, we use the *actual* max radii based on positions.
            # So the score is sum(r_computed).
            score = np.sum(r_computed)
            
            if score > best_score:
                best_score = score
                best_z = z_opt.copy()
                # Store x, y for final computation
                best_x, best_y = x.copy(), y.copy()
                
    if best_x is None:
        # Fallback
        z0 = _init_hexagonal(N, jitter=0.0, rng=rng)
        best_x = z0[:N]
        best_y = z0[N:2*N]
        best_score = _objective_from_positions(best_x, best_y)
        
    # Polish the best solution with a higher alpha to tighten constraints
    if time.time() - t_start < DEADLINE - 5.0:
        # Use the best x, y and their max possible radii as start
        r_start = _compute_radii(best_x, best_y)
        z_start = np.concatenate([best_x, best_y, r_start])
        
        alpha_final = 5000.0
        obj_final = lambda z, a=alpha_final: _penalty_objective(z, a)
        bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N
        
        res = minimize(obj_final, z_start, method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12})
        
        z_polished = res.x
        x_p = z_polished[:N]
        y_p = z_polished[N:2*N]
        r_p_computed = _compute_radii(x_p, y_p)
        score_p = np.sum(r_p_computed)
        
        if score_p > best_score:
            best_x, best_y = x_p, y_p
            best_score = score_p
            
    # Final post-processing
    r = _compute_radii(best_x, best_y)
    # Ensure strict validity
    r = np.maximum(0.0, r - 1e-9)
    best_x = np.clip(best_x, r, 1.0 - r)
    best_y = np.clip(best_y, r, 1.0 - r)
    r = np.minimum(r, np.minimum(np.minimum(best_x, 1.0 - best_x), np.minimum(best_y, 1.0 - best_y)))
    
    out = np.column_stack([best_x, best_y, r])
    return out
