"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import pdist, cdist


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    # Distance to walls
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal lattice initialization for 50 points in unit square."""
    points = []
    dx = 0.12
    dy = dx * np.sqrt(3) / 2
    
    row = 0
    while len(points) < N:
        offset = (dx / 2) if (row % 2 == 1) else 0
        x = -HALF + 0.05 + offset
        while x < HALF - 0.05:
            y = -HALF + 0.05 + row * dy
            if -HALF <= x <= HALF and -HALF <= y <= HALF:
                points.append([x, y])
            x += dx
        row += 1
    
    if len(points) >= N:
        return np.array(points[:N])
    else:
        return _grid_init()


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _smooth_penalty(c, r, alpha=100.0):
    """
    Smooth penalty function for the SNE approach.
    Returns a scalar that is small when all constraints are satisfied.
    """
    c = c.reshape(N, 2)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    
    # Penalty for pair distances < 2r
    pair_violation = np.maximum(0, 2*r - d)
    pair_penalty = np.sum(pair_violation**2)
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    wall_violation = np.maximum(0, r - wall_dists)
    wall_penalty = np.sum(wall_violation**2)
    
    # Keep in bounds
    bound_violation = np.maximum(0, np.abs(c) - (HALF - 1e-9))
    bound_penalty = np.sum(bound_violation**2) * 1000
    
    return pair_penalty + wall_penalty + bound_penalty


def _sne_objective(vars):
    """
    SNE formulation: find positions and r such that active constraints are exactly met.
    vars = [c_0x, c_0y, ..., c_49x, c_49y, r]
    """
    c_flat = vars[:2*N]
    r = vars[2*N]
    c = c_flat.reshape(N, 2)
    
    # Compute all pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    
    # Find active constraints (those close to being violated)
    tol = 0.001
    active_pairs = []
    active_walls = []
    
    for i in range(N):
        for j in range(i+1, N):
            if d[i,j] < 2*r + tol:
                active_pairs.append((i, j))
        
        for w in range(4):
            if wall_dists[i, w] < r + tol:
                active_walls.append((i, w))
    
    # Build residual vector: for each active constraint, residual = distance - target
    residuals = []
    
    for (i, j) in active_pairs:
        # Residual: |c_i - c_j| - 2r = 0
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            residuals.append(dist - 2*r)
        else:
            residuals.append(2*r)  # Avoid division by zero
    
    for (i, w) in active_walls:
        # Residual: wall_dist - r = 0
        residuals.append(wall_dists[i, w] - r)
    
    if len(residuals) == 0:
        return np.array([0.0])
    
    return np.array(residuals)


def _sne_jacobian_approx(vars, c, r, active_pairs, active_walls):
    """
    Approximate Jacobian for the SNE system.
    """
    n_vars = 2*N + 1
    n_eq = len(active_pairs) + len(active_walls)
    if n_eq == 0:
        return np.zeros((1, n_vars))
    
    J = np.zeros((n_eq, n_vars))
    
    eq_idx = 0
    for (i, j) in active_pairs:
        diff = c[i] - c[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-12:
            # d(dist)/d(c_i) = diff/dist, d(dist)/d(c_j) = -diff/dist
            J[eq_idx, 2*i] = diff[0] / dist
            J[eq_idx, 2*i+1] = diff[1] / dist
            J[eq_idx, 2*j] = -diff[0] / dist
            J[eq_idx, 2*j+1] = -diff[1] / dist
            J[eq_idx, 2*N] = -2.0  # d(2r)/dr
        eq_idx += 1
    
    for (i, w) in active_walls:
        # Wall 0: c_x + HALF, wall 1: HALF - c_x, wall 2: c_y + HALF, wall 3: HALF - c_y
        if w == 0:
            J[eq_idx, 2*i] = 1.0
        elif w == 1:
            J[eq_idx, 2*i] = -1.0
        elif w == 2:
            J[eq_idx, 2*i+1] = 1.0
        else:
            J[eq_idx, 2*i+1] = -1.0
        J[eq_idx, 2*N] = -1.0  # d(r)/dr
        eq_idx += 1
    
    return J


def _solve_sne(c_init, r_init, max_iter=50):
    """
    Solve the SNE system starting from initial positions and radius.
    Returns improved (c, r) or None if it fails.
    """
    vars_init = np.concatenate([c_init.ravel(), [r_init]])
    
    def residual_func(v):
        c_v = v[:2*N].reshape(N, 2)
        r_v = v[2*N]
        return _sne_objective(v)
    
    try:
        sol, info, ier, mesg = fsolve(
            residual_func, vars_init,
            full_output=True,
            maxfev=max_iter * (2*N + 1)
        )
        
        if ier != 1:
            return None
        
        c_sol = sol[:2*N].reshape(N, 2)
        r_sol = sol[2*N]
        
        # Verify solution
        if not np.all(np.isfinite(c_sol)) or r_sol <= 0:
            return None
        
        # Check if solution is valid
        c_sol = np.clip(c_sol, -HALF + 1e-9, HALF - 1e-9)
        score = _score_from_centres(c_sol)
        
        if score > r_sol * 0.95:  # Solution is close to optimal
            return c_sol, score
        else:
            return None
    except Exception:
        return None


def _polish_with_slsqp(c_init, r_init):
    """
    Polish using SLSQP with explicit constraints for active contacts.
    """
    c = c_init.copy()
    r = r_init
    
    # Find active constraints
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    
    tol = 0.002
    active_pairs = [(i, j) for i in range(N) for j in range(i+1, N) if d[i,j] < 2*r + tol]
    active_walls = [(i, w) for i in range(N) for w in range(4) if wall_dists[i, w] < r + tol]
    
    # Objective: maximize r subject to constraints
    def objective(vars):
        c_v = vars[:2*N].reshape(N, 2)
        r_v = vars[2*N]
        # Penalize violations
        d_v = cdist(c_v, c_v)
        np.fill_diagonal(d_v, np.inf)
        
        pair_viol = np.maximum(0, 2*r_v - d_v)
        wall_d_v = np.stack([
            c_v[:, 0] + HALF,
            HALF - c_v[:, 0],
            c_v[:, 1] + HALF,
            HALF - c_v[:, 1]
        ], axis=1)
        wall_viol = np.maximum(0, r_v - wall_d_v)
        
        penalty = np.sum(pair_viol**2) + np.sum(wall_viol**2)
        return -r_v + 100.0 * penalty
    
    # Constraints: for each active pair, |c_i - c_j| >= 2r
    # For each active wall, wall_dist >= r
    
    constraints = []
    for (i, j) in active_pairs:
        def make_pair_constr(i, j):
            def pair_constr(vars):
                c_v = vars[:2*N].reshape(N, 2)
                r_v = vars[2*N]
                return np.linalg.norm(c_v[i] - c_v[j]) - 2*r_v
            return pair_constr
        constraints.append({'type': 'ineq', 'fun': make_pair_constr(i, j)})
    
    for (i, w) in active_walls:
        def make_wall_constr(i, w):
            def wall_constr(vars):
                c_v = vars[:2*N].reshape(N, 2)
                r_v = vars[2*N]
                if w == 0:
                    return c_v[i, 0] + HALF - r_v
                elif w == 1:
                    return HALF - c_v[i, 0] - r_v
                elif w == 2:
                    return c_v[i, 1] + HALF - r_v
                else:
                    return HALF - c_v[i, 1] - r_v
            return wall_constr
        constraints.append({'type': 'ineq', 'fun': make_wall_constr(i, w)})
    
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2*N) + [(0.01, 0.5)]
    vars_init = np.concatenate([c.ravel(), [r]])
    
    try:
        res = minimize(
            objective, vars_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        
        c_res = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
        r_res = res.x[2*N]
        score = _score_from_centres(c_res)
        
        if score > r:
            return c_res, score
        else:
            return None
    except Exception:
        return None


def construct():
    t0 = time.time()
    best_centres = _grid_init()
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)
    
    # Try hexagonal initialization too
    hex_centres = _hex_init()
    hex_score = _score_from_centres(hex_centres)
    if hex_score > best_score:
        best_score = hex_score
        best_centres = hex_centres
    
    # Strategy 1: Multi-start with SNE + SLSQP polish
    n_starts = 10
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:
            break
        
        if i == 0:
            c0 = best_centres.copy()
        elif i == 1:
            c0 = _hex_init().copy()
        else:
            c0 = _random_init(rng)
            c0 += rng.normal(0, 0.01, size=c0.shape)
        
        c0 = np.clip(c0, -HALF + 1e-9, HALF - 1e-9)
        r0 = _score_from_centres(c0)
        
        # Try SNE approach
        if time.time() - t0 < DEADLINE - 35:
            sne_result = _solve_sne(c0, r0, max_iter=30)
            if sne_result is not None:
                c_sne, s_sne = sne_result
                if s_sne > best_score:
                    best_score = s_sne
                    best_centres = c_sne
        
        # Try SLSQP polish
        if time.time() - t0 < DEADLINE - 25:
            slsqp_result = _polish_with_slsqp(c0, r0)
            if slsqp_result is not None:
                c_sl, s_sl = slsqp_result
                if s_sl > best_score:
                    best_score = s_sl
                    best_centres = c_sl
    
    # Strategy 2: Final polish on best solution
    if time.time() - t0 < DEADLINE - 15:
        final_result = _polish_with_slsqp(best_centres, best_score)
        if final_result is not None:
            c_f, s_f = final_result
            if s_f > best_score:
                best_score = s_f
                best_centres = c_f
    
    # Strategy 3: SNE on best
    if time.time() - t0 < DEADLINE - 5:
        sne_final = _solve_sne(best_centres, best_score, max_iter=50)
        if sne_final is not None:
            c_sf, s_sf = sne_final
            if s_sf > best_score:
                best_score = s_sf
                best_centres = c_sf
    
    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
# EVOLVE-BLOCK-END
