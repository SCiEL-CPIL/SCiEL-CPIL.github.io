"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill
_t0 = [0.0]  # Mutable container for start time


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


def _identify_active_set(c, r_target, tol=1e-6):
    """Identify pairs and wall constraints that are active (tight) at radius r_target."""
    from scipy.spatial import cKDTree
    
    tree = cKDTree(c)
    # Find all pairs with distance < 2*r_target + tol
    pairs = tree.query_pairs(2 * r_target + tol)
    
    # Wall constraints: circles within r_target + tol of a wall
    active_walls = []
    for i in range(N):
        if c[i, 0] + HALF < r_target + tol:
            active_walls.append((i, 0))  # left wall
        if HALF - c[i, 0] < r_target + tol:
            active_walls.append((i, 1))  # right wall
        if c[i, 1] + HALF < r_target + tol:
            active_walls.append((i, 2))  # bottom wall
        if HALF - c[i, 1] < r_target + tol:
            active_walls.append((i, 3))  # top wall
    
    return pairs, active_walls


def _active_set_slsqp(c0, r_target, pairs, active_walls, max_iter=300):
    """Solve a constrained problem: maximize r subject to active constraints.
    
    Variables: x = centres.ravel() (2N variables) + r (1 variable) = 2N+1 total
    Objective: minimize -r (i.e., maximize r)
    Constraints: 
      - For each active pair (i,j): ||c_i - c_j|| >= 2*r
      - For each active wall (i, w): distance from c_i to wall >= r
      - Bounds: -0.5 <= c_ik <= 0.5
    """
    t0 = _t0[0]
    from scipy.optimize import minimize
    
    n_vars = 2 * N + 1  # 2N for centres + 1 for r
    
    def objective(x):
        # x = [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
        return -x[-1]  # minimize -r = maximize r
    
    # Inequality constraints: g(x) >= 0
    constraints = []
    
    # Pair constraints: ||c_i - c_j||^2 >= (2*r)^2
    # Equivalent: ||c_i - c_j||^2 - 4*r^2 >= 0
    for (i, j) in pairs:
        def pair_constraint(x, i=i, j=j):
            ci = x[2*i:2*i+2]
            cj = x[2*j:2*j+2]
            r = x[-1]
            return np.dot(ci - cj, ci - cj) - 4.0 * r * r
        constraints.append({'type': 'ineq', 'fun': pair_constraint})
    
    # Wall constraints
    for (i, w) in active_walls:
        if w == 0:  # left: c_i[0] + 0.5 >= r
            def wall_constraint(x, i=i):
                return x[2*i] + HALF - x[-1]
            constraints.append({'type': 'ineq', 'fun': wall_constraint})
        elif w == 1:  # right: 0.5 - c_i[0] >= r
            def wall_constraint(x, i=i):
                return HALF - x[2*i] - x[-1]
            constraints.append({'type': 'ineq', 'fun': wall_constraint})
        elif w == 2:  # bottom: c_i[1] + 0.5 >= r
            def wall_constraint(x, i=i):
                return x[2*i+1] + HALF - x[-1]
            constraints.append({'type': 'ineq', 'fun': wall_constraint})
        elif w == 3:  # top: 0.5 - c_i[1] >= r
            def wall_constraint(x, i=i):
                return HALF - x[2*i+1] - x[-1]
            constraints.append({'type': 'ineq', 'fun': wall_constraint})
    
    # Bounds
    bounds = [(-HALF, HALF)] * (2 * N) + [(1e-8, r_target * 1.05)]
    
    # Initial guess
    x0 = np.concatenate([c0.ravel(), [r_target]])
    
    res = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'iprint': 0}
    )
    
    if res.success or res.fun < -r_target:
        c_new = res.x[:2*N].reshape(N, 2)
        r_new = res.x[-1]
        return c_new, r_new
    
    return c0, r_target


def _kdtree_coord_descent(c0, r_target, n_iter=2000, tol=1e-10):
    """Coordinate descent polish using KD-tree for nearest neighbor queries.
    
    For each point, try to move it in the direction that increases the minimum
    distance to its nearest neighbors and walls.
    """
    t0 = _t0[0]
    from scipy.spatial import cKDTree
    
    c = c0.copy()
    current_r = _score_from_centres(c)
    
    for it in range(n_iter):
        if time.time() - t0 > DEADLINE - 5:
            break
            
        # Build KD-tree
        tree = cKDTree(c)
        
        # For each point, find its closest neighbor and closest wall
        improved = False
        
        for i in range(N):
            # Find closest other point
            dist, idx = tree.query(c[i], k=2)
            nearest_idx = idx[1]
            nearest_dist = dist[1]
            
            # Find closest wall distance
            wall_dists = np.array([
                c[i, 0] + HALF,
                HALF - c[i, 0],
                c[i, 1] + HALF,
                HALF - c[i, 1]
            ])
            min_wall_dist = wall_dists.min()
            
            # The limiting constraint for point i
            if nearest_dist / 2.0 < min_wall_dist:
                # Limited by neighbor
                direction = c[i] - c[nearest_idx]
                norm = np.linalg.norm(direction)
                if norm < 1e-15:
                    continue
                direction /= norm
                # Move in direction away from neighbor
                step = (nearest_dist / 2.0 - current_r) * 0.5 + 1e-6
                c_new = c.copy()
                c_new[i] += direction * min(step, 0.01)
                c_new[i] = np.clip(c_new[i], -HALF + 1e-10, HALF - 1e-10)
            else:
                # Limited by wall
                # Determine which wall
                wall_idx = wall_dists.argmin()
                if wall_idx == 0:  # left
                    direction = np.array([1.0, 0.0])
                elif wall_idx == 1:  # right
                    direction = np.array([-1.0, 0.0])
                elif wall_idx == 2:  # bottom
                    direction = np.array([0.0, 1.0])
                else:  # top
                    direction = np.array([0.0, -1.0])
                step = (min_wall_dist - current_r) * 0.5 + 1e-6
                c_new = c.copy()
                c_new[i] += direction * min(step, 0.01)
                c_new[i] = np.clip(c_new[i], -HALF + 1e-10, HALF - 1e-10)
            
            new_r = _score_from_centres(c_new)
            if new_r > current_r + tol:
                c = c_new
                current_r = new_r
                improved = True
        
        if not improved:
            # Try smaller steps
            for scale in [0.5, 0.1, 0.01]:
                improved = False
                for i in range(N):
                    dist, idx = tree.query(c[i], k=2)
                    nearest_idx = idx[1]
                    nearest_dist = dist[1]
                    
                    wall_dists = np.array([
                        c[i, 0] + HALF,
                        HALF - c[i, 0],
                        c[i, 1] + HALF,
                        HALF - c[i, 1]
                    ])
                    min_wall_dist = wall_dists.min()
                    
                    if nearest_dist / 2.0 < min_wall_dist:
                        direction = c[i] - c[nearest_idx]
                        norm = np.linalg.norm(direction)
                        if norm < 1e-15:
                            continue
                        direction /= norm
                        step = scale * 0.001
                        c_new = c.copy()
                        c_new[i] += direction * step
                        c_new[i] = np.clip(c_new[i], -HALF + 1e-10, HALF - 1e-10)
                    else:
                        wall_idx = wall_dists.argmin()
                        if wall_idx == 0:
                            direction = np.array([1.0, 0.0])
                        elif wall_idx == 1:
                            direction = np.array([-1.0, 0.0])
                        elif wall_idx == 2:
                            direction = np.array([0.0, 1.0])
                        else:
                            direction = np.array([0.0, -1.0])
                        step = scale * 0.001
                        c_new = c.copy()
                        c_new[i] += direction * step
                        c_new[i] = np.clip(c_new[i], -HALF + 1e-10, HALF - 1e-10)
                    
                    new_r = _score_from_centres(c_new)
                    if new_r > current_r + tol:
                        c = c_new
                        current_r = new_r
                        improved = True
                
                if improved:
                    break
            else:
                break
    
    return c, current_r


def _smooth_score(c_flat, beta=50.0):
    """Smooth surrogate for the radius score using log-sum-exp.
    
    Instead of min(distances), we use a soft-min that is differentiable.
    This allows gradient-based optimizers to navigate the landscape smoothly.
    """
    c = c_flat.reshape(N, 2)
    # Clip to avoid boundary issues in gradients, though bounds handle this
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances
    d = pdist(c)
    if d.size == 0:
        return 0.0
    
    # Wall distances for each point
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    # Soft-min of pairwise distances (each pair contributes distance/2 to radius)
    # Use log-sum-exp trick for numerical stability
    d_half = d / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    # Soft-min of wall distances
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    # Combine: we want min(soft_min_pair, soft_min_wall)
    # Use another soft-min
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat, beta=50.0):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat, beta)


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    # Hexagonal close packing for better initial configuration
    # 8 cols, 7 rows approx
    rows = int(np.ceil(N / 8))
    cols = 8
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2) * 0.22
            y = (i - (rows - 1) / 2) * 0.2
            if i % 2 == 1:
                x += 0.11
            points.append([x, y])
    return np.array(points[:N])


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    _t0[0] = t0
    
    # Try multiple initial configurations
    inits = [_grid_init(), _hex_init()]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    rng = np.random.default_rng(42)

    # Strategy 1: Local optimisation from multiple starts with smooth objective
    # Start with the better initial config
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    # Use adaptive beta: start low for smooth exploration, increase for refinement
    n_starts = 25
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        elif i == 2:
            x0 = _grid_init().ravel()
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)

        try:
            # Multi-stage optimization:
            # Stage 1: Low beta (smooth) to find basin
            # Stage 2: Medium beta to refine
            # Stage 3: High beta to lock in
            betas = [10.0, 30.0, 100.0]
            current_x = x0
            
            for beta in betas:
                if time.time() - t0 > DEADLINE - 35:
                    break
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), current_x,
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'iprint': 0}
                )
                current_x = np.clip(res.x, -HALF + 1e-9, HALF - 1e-9)
                c = current_x.reshape(N, 2)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
        except Exception:
            pass

    # Strategy 2: Simulated Annealing with targeted local moves
    if time.time() - t0 < DEADLINE - 25:
        def _sa_optimize(c0, t_max=0.01, t_min=1e-5, n_iter=5000, rng_sa=None):
            """Simulated annealing with local refinement."""
            if rng_sa is None:
                rng_sa = np.random.default_rng()
            
            c = c0.copy()
            current_score = _score_from_centres(c)
            best_c = c.copy()
            best_s = current_score
            
            for it in range(n_iter):
                if time.time() - t0 > DEADLINE - 20:
                    break
                
                # Temperature schedule
                T = t_max * (t_min / t_max) ** (it / n_iter)
                
                # Generate move: perturb a random subset of points
                n_move = rng_sa.integers(1, 6)
                idx = rng_sa.choice(N, size=n_move, replace=False)
                
                c_new = c.copy()
                # Scale perturbation with temperature
                scale = 0.02 + 0.08 * (T / t_max)
                delta = rng_sa.normal(0, scale, size=(n_move, 2))
                c_new[idx] += delta
                c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
                
                new_score = _score_from_centres(c_new)
                delta_s = new_score - current_score
                
                # Accept if better or with probability exp(-delta_s/T) for worse
                if delta_s > 0 or (T > 0 and rng_sa.random() < np.exp(max(-100, delta_s / T))):
                    c = c_new
                    current_score = new_score
                    if current_score > best_s:
                        best_s = current_score
                        best_c = c.copy()
                
                # Every 50 iterations, do a local refinement on best with high beta
                if it % 50 == 0 and it > 0:
                    try:
                        # Use high beta for refinement to sharpen the minimum
                        res = minimize(
                            lambda cf: _negative_score(cf, 100.0), best_c.ravel(),
                            method='L-BFGS-B',
                            bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                            options={'maxiter': 200, 'ftol': 1e-13, 'iprint': 0}
                        )
                        c_refined = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                        s_refined = _score_from_centres(c_refined)
                        if s_refined > best_s:
                            best_s = s_refined
                            best_c = c_refined
                            c = best_c.copy()
                            current_score = best_s
                    except Exception:
                        pass
            
            return best_c, best_s
        
        sa_c, sa_s = _sa_optimize(best_centres, t_max=0.015, t_min=1e-6, n_iter=6000, rng_sa=rng)
        if sa_s > best_score:
            best_score = sa_s
            best_centres = sa_c

    # Strategy 3: Active-set SLSQP - directly maximize r with hard constraints
    if time.time() - t0 < DEADLINE - 15:
        try:
            r_current = best_score
            # Identify active constraints
            pairs, active_walls = _identify_active_set(best_centres, r_current, tol=1e-5)
            
            if len(pairs) > 0 or len(active_walls) > 0:
                c_new, r_new = _active_set_slsqp(best_centres, r_current, pairs, active_walls, max_iter=500)
                if r_new > best_score:
                    best_score = r_new
                    best_centres = c_new
        except Exception:
            pass

    # Strategy 4: KD-tree coordinate descent polish
    if time.time() - t0 < DEADLINE - 5:
        try:
            c_cd, r_cd = _kdtree_coord_descent(best_centres, best_score, n_iter=3000)
            if r_cd > best_score:
                best_score = r_cd
                best_centres = c_cd
        except Exception:
            pass

    # Strategy 5: Final active-set refinement with tighter tolerance
    if time.time() - t0 < DEADLINE - 2:
        try:
            r_current = best_score
            pairs, active_walls = _identify_active_set(best_centres, r_current, tol=1e-7)
            
            if len(pairs) > 0 or len(active_walls) > 0:
                c_new, r_new = _active_set_slsqp(best_centres, r_current, pairs, active_walls, max_iter=200)
                if r_new > best_score:
                    best_score = r_new
                    best_centres = c_new
        except Exception:
            pass

    # Strategy 6: Final KD-tree polish with very small steps
    if time.time() - t0 < DEADLINE - 0.5:
        try:
            c_cd, r_cd = _kdtree_coord_descent(best_centres, best_score, n_iter=500, tol=1e-12)
            if r_cd > best_score:
                best_score = r_cd
                best_centres = c_cd
        except Exception:
            pass

    # Ensure valid output
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
